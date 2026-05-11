# SPDX-License-Identifier: MIT
"""Real-time inference loop for the slow Conductor / fast Player split.

This module implements §4.4 of ``docs/IMPLEMENTATION_PLAN.md``: a
single-step ``act(observation)`` API that

1. encodes the current observation through the Tri-DINO tower (cheap),
2. predicts an estimate of the next [EOS] embedding using
   :class:`usam.conductor.drift.FDriftMLP` (cheaper),
3. asks :func:`usam.conductor.drift.should_refresh` whether the
   committed plan is stale,
4. if so, re-runs the full Conductor (Qwen3-VL-4B) and refreshes the
   :class:`PlanCache` once,
5. denoises an action chunk through the Player MM-DiT, reading the
   pre-projected K, V from the cache.

Real Player wiring
------------------
The actual Player MM-DiT consumes the cache via a ``kv_cache=`` kwarg
in its ``forward``; see ``lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py``.
The :class:`RealtimeController` here is *agnostic* to that signature —
it treats the player as a callable
``player(rgb_dino, depth_dino, proprio, plan_cache,
n_steps) -> action_chunk``. Concrete plumbing of the cache through
each MM-DiT block is the model-architect's responsibility.

LoRA routing concurrency
------------------------
:class:`usam.encoders.tri_dino.TriDINOTower` routes modality through
a stateful per-instance attribute on each LoRALinear. **Do not call
``forward(modality_a)`` and ``forward(modality_b)`` concurrently on
the same encoder.** This loop calls one modality at a time so it is
safe; document this for the training-engineer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from usam.conductor.classifier import SubtaskCompletionHead
from usam.conductor.conductor import Conductor, ConductorOutput
from usam.conductor.drift import (
    DriftConfig,
    FDriftMLP,
    cosine_distance,
    should_refresh,
)
from usam.conductor.plan_cache import PlanCache


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """One control step's output.

    Parameters
    ----------
    action_chunk : Tensor
        Whatever the Player returned, typically ``[B, A, action_dim]``.
    refreshed : bool
        ``True`` if the Conductor was re-run on this step.
    d_t : float
        Cosine drift distance computed on this step.
    refresh_count : int
        Total cumulative refreshes since loop start.
    """

    action_chunk: Tensor
    refreshed: bool
    d_t: float
    refresh_count: int


# ---------------------------------------------------------------------------
# Player protocol
# ---------------------------------------------------------------------------
PlayerCallable = Callable[..., Tensor]
"""Signature: ``player(rgb_dino, depth_dino, proprio, plan_cache, n_steps) -> Tensor``."""


# ---------------------------------------------------------------------------
# Tri-DINO callable protocol
# ---------------------------------------------------------------------------
TriDinoCallable = Callable[[Tensor, str], Tensor]
"""Signature: ``tri_dino(x, modality) -> Tensor[B, N_tokens, D]``."""


# ---------------------------------------------------------------------------
# Realtime controller
# ---------------------------------------------------------------------------
class RealtimeController:
    """Single-step real-time control loop wiring all the pieces.

    Parameters
    ----------
    conductor : Conductor
        Wrapped Qwen3-VL-4B (or :class:`MockConductorBackbone` in tests).
    f_drift : FDriftMLP
        Cheap drift predictor.
    plan_cache : PlanCache
        Pre-projected K, V cache.
    drift_config : DriftConfig
        Trigger thresholds.
    tri_dino : callable, optional
        ``tri_dino(x, modality) -> tokens``. When ``None``, the
        controller skips visual encoding entirely (caller must pass
        encoded features into :meth:`step`). The realtime loop uses the
        full encoder; the smoke test passes pre-encoded tensors via
        ``override_features`` to avoid wiring DINOv3 weights.
    player : callable
        Callable returning the action chunk. See :data:`PlayerCallable`.
    k_projs_image, v_projs_image : sequence of nn.Linear
        Player layer cross-attn projections. Forwarded to
        :meth:`PlanCache.refresh`.
    k_projs_action, v_projs_action : sequence of nn.Linear or None
        Same for the action branch.
    n_denoise_steps : int
        Denoising steps per control tick (default 10).
    subtask_head : SubtaskCompletionHead or None
        Optional classifier; when provided we read its logit and feed
        it into :func:`should_refresh`.
    history_window : int
        Frames the subtask head sees (default 16).
    obs_dim, proprio_dim : int
        Dims for the subtask head's history buffers.
    """

    def __init__(
        self,
        conductor: Conductor,
        f_drift: FDriftMLP,
        plan_cache: PlanCache,
        drift_config: DriftConfig,
        player: PlayerCallable,
        k_projs_image: Sequence[nn.Linear],
        v_projs_image: Sequence[nn.Linear],
        k_projs_action: Sequence[nn.Linear] | None = None,
        v_projs_action: Sequence[nn.Linear] | None = None,
        tri_dino: TriDinoCallable | None = None,
        n_denoise_steps: int = 10,
        subtask_head: SubtaskCompletionHead | None = None,
        history_window: int = 16,
        obs_dim: int = 768,
        proprio_dim: int = 50,
    ) -> None:
        assert n_denoise_steps > 0, f"n_denoise_steps must be positive"
        assert history_window > 0, f"history_window must be positive"

        self.conductor = conductor
        self.f_drift = f_drift
        self.plan_cache = plan_cache
        self.drift_config = drift_config
        self.tri_dino = tri_dino
        self.player = player
        self.k_projs_image = list(k_projs_image)
        self.v_projs_image = list(v_projs_image)
        self.k_projs_action = list(k_projs_action) if k_projs_action is not None else None
        self.v_projs_action = list(v_projs_action) if v_projs_action is not None else None
        self.n_denoise_steps = int(n_denoise_steps)
        self.subtask_head = subtask_head
        self.history_window = int(history_window)
        self.obs_dim = int(obs_dim)
        self.proprio_dim = int(proprio_dim)

        # Mutable per-episode state.
        self._t: int = 0
        self._last_refresh_t: int = -1
        self._refresh_count: int = 0
        self._instruction: str | None = None
        self._obs_history: deque[Tensor] = deque(maxlen=self.history_window)
        self._proprio_history: deque[Tensor] = deque(maxlen=self.history_window)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, instruction: str | None = None) -> None:
        """Reset per-episode state. Forces a refresh on the next :meth:`step`."""
        self._t = 0
        self._last_refresh_t = -1
        self._refresh_count = 0
        self._instruction = instruction
        self._obs_history.clear()
        self._proprio_history.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_visual(
        self,
        rgb: Tensor,
        depth: Tensor | None,
        override: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        """Run Tri-DINO on each modality (or use overrides)."""
        out: dict[str, Tensor] = {}
        if override is not None:
            out.update(override)
            if "rgb" not in out:
                assert self.tri_dino is not None, "tri_dino encoder required for missing rgb"
                out["rgb"] = self.tri_dino(rgb, "rgb")
            return out
        assert self.tri_dino is not None, (
            "tri_dino encoder is None; either provide one or pass override_features"
        )
        out["rgb"] = self.tri_dino(rgb, "rgb")
        if depth is not None:
            out["depth"] = self.tri_dino(depth, "depth")
        return out

    def _refresh(self, observation: dict[str, Tensor], instruction: str | None) -> ConductorOutput:
        """Run the Conductor and refresh the PlanCache."""
        out = self.conductor.encode(observation, instruction)
        self.plan_cache.refresh(
            p_hat=out.P_hat,
            e=out.e,
            k_projs_image=self.k_projs_image,
            v_projs_image=self.v_projs_image,
            k_projs_action=self.k_projs_action,
            v_projs_action=self.v_projs_action,
            t=self._t,
        )
        self._last_refresh_t = self._t
        self._refresh_count += 1
        return out

    def _subtask_logit(
        self,
        e_proj: Tensor,
        rgb_cls: Tensor,
        proprio: Tensor,
    ) -> float:
        """Return the subtask-completion logit (or ``-inf`` if no head)."""
        if self.subtask_head is None:
            return float("-inf")
        # Pad histories to ``history_window`` if we are early in the episode.
        obs_hist = list(self._obs_history) + [rgb_cls]
        prop_hist = list(self._proprio_history) + [proprio]
        while len(obs_hist) < self.history_window:
            obs_hist.insert(0, rgb_cls)
        while len(prop_hist) < self.history_window:
            prop_hist.insert(0, proprio)
        obs_hist = obs_hist[-self.history_window :]
        prop_hist = prop_hist[-self.history_window :]
        window_obs = torch.stack(obs_hist, dim=1)  # [B, W, D]
        window_prop = torch.stack(prop_hist, dim=1)
        logit = self.subtask_head(e_proj, window_obs, window_prop)
        return float(logit.detach().squeeze().item())

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        rgb: Tensor,
        depth: Tensor | None = None,
        *,
        proprio: Tensor,
        instruction: str | None = None,
        override_features: dict[str, Tensor] | None = None,
        force_drift_d: float | None = None,
    ) -> StepResult:
        """Run one control step.

        Pseudocode (§4.4 of the implementation plan)::

            rgb_dino   = dinov3.rgb(obs.rgb)
            depth_dino = dinov3.depth(obs.depth)
            e_now_est  = f_drift(rgb_dino_cls, e_committed)
            d_t        = 1 - cos(e_committed, e_now_est)
            if should_refresh(...):
                e, P_hat = conductor(instruction, obs.rgb)
                plan_cache.refresh(P_hat, e, k_projs, v_projs, t)
            action = player.denoise(rgb_dino, depth_dino,
                                    proprio, plan_cache, n_steps=10)
            yield action

        Parameters
        ----------
        rgb : Tensor
            ``[B, 3, H, W]`` raw RGB. Used by both Tri-DINO and the
            Conductor.
        depth : Tensor or None
            Optional auxiliary modality for Tri-DINO.
        proprio : Tensor
            ``[B, proprio_dim]`` proprio.
        instruction : str or None
            Per-step instruction; falls back to :meth:`reset`'s value.
        override_features : dict, optional
            Pre-encoded ``{"rgb": ..., "depth": ...}``.
            Used by the smoke test and by training-time pipelines that
            already cache DINO features.
        force_drift_d : float or None
            Test hook: if provided, override the computed ``d_t``. Used
            by :file:`tests/integration/test_smoke_realtime.py` to feed
            a synthetic drift sequence.

        Returns
        -------
        StepResult
            See :class:`StepResult`.
        """
        instruction = instruction or self._instruction

        feats = self._encode_visual(rgb, depth, override_features)
        rgb_tokens = feats["rgb"]
        # rgb_dino_cls = the [CLS] token (token 0).
        rgb_cls = rgb_tokens[:, 0, :]

        # Drift estimation requires a committed plan. On step 0 we
        # short-circuit and force-refresh.
        if not self.plan_cache.is_valid():
            d_t = float("inf")
            episode_start = True
        else:
            e_committed = self.plan_cache.committed_emb  # [B, D_e_proj]
            e_now_est = self.f_drift(rgb_cls, e_committed)
            d_t = float(cosine_distance(e_committed, e_now_est).mean().item())
            episode_start = self._t == 0

        if force_drift_d is not None:
            d_t = float(force_drift_d)

        # Subtask logit lookup. We need a current ``e_proj`` for the
        # head; if the cache is invalid the committed_emb attribute is
        # unset, so we skip and use -inf to suppress that trigger.
        subtask_logit = float("-inf")
        if self.plan_cache.is_valid():
            subtask_logit = self._subtask_logit(
                e_proj=self.plan_cache.committed_emb,
                rgb_cls=rgb_cls,
                proprio=proprio,
            )

        refresh = should_refresh(
            t=self._t,
            d_t=d_t,
            last_refresh_t=self._last_refresh_t,
            config=self.drift_config,
            subtask_completion_logit=subtask_logit,
            episode_start=episode_start,
        )
        refreshed_now = False
        if refresh:
            self._refresh({"rgb": rgb}, instruction)
            refreshed_now = True

        # Update history *after* the refresh decision so the current
        # frame contributes to the next step's window.
        self._obs_history.append(rgb_cls)
        self._proprio_history.append(proprio)

        # Player. The smoke test substitutes a mock player; production
        # wiring binds to the real MM-DiT.
        action = self.player(
            rgb_dino=rgb_tokens,
            depth_dino=feats.get("depth"),
            proprio=proprio,
            plan_cache=self.plan_cache,
            n_steps=self.n_denoise_steps,
        )

        result = StepResult(
            action_chunk=action,
            refreshed=refreshed_now,
            d_t=d_t,
            refresh_count=self._refresh_count,
        )
        self._t += 1
        return result


__all__ = ["RealtimeController", "StepResult"]
