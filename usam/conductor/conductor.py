# SPDX-License-Identifier: MIT
"""Conductor — slow language/vision encoder around Qwen3-VL-4B.

The Conductor's job is to map a single ``(instruction, keyframe_rgb)``
pair to two artifacts the Player consumes:

* ``e``: the L2-normalized [EOS] hidden state, used as a low-dim drift
  reference (``[B, 1, D_e]``).
* ``P_hat``: the last ``n_plan_tokens`` of the VLM hidden state,
  projected into the Player's ``d_model`` ("plan tokens", shape
  ``[B, n_plan_tokens, d_model]``).

The real-time loop runs the Conductor only when
:func:`usam.conductor.drift.should_refresh` fires; the rest of the time
the Player consumes a cached projection of ``P_hat`` (see
:class:`usam.conductor.plan_cache.PlanCache`).

LDA-1B reuse
------------
Loading Qwen3-VL-4B is delegated to LDA's
``lda.model.modules.vlm._QWen3_VL_Interface`` (the same wrapper used at
training time). We do not refork this code — the Conductor just *uses*
it. For unit tests we accept a ``backbone_override`` argument and ship
a tiny :class:`MockConductorBackbone` that mirrors the expected
interface (output dim + ability to return a hidden-state tensor of
shape ``[B, T, D]``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConductorOutput:
    """Bundle returned by :meth:`Conductor.encode`.

    Parameters
    ----------
    e : Tensor
        ``[B, 1, D_e]`` L2-normalized [EOS] embedding *projected* into
        the dim consumed by :class:`usam.conductor.drift.FDriftMLP`.
        Re-normalized after projection.
    P_hat : Tensor
        ``[B, n_plan_tokens, d_model]`` plan tokens projected into the
        Player's hidden dim.
    e_raw : Tensor
        ``[B, 1, hidden_size]`` raw [EOS] hidden state (un-projected).
        Useful for downstream introspection / debugging; the realtime
        loop ignores this.
    """

    e: Tensor
    P_hat: Tensor
    e_raw: Tensor


# ---------------------------------------------------------------------------
# Mock backbone for unit testing
# ---------------------------------------------------------------------------
class MockConductorBackbone(nn.Module):
    """Tiny stand-in for Qwen3-VL-4B used in unit tests.

    Mirrors the expected Conductor surface: a forward that takes
    pre-tokenized inputs (we pretend) and returns a hidden state tensor
    of shape ``[B, T, hidden_size]``. Using a small ``hidden_size``
    keeps the smoke tests fast.

    Parameters
    ----------
    hidden_size : int
        Hidden dim. Defaults to 64 — small enough to keep the f_drift
        MLP under its 100K-param budget without an additional projection.
    seq_len : int
        Sequence length emitted. Must be ``>= n_plan_tokens + 1`` so
        we can extract both the plan tokens and a separate [EOS] token.
    """

    def __init__(self, hidden_size: int = 64, seq_len: int = 48) -> None:
        super().__init__()
        assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        self.hidden_size = int(hidden_size)
        self.seq_len = int(seq_len)
        # A tiny ViT-ish trunk: (img -> tokens) + a few attention-free MLP blocks.
        self.tok_embed = nn.Linear(3, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        # Position embeddings give us a deterministic dependence on token index.
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_size))

    def forward(self, pixel_values: Tensor, **_kwargs: Any) -> Tensor:
        """Return ``[B, seq_len, hidden_size]`` hidden states.

        ``pixel_values`` is ``[B, 3, H, W]``; we flatten to a [B, ...]
        token bag and project. The exact pattern is irrelevant — we
        just need a deterministic, differentiable function from pixels
        to hidden states for the Conductor wrapper to consume.
        """
        b, c, h, w = pixel_values.shape
        # Compress to seq_len tokens via mean pooling over disjoint
        # H*W // seq_len chunks — deterministic, batch-independent.
        x = pixel_values.permute(0, 2, 3, 1).reshape(b, h * w, c)
        chunk = max(1, x.shape[1] // self.seq_len)
        # Use ``.reshape`` (not ``.view``) since slicing may produce a
        # non-contiguous tensor depending on the underlying layout.
        x = x[:, : chunk * self.seq_len, :].reshape(b, self.seq_len, chunk, c).mean(dim=2)
        x = self.tok_embed(x)
        x = x + self.pos
        x = self.norm(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Conductor wrapper
# ---------------------------------------------------------------------------
class Conductor(nn.Module):
    """Wraps a frozen VLM and emits ``(e, P_hat)`` for the Player.

    Parameters
    ----------
    qwen_ckpt : str
        Path/handle for the underlying Qwen3-VL-4B checkpoint. Only
        consulted when ``backbone_override is None``. Pass ``""`` plus
        a ``backbone_override`` for unit tests.
    n_plan_tokens : int
        Number of plan tokens (default 32, matching the implementation
        plan §11.3).
    player_d_model : int
        Hidden dim of the Player MM-DiT. ``P_hat`` is projected into
        this dim.
    e_proj_dim : int
        Dim of the small [EOS] projection consumed by
        :class:`usam.conductor.drift.FDriftMLP`. Default 64 keeps the
        drift MLP under the 100K-param budget.
    backbone_override : nn.Module or None
        Pre-built backbone object. Required for unit tests.
    backbone_hidden : int or None
        Hidden dim of the backbone. If ``None``, we ask the override
        for ``.hidden_size``. Required if no override is given.
    backbone_seq_len : int or None
        Sequence length of the backbone output. Used to assert that
        ``n_plan_tokens + 1 <= seq_len``. Optional; defaults to whatever
        the override reports.

    Notes
    -----
    Loading the real Qwen3-VL-4B is gated on
    ``backbone_override is None``. We import LDA's wrapper lazily so
    the unit tests don't pull in ``transformers`` or ``flash_attn``.
    """

    def __init__(
        self,
        qwen_ckpt: str = "",
        n_plan_tokens: int = 32,
        player_d_model: int = 2048,
        e_proj_dim: int = 64,
        backbone_override: nn.Module | None = None,
        backbone_hidden: int | None = None,
        backbone_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        assert n_plan_tokens > 0, f"n_plan_tokens must be positive, got {n_plan_tokens}"
        assert player_d_model > 0, f"player_d_model must be positive, got {player_d_model}"
        assert e_proj_dim > 0, f"e_proj_dim must be positive, got {e_proj_dim}"

        self.n_plan_tokens = int(n_plan_tokens)
        self.player_d_model = int(player_d_model)
        self.e_proj_dim = int(e_proj_dim)

        if backbone_override is not None:
            self.backbone = backbone_override
            hidden = backbone_hidden or getattr(backbone_override, "hidden_size", None)
            seq_len = backbone_seq_len or getattr(backbone_override, "seq_len", None)
        else:
            self.backbone = self._load_qwen3vl(qwen_ckpt)
            hidden = backbone_hidden or getattr(self.backbone, "hidden_size", None)
            if hidden is None:
                cfg = getattr(self.backbone, "config", None)
                hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
            seq_len = backbone_seq_len

        assert hidden is not None, (
            "Could not infer backbone hidden_size; pass `backbone_hidden=...`."
        )
        self.backbone_hidden = int(hidden)
        if seq_len is not None:
            assert seq_len >= self.n_plan_tokens + 1, (
                f"backbone seq_len {seq_len} must be >= n_plan_tokens + 1 "
                f"({self.n_plan_tokens + 1})"
            )

        # Freeze the VLM. The plan calls for a fully frozen Conductor.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Trainable projections live outside the frozen backbone.
        self.plan_proj = nn.Linear(self.backbone_hidden, self.player_d_model)
        self.e_proj = nn.Linear(self.backbone_hidden, self.e_proj_dim)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_qwen3vl(ckpt: str) -> nn.Module:
        """Load Qwen3-VL-4B via LDA's interface. Imported lazily."""
        # Avoid importing at module top so unit tests don't need transformers.
        from omegaconf import OmegaConf  # type: ignore

        from lda.model.modules.vlm.QWen3 import _QWen3_VL_Interface  # type: ignore

        cfg = OmegaConf.create({
            "framework": {"qwenvl": {"base_vlm": ckpt or "Qwen/Qwen3-VL-4B-Instruct"}},
            "datasets": {"vla_data": {}},
        })
        return _QWen3_VL_Interface(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _run_backbone(
        self,
        observation: dict[str, Tensor] | Tensor,
        instruction: str | list[str] | None,
    ) -> Tensor:
        """Run whatever underlying VLM is wired in and return ``[B, T, D]``.

        The mock backbone takes ``pixel_values`` directly. The real
        Qwen3-VL-4B wrapper (``_QWen3_VL_Interface``) expects a fully
        constructed ``BatchFeature`` from
        ``build_qwenvl_inputs``. We dispatch on the backbone surface.
        """
        if isinstance(observation, dict):
            # `or` would call __bool__ on a multi-element tensor (ambiguous);
            # use explicit key fallback instead.
            if "rgb" in observation:
                pixel_values = observation["rgb"]
            else:
                pixel_values = observation.get("pixel_values")
            assert pixel_values is not None, (
                "observation dict must contain 'rgb' or 'pixel_values'"
            )
        else:
            pixel_values = observation

        # Mock backbone path: just call forward(pixel_values=...).
        if isinstance(self.backbone, MockConductorBackbone):
            return self.backbone(pixel_values)

        # Real Qwen3-VL path. We delegate input construction to the
        # wrapper because tokenization is messy and stateful.
        if hasattr(self.backbone, "build_qwenvl_inputs"):
            assert instruction is not None, (
                "Real Qwen3-VL Conductor requires an instruction string"
            )
            instr_list = [instruction] if isinstance(instruction, str) else list(instruction)
            # build_qwenvl_inputs expects a list-of-list-of-PIL-images.
            # We pass the raw pixel tensor via a single-image list per
            # sample; downstream callers in `realtime.py` are expected
            # to pre-format. For now we surface a clear error so the
            # team lead wires the formatting to match training.
            raise NotImplementedError(
                "Real Qwen3-VL forward path is not exercised by unit tests; "
                "wire image preprocessing in usam/inference/realtime.py before use."
            )

        # Generic fallback: assume a callable that takes pixel_values.
        return self.backbone(pixel_values=pixel_values)

    @torch.no_grad()
    def encode(
        self,
        observation: dict[str, Tensor] | Tensor,
        instruction: str | list[str] | None = None,
    ) -> ConductorOutput:
        """Encode one ``(observation, instruction)`` pair into ``(e, P_hat)``.

        Parameters
        ----------
        observation : dict or Tensor
            Either a dict containing key ``"rgb"`` / ``"pixel_values"``
            with an RGB tensor ``[B, 3, H, W]``, or the tensor itself.
        instruction : str or list[str] or None
            Natural-language instruction string(s). Required when using
            the real Qwen3-VL backbone; ignored by the mock.

        Returns
        -------
        ConductorOutput
            ``e``: ``[B, 1, D_e]`` L2-normalized.
            ``P_hat``: ``[B, n_plan_tokens, player_d_model]``.
            ``e_proj``: ``[B, e_proj_dim]`` (for the drift MLP).
        """
        hidden = self._run_backbone(observation, instruction)
        assert hidden.dim() == 3, (
            f"backbone must return [B, T, D], got {tuple(hidden.shape)}"
        )
        b, t, d = hidden.shape
        assert d == self.backbone_hidden, (
            f"backbone hidden dim mismatch: cfg={self.backbone_hidden}, runtime={d}"
        )
        assert t >= self.n_plan_tokens + 1, (
            f"backbone seq_len {t} must be >= n_plan_tokens + 1 "
            f"({self.n_plan_tokens + 1})"
        )

        # Plan tokens: the **last** n_plan_tokens. Convention from
        # implementation plan §4.2.
        plan_tokens = hidden[:, -self.n_plan_tokens :, :]
        # [EOS] = the very last token. We project to the f_drift dim
        # then re-L2-normalize so cosine distance is well-defined.
        eos_raw = hidden[:, -1:, :]  # [B, 1, hidden_size]
        eos_proj = self.e_proj(eos_raw)  # [B, 1, e_proj_dim]
        e = F.normalize(eos_proj, dim=-1, eps=1e-8)

        # Project plan tokens to the Player's d_model.
        P_hat = self.plan_proj(plan_tokens)

        return ConductorOutput(e=e, P_hat=P_hat, e_raw=eos_raw)


__all__ = ["Conductor", "ConductorOutput", "MockConductorBackbone"]
