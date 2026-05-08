# SPDX-License-Identifier: MIT
"""Plan-KV-Cache for the slow Conductor / fast Player split.

The Conductor (Qwen3-VL-4B) periodically emits ``P_hat: [B, n_plan, D]`` —
a small bank of "plan tokens" summarizing the current language + visual
context. The Player MM-DiT cross-attends to this plan from many layers.

Naively, every Player forward would recompute ``K = k_proj(P_hat)`` and
``V = v_proj(P_hat)`` for every layer, every diffusion step, every
control step. That's wasted FLOPs because ``P_hat`` only changes when the
drift trigger fires (≈once a second; see :mod:`usam.conductor.drift`).

This module pre-projects ``P_hat`` through all Player layers' cross-attn
``k_proj`` / ``v_proj`` linears once at refresh time. The Player's
cross-attention reads directly from the cache — no per-step
re-projection.

Contract for the Player MM-DiT
------------------------------
The Player block consumes the cache via a ``kv_cache=`` kwarg. The cache
must expose two pieces of information per cross-attn site (image branch
and action branch in MM-DiT):

* ``k_image[L]`` and ``v_image[L]`` — pre-projected through the L-th
  layer's ``img_cross_attn.to_k`` / ``to_v``.
* ``k_action[L]`` and ``v_action[L]`` — pre-projected through the L-th
  layer's ``action_cross_attn.to_k`` / ``to_v``.

If a layer has only one cross-attention site (e.g. unified image+action
attention), pass a single ``k_projs`` / ``v_projs`` list and read the
``image`` slot.

Bit-exact equivalence
---------------------
Calling cross-attention with the cached K/V vs. recomputing them from
``P_hat`` must yield bit-exact outputs at fp32 (and ≤ 1e-3 absolute
error at fp16). The unit test :func:`tests.unit.test_plan_cache` enforces
this directly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import torch
from torch import Tensor, nn


@dataclass
class PlanCacheState:
    """Snapshot of a refreshed plan cache (for cache-dropout sampling).

    The real :class:`PlanCache` mutates in-place on every :meth:`refresh`.
    For training-time cache dropout we sometimes want a *stale* snapshot
    of an earlier refresh. :class:`PlanCacheState` is the immutable
    record of one refresh; clone it via :meth:`PlanCache.snapshot`.
    """

    k_image: tuple[Tensor, ...]
    v_image: tuple[Tensor, ...]
    k_action: tuple[Tensor, ...]
    v_action: tuple[Tensor, ...]
    committed_emb: Tensor
    refresh_t: int


class PlanCache:
    """Holds pre-projected K, V for the Conductor's plan tokens.

    Parameters
    ----------
    n_layers : int
        Number of Player MM-DiT blocks. The cache stores one (K, V) pair
        per layer per cross-attention site (image branch and action
        branch).
    d_model : int
        Player hidden dim. ``P_hat`` arrives as ``[B, n_plan, d_model]``
        and is projected to ``[B, n_plan, d_model]`` per site.
    n_plan : int
        Number of plan tokens emitted by the Conductor (default 32).
    dtype : torch.dtype
        Storage dtype for the cached K/V. ``torch.bfloat16`` matches the
        Player's compute dtype on H200/A100.

    Notes
    -----
    * The cache is *unbatched* in the sense that ``B`` is whatever batch
      shape ``P_hat`` is refreshed with. Real-time inference uses
      ``B=1``; training uses larger batches.
    * The cache is invalid (``is_valid() is False``) until the first
      :meth:`refresh` call.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_plan: int = 32,
        dtype: torch.dtype = torch.bfloat16,
        history_size: int = 8,
    ) -> None:
        assert n_layers > 0, f"n_layers must be positive, got {n_layers}"
        assert d_model > 0, f"d_model must be positive, got {d_model}"
        assert n_plan > 0, f"n_plan must be positive, got {n_plan}"
        assert history_size >= 0, f"history_size must be non-negative, got {history_size}"

        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.n_plan = int(n_plan)
        self.dtype = dtype

        # Storage. Filled lazily by refresh(); shape determined by P_hat.
        self._k_image: list[Tensor | None] = [None] * n_layers
        self._v_image: list[Tensor | None] = [None] * n_layers
        self._k_action: list[Tensor | None] = [None] * n_layers
        self._v_action: list[Tensor | None] = [None] * n_layers
        self._committed_emb: Tensor | None = None
        self._refresh_t: int = -1
        self._valid: bool = False

        # Bounded history of refreshed snapshots, used by the cache-dropout
        # helper at training time (see usam.conductor.cache_dropout).
        self._history_size = int(history_size)
        self._history: Deque[PlanCacheState] = deque(maxlen=history_size or None)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refresh(
        self,
        p_hat: Tensor,
        e: Tensor,
        k_projs_image: Sequence[nn.Linear],
        v_projs_image: Sequence[nn.Linear],
        k_projs_action: Sequence[nn.Linear] | None,
        v_projs_action: Sequence[nn.Linear] | None,
        t: int,
    ) -> None:
        """Re-project ``p_hat`` through every Player layer's cross-attn K/V.

        Parameters
        ----------
        p_hat : Tensor
            ``[B, n_plan, d_model]`` plan tokens from the Conductor.
        e : Tensor
            ``[B, D_emb]`` (or ``[B, 1, D_emb]``) committed [EOS] embedding.
            Stored verbatim so the drift module can reference it later.
        k_projs_image, v_projs_image : sequence of nn.Linear
            One per Player layer, mapping ``[..., d_model] -> [..., d_model]``.
            Used by the image-branch cross-attention.
        k_projs_action, v_projs_action : sequence of nn.Linear or None
            Same, for the action-branch cross-attention. Pass ``None`` to
            mirror the image-branch projections (rare; mostly for tests).
        t : int
            Current control step. Stored so :meth:`should_refresh` can
            compare against ``last_refresh_t``.

        Raises
        ------
        AssertionError
            If shapes mismatch the configured ``n_plan`` / ``d_model`` /
            ``n_layers``.
        """
        assert p_hat.dim() == 3, f"p_hat must be [B, n_plan, D], got {tuple(p_hat.shape)}"
        b, n_plan, d = p_hat.shape
        assert n_plan == self.n_plan, f"n_plan mismatch: cache={self.n_plan}, p_hat={n_plan}"
        assert d == self.d_model, f"d_model mismatch: cache={self.d_model}, p_hat={d}"
        assert len(k_projs_image) == self.n_layers, (
            f"k_projs_image length mismatch: cache={self.n_layers}, got={len(k_projs_image)}"
        )
        assert len(v_projs_image) == self.n_layers, (
            f"v_projs_image length mismatch: cache={self.n_layers}, got={len(v_projs_image)}"
        )
        if k_projs_action is None or v_projs_action is None:
            k_projs_action = list(k_projs_image)
            v_projs_action = list(v_projs_image)
        assert len(k_projs_action) == self.n_layers, (
            f"k_projs_action length mismatch: cache={self.n_layers}, got={len(k_projs_action)}"
        )
        assert len(v_projs_action) == self.n_layers, (
            f"v_projs_action length mismatch: cache={self.n_layers}, got={len(v_projs_action)}"
        )

        # Project layer-by-layer. We deliberately *do not* batch this with
        # einsum across layers — each layer has its own weight matrix and
        # storage cost is dominated by the cached K/V tensors anyway.
        for layer_idx in range(self.n_layers):
            self._k_image[layer_idx] = k_projs_image[layer_idx](p_hat).to(self.dtype)
            self._v_image[layer_idx] = v_projs_image[layer_idx](p_hat).to(self.dtype)
            self._k_action[layer_idx] = k_projs_action[layer_idx](p_hat).to(self.dtype)
            self._v_action[layer_idx] = v_projs_action[layer_idx](p_hat).to(self.dtype)

        # Store the [EOS] embedding (without re-normalizing) so the drift
        # check has a stable reference. Squeeze a trailing length-1 axis
        # if e arrived as [B, 1, D].
        if e.dim() == 3 and e.shape[1] == 1:
            e = e.squeeze(1)
        assert e.dim() == 2, f"e must be [B, D] or [B, 1, D], got {tuple(e.shape)}"
        self._committed_emb = e.detach().clone()

        self._refresh_t = int(t)
        self._valid = True

        # Push the freshly-installed state into the history ring buffer so
        # that ``apply_cache_dropout`` has somewhere to draw a stale plan
        # from at training time. We snapshot *after* the refresh fields
        # have been committed so the snapshot is internally consistent.
        if self._history_size > 0:
            self._history.append(self.snapshot())

    # ------------------------------------------------------------------
    # History access (for cache-dropout)
    # ------------------------------------------------------------------
    @property
    def history(self) -> tuple[PlanCacheState, ...]:
        """Bounded history of the last ``history_size`` refreshed snapshots."""
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get(self, layer_idx: int, branch: str = "image") -> tuple[Tensor, Tensor]:
        """Return ``(K, V)`` for layer ``layer_idx`` in branch ``branch``.

        Parameters
        ----------
        layer_idx : int
            ``0 <= layer_idx < n_layers``.
        branch : {"image", "action"}
            Which Player cross-attention site to read.

        Returns
        -------
        tuple of Tensor
            Each ``[B, n_plan, d_model]``.
        """
        assert self._valid, "PlanCache.get called before refresh()"
        assert 0 <= layer_idx < self.n_layers, (
            f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
        )
        if branch == "image":
            k = self._k_image[layer_idx]
            v = self._v_image[layer_idx]
        elif branch == "action":
            k = self._k_action[layer_idx]
            v = self._v_action[layer_idx]
        else:
            raise ValueError(f"unknown branch {branch!r}; expected 'image' or 'action'")
        assert k is not None and v is not None, (
            f"cache for layer {layer_idx} branch {branch} is empty"
        )
        return k, v

    def is_valid(self) -> bool:
        """``True`` once :meth:`refresh` has been called at least once."""
        return self._valid

    @property
    def committed_emb(self) -> Tensor:
        """The Conductor's committed [EOS] embedding from the last refresh."""
        assert self._committed_emb is not None, "committed_emb requested before refresh()"
        return self._committed_emb

    @property
    def refresh_t(self) -> int:
        """Control step at the last :meth:`refresh` call. ``-1`` if never refreshed."""
        return self._refresh_t

    # ------------------------------------------------------------------
    # Snapshots (for cache-dropout)
    # ------------------------------------------------------------------
    def snapshot(self) -> PlanCacheState:
        """Return an immutable copy of the current cache contents.

        Used by :func:`usam.conductor.cache_dropout.apply_cache_dropout` to
        keep a window of recent cache states for stale-plan training.
        """
        assert self._valid, "snapshot() called before refresh()"
        assert self._committed_emb is not None
        return PlanCacheState(
            k_image=tuple(t.clone() for t in self._k_image if t is not None),
            v_image=tuple(t.clone() for t in self._v_image if t is not None),
            k_action=tuple(t.clone() for t in self._k_action if t is not None),
            v_action=tuple(t.clone() for t in self._v_action if t is not None),
            committed_emb=self._committed_emb.clone(),
            refresh_t=self._refresh_t,
        )

    def load_state(self, state: PlanCacheState) -> None:
        """Overwrite the cache contents with a previously-captured snapshot.

        Counterpart to :meth:`snapshot`. Used by the cache-dropout helper
        to install a stale plan during training.
        """
        assert len(state.k_image) == self.n_layers, "snapshot layer count mismatch"
        self._k_image = [t.clone() for t in state.k_image]
        self._v_image = [t.clone() for t in state.v_image]
        self._k_action = [t.clone() for t in state.k_action]
        self._v_action = [t.clone() for t in state.v_action]
        self._committed_emb = state.committed_emb.clone()
        self._refresh_t = state.refresh_t
        self._valid = True


__all__ = ["PlanCache", "PlanCacheState"]
