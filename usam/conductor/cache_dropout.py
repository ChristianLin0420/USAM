# SPDX-License-Identifier: MIT
"""Training-time stale-plan dropout for the Plan-KV-Cache.

Contract for the training-engineer
----------------------------------
The Player must learn to be robust against plans that are *up to*
``window`` frames stale, because at inference time the drift trigger
may not fire immediately. We simulate that staleness during training
by occasionally substituting the *live* :class:`PlanCache` with a
*stale* snapshot taken from a uniformly-random earlier timestep within
the last ``window`` control frames.

API
---
``apply_cache_dropout(plan_cache, t, p=0.5, window=60, history=None,
generator=None) -> PlanCache``

* With probability ``1 - p`` (or when no eligible stale snapshot
  exists), returns the live cache unchanged.
* Otherwise picks a uniformly-random snapshot from ``plan_cache.history``
  (or the explicitly-supplied ``history``) whose ``refresh_t`` is in
  ``[t - window, t]`` and installs it onto a shallow clone of the
  cache.

The training loop is responsible for:

1. Letting :class:`PlanCache.refresh` populate the history (it does so
   automatically; ``history_size`` defaults to 8 snapshots).
2. Calling :func:`apply_cache_dropout` once per training step *after*
   the live cache has been refreshed for that step.

This function deliberately does *not* mutate the input cache — it
constructs a new :class:`PlanCache` with the stale state loaded — so
that branching off a stale cache for a single step doesn't poison the
real cache state for subsequent steps.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from usam.conductor.plan_cache import PlanCache, PlanCacheState


def apply_cache_dropout(
    plan_cache: PlanCache,
    t: int,
    p: float = 0.5,
    window: int = 60,
    history: Sequence[PlanCacheState] | None = None,
    generator: torch.Generator | None = None,
) -> PlanCache:
    """Return either the live cache or a uniformly-random stale snapshot.

    Parameters
    ----------
    plan_cache : PlanCache
        The currently-live cache (i.e. the one most recently refreshed
        by the Conductor).
    t : int
        Current control step.
    p : float, optional
        Probability of substituting a stale snapshot (default 0.5,
        matching :ref:`configs/train/stage_b1_pretrain.yaml`).
    window : int, optional
        Maximum staleness in control frames; only snapshots with
        ``t - refresh_t <= window`` are eligible (default 60 = 2 s
        @ 30 Hz).
    history : sequence of PlanCacheState, optional
        Override the cache's built-in history (see
        :attr:`PlanCache.history`). Useful when the training loop
        maintains its own buffer.
    generator : torch.Generator, optional
        Optional RNG for reproducibility. If ``None``, uses the default
        global generator.

    Returns
    -------
    PlanCache
        Either ``plan_cache`` itself (no dropout) or a fresh
        :class:`PlanCache` whose contents are a stale snapshot.
    """
    assert isinstance(plan_cache, PlanCache), (
        f"plan_cache must be PlanCache, got {type(plan_cache)}"
    )
    assert 0.0 <= p <= 1.0, f"p must be in [0, 1], got {p}"
    assert window > 0, f"window must be positive, got {window}"
    assert isinstance(t, int), f"t must be int, got {type(t)}"

    # Coin flip on substitution.
    coin = torch.rand((), generator=generator).item()
    if coin >= p:
        return plan_cache

    # Pull history from the cache if not explicitly supplied.
    if history is None:
        history = plan_cache.history
    # Filter eligible stale states.
    eligible = [s for s in history if (t - s.refresh_t) <= window and s.refresh_t <= t]
    if not eligible:
        return plan_cache

    idx = int(torch.randint(low=0, high=len(eligible), size=(), generator=generator).item())
    chosen = eligible[idx]

    stale = PlanCache(
        n_layers=plan_cache.n_layers,
        d_model=plan_cache.d_model,
        n_plan=plan_cache.n_plan,
        dtype=plan_cache.dtype,
        history_size=0,  # the stale clone shouldn't accumulate history of its own
    )
    stale.load_state(chosen)
    return stale


__all__ = ["apply_cache_dropout"]
