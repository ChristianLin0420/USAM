# SPDX-License-Identifier: MIT
"""Open-loop ADE evaluation loop. **Stub** — filled in by inference-engineer in Wave 4.

Open-loop ADE = "Average Displacement Error" between predicted action
chunks and ground-truth action chunks on a held-out shard. The eval
walks each episode, queries the policy at every supported start
timestamp, and reports average L2 over the chunk horizon.

This file deliberately stops at the API contract — the Wave 4 owner
implements the body once the dataloader API and Player wiring are
finalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class OpenLoopMetrics:
    """Aggregated metrics produced by :func:`run_openloop_eval`.

    Parameters
    ----------
    ade : float
        Average displacement error across all evaluation samples.
    fde : float
        Final displacement error (last-step error).
    n_samples : int
        Number of (episode, start_t) pairs evaluated.
    per_horizon_l2 : list[float]
        Per-step L2 averaged across samples; length ``= action_chunk``.
    """

    ade: float
    fde: float
    n_samples: int
    per_horizon_l2: list[float]


def run_openloop_eval(
    policy: Any,
    dataset: Iterable[dict[str, Any]],
    *,
    action_chunk: int = 16,
    device: str = "cuda",
) -> OpenLoopMetrics:
    """Evaluate ``policy`` open-loop on ``dataset``.

    Parameters
    ----------
    policy : object
        Anything exposing ``predict_action(observation, instruction)
        -> action_chunk: Tensor[B, A, action_dim]``. Concrete wiring
        is the inference-engineer's responsibility in Wave 4.
    dataset : iterable of dict
        Each item must contain ``rgb``, ``proprio``, ``instruction``,
        and ``action_chunk_gt``.
    action_chunk : int, optional
        Expected chunk horizon (default 16).
    device : str, optional
        Compute device.

    Returns
    -------
    OpenLoopMetrics

    Notes
    -----
    Stub — the Wave 4 owner implements the body. The signature is
    fixed so downstream tests can be written against it.
    """
    raise NotImplementedError(
        "run_openloop_eval is a Wave 4 stub. "
        "Implementation owner: inference-engineer. "
        "See docs/IMPLEMENTATION_PLAN.md §11."
    )


__all__ = ["OpenLoopMetrics", "run_openloop_eval"]
