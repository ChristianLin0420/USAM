# SPDX-License-Identifier: MIT
"""Auxiliary supervision heads for USAM.

Public exports:

* :class:`GeomConsistencyLoss` — soft-Spearman depth↔RGB rank consistency.
* :class:`FlowActionConsistencyLoss` — forward-action MLP vs. flow-DINO
  magnitude consistency.
* :func:`soft_rank`, :func:`soft_spearman` — differentiable rank primitives
  (re-exported for testing and for downstream agents that want to reuse
  them).
* :func:`flow_magnitude` — fixed deterministic flow-magnitude decoder.
"""
from __future__ import annotations

from usam.aux_heads.depth_consistency import (
    GeomConsistencyLoss,
    soft_rank,
    soft_spearman,
)
from usam.aux_heads.flow_action import FlowActionConsistencyLoss, flow_magnitude

__all__ = [
    "GeomConsistencyLoss",
    "FlowActionConsistencyLoss",
    "soft_rank",
    "soft_spearman",
    "flow_magnitude",
]
