# SPDX-License-Identifier: MIT
"""USAM Conductor package — slow language/vision encoder + Plan-KV-Cache.

Public API:

* :class:`Conductor` — wraps Qwen3-VL-4B, emits ``(e, P_hat)``.
* :class:`ConductorOutput` — dataclass returned by :meth:`Conductor.encode`.
* :class:`MockConductorBackbone` — tiny stand-in for unit tests.
* :class:`PlanCache` / :class:`PlanCacheState` — pre-projected K, V cache.
* :class:`FDriftMLP`, :class:`DriftConfig`, :func:`should_refresh`,
  :func:`calibrate_taus`, :func:`cosine_distance` — drift trigger.
* :class:`SubtaskCompletionHead` — subtask-boundary classifier.
* :func:`apply_cache_dropout` — training-time stale-plan dropout helper.
"""
from __future__ import annotations

from usam.conductor.cache_dropout import apply_cache_dropout
from usam.conductor.classifier import SubtaskCompletionHead
from usam.conductor.conductor import (
    Conductor,
    ConductorOutput,
    MockConductorBackbone,
)
from usam.conductor.drift import (
    DriftConfig,
    FDriftMLP,
    calibrate_taus,
    cosine_distance,
    should_refresh,
)
from usam.conductor.plan_cache import PlanCache, PlanCacheState

__all__ = [
    "Conductor",
    "ConductorOutput",
    "MockConductorBackbone",
    "PlanCache",
    "PlanCacheState",
    "FDriftMLP",
    "DriftConfig",
    "should_refresh",
    "calibrate_taus",
    "cosine_distance",
    "SubtaskCompletionHead",
    "apply_cache_dropout",
]
