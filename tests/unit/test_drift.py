# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`usam.conductor.drift`.

One test per refresh trigger:

* Episode start (``t == 0``)
* Timer expiry (``elapsed >= timer_hard``)
* Hard threshold breach (``d_t > tau_hard``)
* Soft+timer combo
* Subtask-completion classifier
* No-trigger baseline

Plus the calibration helper and the ``f_drift`` parameter budget.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from usam.conductor.drift import (
    DriftConfig,
    FDriftMLP,
    calibrate_taus,
    cosine_distance,
    should_refresh,
)


# ---------------------------------------------------------------------------
# FDriftMLP — parameter budget and shapes
# ---------------------------------------------------------------------------
def test_fdrift_param_budget_under_100k() -> None:
    """Charter rule: ``f_drift`` must be cheap (< 100K params)."""
    mlp = FDriftMLP(rgb_dino_dim=768, e_dim=64, hidden=64)
    n = sum(p.numel() for p in mlp.parameters())
    assert n < 100_000, f"FDriftMLP has {n} params, expected < 100_000"


def test_fdrift_forward_shape() -> None:
    mlp = FDriftMLP(rgb_dino_dim=768, e_dim=64, hidden=64)
    rgb = torch.randn(3, 768)
    e = torch.randn(3, 64)
    out = mlp(rgb, e)
    assert out.shape == (3, 64)


def test_fdrift_accepts_squeezed_e() -> None:
    """``e_committed`` arriving as ``[B, 1, D]`` must be auto-squeezed."""
    mlp = FDriftMLP(rgb_dino_dim=64, e_dim=32, hidden=32)
    rgb = torch.randn(2, 64)
    e = torch.randn(2, 1, 32)
    out = mlp(rgb, e)
    assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# should_refresh — one test per trigger
# ---------------------------------------------------------------------------
def test_trigger_episode_start() -> None:
    """``t == 0`` always triggers a refresh."""
    cfg = DriftConfig()
    assert should_refresh(t=0, d_t=0.0, last_refresh_t=-1, config=cfg) is True


def test_trigger_explicit_episode_start() -> None:
    """Explicit ``episode_start=True`` triggers regardless of ``t``."""
    cfg = DriftConfig()
    assert should_refresh(t=42, d_t=0.0, last_refresh_t=10, config=cfg, episode_start=True) is True


def test_trigger_timer_hard_expiry() -> None:
    """``t - last_refresh_t >= timer_hard`` forces a refresh."""
    cfg = DriftConfig(tau_hard=0.5, tau_soft=0.2, timer_hard=60, timer_soft=30)
    assert should_refresh(t=60, d_t=0.0, last_refresh_t=0, config=cfg) is True
    # Just before the timer should NOT fire (no other signal).
    assert should_refresh(t=59, d_t=0.0, last_refresh_t=0, config=cfg) is False


def test_trigger_hard_threshold_breach() -> None:
    """``d_t > tau_hard`` triggers regardless of timers."""
    cfg = DriftConfig(tau_hard=0.20, tau_soft=0.05, timer_hard=60, timer_soft=30)
    # Far below the timer windows but still triggers.
    assert should_refresh(t=5, d_t=0.21, last_refresh_t=4, config=cfg) is True
    # At-threshold does not (strict ``>``).
    assert should_refresh(t=5, d_t=0.20, last_refresh_t=4, config=cfg) is False


def test_trigger_soft_plus_timer_combo() -> None:
    """``d_t > tau_soft`` AND ``elapsed >= timer_soft`` triggers."""
    cfg = DriftConfig(tau_hard=0.50, tau_soft=0.10, timer_hard=60, timer_soft=30)
    # Soft breached but timer too short → no trigger.
    assert should_refresh(t=15, d_t=0.20, last_refresh_t=0, config=cfg) is False
    # Both conditions met → trigger.
    assert should_refresh(t=30, d_t=0.20, last_refresh_t=0, config=cfg) is True


def test_trigger_subtask_completion() -> None:
    """Positive subtask logit triggers refresh."""
    cfg = DriftConfig()
    # No drift, no timer expiry, but classifier says "completed".
    assert (
        should_refresh(
            t=5,
            d_t=0.0,
            last_refresh_t=4,
            config=cfg,
            subtask_completion_logit=0.1,
        )
        is True
    )
    # Negative logit → no trigger.
    assert (
        should_refresh(
            t=5,
            d_t=0.0,
            last_refresh_t=4,
            config=cfg,
            subtask_completion_logit=-0.1,
        )
        is False
    )


def test_no_trigger_baseline() -> None:
    """All conditions below threshold → no refresh."""
    cfg = DriftConfig(tau_hard=0.5, tau_soft=0.2, timer_hard=60, timer_soft=30)
    assert (
        should_refresh(t=5, d_t=0.05, last_refresh_t=4, config=cfg)
        is False
    )


def test_negative_last_refresh_forces_refresh() -> None:
    """``last_refresh_t < 0`` (never refreshed) past t=0 still forces refresh."""
    cfg = DriftConfig()
    assert should_refresh(t=10, d_t=0.0, last_refresh_t=-1, config=cfg) is True


# ---------------------------------------------------------------------------
# Cosine distance helper
# ---------------------------------------------------------------------------
def test_cosine_distance_identity_is_zero() -> None:
    a = torch.randn(4, 8)
    a = F.normalize(a, dim=-1)
    d = cosine_distance(a, a)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_cosine_distance_orthogonal_is_one() -> None:
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    d = cosine_distance(a, b)
    assert torch.allclose(d, torch.tensor([1.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# calibrate_taus — percentiles
# ---------------------------------------------------------------------------
def test_calibrate_taus_percentiles() -> None:
    """``tau_soft`` ≈ P50, ``tau_hard`` ≈ P90 over the drift log."""
    log = [float(x) / 100.0 for x in range(101)]  # 0.00, 0.01, ..., 1.00
    cfg = calibrate_taus(log)
    assert math.isclose(cfg.tau_soft, 0.5, abs_tol=1e-6)
    assert math.isclose(cfg.tau_hard, 0.9, abs_tol=1e-6)


def test_calibrate_taus_singleton() -> None:
    """A 1-element log degenerates to the single value for both thresholds."""
    cfg = calibrate_taus([0.42])
    assert cfg.tau_soft == 0.42
    assert cfg.tau_hard == 0.42


def test_calibrate_taus_rejects_empty() -> None:
    raised = False
    try:
        calibrate_taus([])
    except AssertionError:
        raised = True
    assert raised, "empty drift_log must raise"
