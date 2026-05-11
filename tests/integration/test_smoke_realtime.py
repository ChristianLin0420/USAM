# SPDX-License-Identifier: MIT
"""Smoke test for the real-time control loop.

Runs 100 control steps against a mocked Player and a synthetic drift
sequence. Asserts:

* The loop completes without raising.
* The cumulative cache-refresh count matches the hand-traced value.

Hand trace
----------
``d_t`` sequence: ``[0.0]*30 + [0.6]*1 + [0.0]*30 + [0.4]*39`` (100 steps).
Config: ``tau_hard=0.5``, ``tau_soft=0.3``, ``timer_soft=10``,
``timer_hard=50``.

* t=0  → episode-start REFRESH (count=1, last=0).
* t=1..29 (d_t=0.0): below all thresholds; elapsed < 50 → no.
* t=30 (d_t=0.6 > 0.5): hard breach → REFRESH (count=2, last=30).
* t=31..60 (d_t=0.0): elapsed up to 30 < 50 → no.
* t=61 (d_t=0.4): elapsed=31 ≥ timer_soft=10 and d_t > tau_soft=0.3 →
  REFRESH (count=3, last=61).
* t=62..70: elapsed 1..9 < 10 → no.
* t=71: elapsed=10 → REFRESH (count=4, last=71).
* t=72..80: elapsed 1..9 → no.
* t=81: elapsed=10 → REFRESH (count=5, last=81).
* t=82..90: elapsed 1..9 → no.
* t=91: elapsed=10 → REFRESH (count=6, last=91).
* t=92..99: elapsed 1..8 → no.

Expected total refreshes = **6**.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from usam.conductor.conductor import Conductor, MockConductorBackbone
from usam.conductor.drift import DriftConfig, FDriftMLP
from usam.conductor.plan_cache import PlanCache
from usam.inference.realtime import RealtimeController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_player(
    rgb_dino: Tensor,
    depth_dino: Tensor | None,
    proprio: Tensor,
    plan_cache,
    n_steps: int,
) -> Tensor:
    """Toy player: returns a ``[B, 16, 7]`` tensor derived from the inputs."""
    b = rgb_dino.shape[0]
    # Read at least one cache entry to exercise the cache path.
    k, v = plan_cache.get(0, branch="image")
    out = (rgb_dino[:, 0, :].mean(dim=-1, keepdim=True)
           + proprio.mean(dim=-1, keepdim=True)
           + k.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
           + v.mean(dim=(1, 2), keepdim=False).unsqueeze(-1))
    return out.unsqueeze(-1).expand(b, 16, 7).contiguous()


def _build_controller(
    n_layers: int = 2,
    d_model: int = 64,
    n_plan: int = 32,
    drift_cfg: DriftConfig | None = None,
) -> RealtimeController:
    backbone = MockConductorBackbone(hidden_size=64, seq_len=n_plan + 4)
    conductor = Conductor(
        qwen_ckpt="",
        n_plan_tokens=n_plan,
        player_d_model=d_model,
        e_proj_dim=64,
        backbone_override=backbone,
        backbone_hidden=64,
        backbone_seq_len=n_plan + 4,
    )

    f_drift = FDriftMLP(rgb_dino_dim=64, e_dim=64, hidden=32)
    plan_cache = PlanCache(n_layers=n_layers, d_model=d_model, n_plan=n_plan, dtype=torch.float32)

    k_projs_image = nn.ModuleList(
        [nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)]
    )
    v_projs_image = nn.ModuleList(
        [nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)]
    )

    drift_cfg = drift_cfg or DriftConfig(
        tau_hard=0.5, tau_soft=0.3, timer_hard=50, timer_soft=10
    )

    controller = RealtimeController(
        conductor=conductor,
        f_drift=f_drift,
        plan_cache=plan_cache,
        drift_config=drift_cfg,
        player=_mock_player,
        k_projs_image=k_projs_image,
        v_projs_image=v_projs_image,
        k_projs_action=None,
        v_projs_action=None,
        tri_dino=None,  # we feed pre-encoded features
        n_denoise_steps=2,
        subtask_head=None,
    )
    return controller


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_realtime_runs_100_steps_without_error() -> None:
    """Smoke: 100 steps, every step returns a finite tensor."""
    torch.manual_seed(0)
    controller = _build_controller()
    controller.reset(instruction="pick up the cup")

    b = 1
    rgb = torch.randn(b, 3, 224, 224)
    proprio = torch.randn(b, 50)
    rgb_tokens = torch.randn(b, 65, 64)  # CLS + 64 patch tokens

    for t in range(100):
        result = controller.step(
            rgb=rgb,
            depth=None,
            proprio=proprio,
            instruction="pick up the cup",
            override_features={"rgb": rgb_tokens},
            force_drift_d=0.0,
        )
        assert torch.isfinite(result.action_chunk).all()
        assert result.action_chunk.shape == (b, 16, 7)
    # With d_t=0 every step and a generous timer, only the episode-start
    # refresh should fire; timer_hard=50 means we'll see at least one
    # additional timer-driven refresh between steps 50 and 99.
    assert controller._refresh_count >= 1


def test_realtime_refresh_count_matches_handtrace() -> None:
    """Synthetic ``d_t`` sequence → expected ``cache_refresh_count == 6``."""
    torch.manual_seed(1)
    cfg = DriftConfig(tau_hard=0.5, tau_soft=0.3, timer_hard=50, timer_soft=10)
    controller = _build_controller(drift_cfg=cfg)
    controller.reset(instruction="open the drawer")

    d_seq = [0.0] * 30 + [0.6] * 1 + [0.0] * 30 + [0.4] * 39
    assert len(d_seq) == 100

    b = 1
    rgb = torch.randn(b, 3, 224, 224)
    proprio = torch.randn(b, 50)
    rgb_tokens = torch.randn(b, 65, 64)

    for t, d in enumerate(d_seq):
        result = controller.step(
            rgb=rgb,
            depth=None,
            proprio=proprio,
            instruction="open the drawer",
            override_features={"rgb": rgb_tokens},
            force_drift_d=d,
        )
        assert torch.isfinite(result.action_chunk).all()

    # Hand-traced expected count = 6 (see module docstring).
    assert controller._refresh_count == 6, (
        f"expected 6 refreshes, got {controller._refresh_count}"
    )


def test_realtime_refresh_at_episode_start() -> None:
    """First step always triggers a refresh."""
    controller = _build_controller()
    controller.reset()
    rgb = torch.randn(1, 3, 224, 224)
    proprio = torch.randn(1, 50)
    rgb_tokens = torch.randn(1, 65, 64)
    result = controller.step(
        rgb=rgb,
        depth=None,
        proprio=proprio,
        override_features={"rgb": rgb_tokens},
        force_drift_d=0.0,
    )
    assert result.refreshed is True
    assert result.refresh_count == 1
