# SPDX-License-Identifier: MIT
"""pytest fixtures for USAM unit + integration tests.

The fixtures here fall into three groups:

1. **Tiny dataset roots** — ``tiny_droid_root``, ``tiny_agibot_dataset``,
   ``tiny_robomind_dataset``. If the LFS-materialized fixture under
   ``tests/golden_data/tiny_<source>`` is missing, the matching
   ``_synthesize_<source>`` helper materializes it on-the-fly. The
   synthesizers use seeded RNG so re-runs are byte-identical.

2. **Mock VLM components** — ``mock_conductor`` and ``mock_player``. These
   wrap a randomly-initialized ``MockConductorBackbone`` and a tiny
   tensor-shape player so integration tests don't have to load Qwen3-VL or
   the 350M-param smoke MM-DiT.

3. **GPU gates** — ``cuda_or_skip`` and ``cuda_8gpu_or_skip``. These skip the
   surrounding test if the matching CUDA topology is not available; CI
   already filters with ``-m "not gpu_1 and not gpu_8"`` so the gates are
   secondary defence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_ROOT = _REPO_ROOT / "tests" / "golden_data"
_TINY_DROID = _GOLDEN_ROOT / "tiny_droid"
_TINY_AGIBOT = _GOLDEN_ROOT / "tiny_agibot"
_TINY_ROBOMIND = _GOLDEN_ROOT / "tiny_robomind"


# Ensure repo root is on sys.path so `from usam... import ...` works without an
# editable install. The pyproject ships a `setuptools` build but we don't want
# to require `pip install -e .` to run tests.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _real_fixture_present(root: Path) -> bool:
    """True only if a materialized fixture under ``root`` is non-empty."""
    if not root.exists():
        return False
    info = root / "meta" / "info.json"
    return info.exists() and info.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tiny dataset roots
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_droid_root() -> Path:
    """Path to a working ``tiny_droid`` fixture.

    Materializes a synthetic stand-in if the LFS fixture is missing.
    """
    if _real_fixture_present(_TINY_DROID):
        return _TINY_DROID

    # Synthesize. Import is lazy so a broken synthesizer doesn't break
    # collection of unrelated tests.
    from tests.golden_data._synthesize_tiny_droid import synthesize_tiny_droid

    return synthesize_tiny_droid(_TINY_DROID)


@pytest.fixture(scope="session")
def tiny_agibot_dataset() -> Path:
    """Path to a working ``tiny_agibot`` fixture.

    Materializes via :func:`tests.golden_data._synthesize_agibot.synthesize_tiny_agibot`
    when the LFS fixture is missing. The synthetic AgiBot fixture exercises
    the multi-camera (head + wrist) and multi-level instruction paths.
    """
    if _real_fixture_present(_TINY_AGIBOT):
        return _TINY_AGIBOT

    from tests.golden_data._synthesize_agibot import synthesize_tiny_agibot

    return synthesize_tiny_agibot(_TINY_AGIBOT)


@pytest.fixture(scope="session")
def tiny_robomind_dataset() -> Path:
    """Path to a working ``tiny_robomind`` fixture (Tien Kung embodiment).

    Materializes via :func:`tests.golden_data._synthesize_robomind.synthesize_tiny_robomind`
    when the LFS fixture is missing.
    """
    if _real_fixture_present(_TINY_ROBOMIND):
        return _TINY_ROBOMIND

    from tests.golden_data._synthesize_robomind import synthesize_tiny_robomind

    return synthesize_tiny_robomind(_TINY_ROBOMIND)


# ---------------------------------------------------------------------------
# Mock VLM components
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_conductor() -> Any:
    """A tiny :class:`usam.conductor.conductor.Conductor` for unit tests.

    Wraps a 64-dim :class:`~usam.conductor.conductor.MockConductorBackbone`
    so no Qwen3-VL checkpoint is loaded. Returns a *fresh* instance per
    test (function scope) — the conductor mutates its committed embedding
    cache, so we don't share it across tests.
    """
    from usam.conductor.conductor import Conductor, MockConductorBackbone

    n_plan = 32
    backbone = MockConductorBackbone(hidden_size=64, seq_len=n_plan + 4)
    return Conductor(
        qwen_ckpt="",
        n_plan_tokens=n_plan,
        player_d_model=64,
        e_proj_dim=64,
        backbone_override=backbone,
        backbone_hidden=64,
        backbone_seq_len=n_plan + 4,
    )


@pytest.fixture
def mock_player() -> Callable[..., Any]:
    """A toy "player" callable matching the :class:`RealtimeController` contract.

    The signature is ``(rgb_dino, depth_dino, proprio,
    plan_cache, n_steps) -> Tensor[B, 16, 7]``. The tensor body is derived
    from the inputs so smoke tests can verify wiring without sourcing a real
    diffusion player. Reads the layer-0 image-branch K/V to exercise the
    cache path.
    """
    import torch
    from torch import Tensor

    def _player(
        rgb_dino: Tensor,
        depth_dino: Tensor | None,
        proprio: Tensor,
        plan_cache: Any,
        n_steps: int,
    ) -> Tensor:
        b = rgb_dino.shape[0]
        k, v = plan_cache.get(0, branch="image")
        out = (
            rgb_dino[:, 0, :].mean(dim=-1, keepdim=True)
            + proprio.mean(dim=-1, keepdim=True)
            + k.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            + v.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        )
        return out.unsqueeze(-1).expand(b, 16, 7).contiguous()

    return _player


# ---------------------------------------------------------------------------
# GPU gates
# ---------------------------------------------------------------------------
@pytest.fixture
def cuda_or_skip() -> str:
    """Return ``"cuda"`` if a GPU is visible; otherwise skip the test.

    Use as ``def test_x(cuda_or_skip): model.to(cuda_or_skip)``. Tests that
    rely on this should also be marked ``@pytest.mark.gpu_1`` so the CI
    filter excludes them on CPU-only runners.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    return "cuda"


@pytest.fixture
def cuda_8gpu_or_skip() -> str:
    """Return ``"cuda"`` only when ≥ 8 GPUs are visible; otherwise skip.

    Used by the multi-GPU smoke train. The companion marker is
    ``@pytest.mark.gpu_8`` (already declared in ``pyproject.toml``).
    """
    import torch

    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 CUDA GPUs")
    return "cuda"
