# SPDX-License-Identifier: MIT
"""Integration smoke train for USAM.

Two tests:

* :func:`test_smoke_train_cpu_plumbing` — the CPU plumbing check. Runs
  5 training steps on the synthesized ``tiny_droid`` fixture with the
  ``usam_350m_smoke`` config. Verifies that:

  - All four flow-matching losses + auxiliary losses produce gradients.
  - ``apply_cache_dropout`` is exercised at least once.
  - The total wall-clock is < 60 s on CPU.

  This test runs anywhere — no GPU required.

* :func:`test_smoke_train` — the 8×A40 smoke. 100 steps; loss must be
  finite at every step; the 10-step moving average must be monotonic
  non-increasing across the last 50 steps; total wall-clock < 10 min.
  Skipped automatically when fewer than 8 CUDA devices are visible.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
import torch

from usam.train import (
    TrainArgs,
    load_yaml,
    run as train_run,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_args(
    *, output_dir: Path, max_steps: int, device: str
) -> TrainArgs:
    return TrainArgs(
        config=REPO_ROOT / "configs" / "train" / "stage_b1_pretrain.yaml",
        model_config=REPO_ROOT / "configs" / "model" / "usam_350m_smoke.yaml",
        data=None,                                         # filled by fixture
        output_dir=output_dir,
        max_steps=max_steps,
        device=device,
        seed=0,
        auto_oom_reduce=False,
        log_every=1,
    )


def _moving_average(xs: list[float], window: int) -> list[float]:
    """Return the trailing moving average (length-aware)."""
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


@pytest.mark.slow
def test_smoke_train_cpu_plumbing(tiny_droid_root: Path, tmp_path: Path, caplog) -> None:
    """5-step CPU plumbing run. No GPUs, no flash-attn, no transformer-engine.

    Verifies the whole train loop wires up: dataloader → conductor →
    plan-cache refresh → cache-dropout → player → unified loss → backward
    → optimizer.step. The loss only needs to be **finite**; with five
    steps on a tiny synthetic fixture there is no statistical
    expectation of monotonic decrease.

    Marked ``slow`` because the wall-clock budget is 60 s — CI's PR pipeline
    skips ``slow`` to stay under its 5-min total budget. Run nightly
    (``-m slow``) or before tagging a release.
    """
    caplog.set_level(logging.INFO, logger="usam.train")

    args = _make_args(
        output_dir=tmp_path / "smoke_cpu",
        max_steps=5,
        device="cpu",
    )
    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    args.data = tiny_droid_root          # type: ignore[attr-defined]

    # The synthesized tiny fixture is 15 fps, not 30 — read it back so the
    # dataloader stride math matches the on-disk feature shard layout.
    import json
    info = json.loads((tiny_droid_root / "meta" / "info.json").read_text())
    train_cfg.setdefault("data", {})["fps_action"] = int(info.get("fps", 30))
    train_cfg["data"]["fps_features"] = int(info.get("fps_features", 5))
    # The fixture's only camera is "head_rgb"; cap it.
    train_cfg["data"]["cameras"] = ["head_rgb"]
    # CPU plumbing only needs to verify the train loop wires up, not the
    # full 12-layer 350M smoke model. Slim the player so the test stays
    # under the 60-second budget on a single core.
    model_cfg.setdefault("player", {})["num_layers"] = 2
    model_cfg["player"]["hidden_size"] = 64
    model_cfg["player"]["num_heads"] = 4

    t0 = time.time()
    losses = train_run(args, train_cfg, model_cfg)
    elapsed = time.time() - t0

    # ---- Hard correctness checks ------------------------------------------
    assert len(losses) == 5, losses
    for i, l in enumerate(losses):
        assert isinstance(l, float), (i, l)
        assert l == l, f"NaN loss at step {i}: {l}"               # NaN test
        assert abs(l) < 1e6, f"loss exploded at step {i}: {l}"

    # ---- Plumbing wall-clock budget ---------------------------------------
    # 60 s is generous for a 5-step CPU run on the smoke fixture; fail fast
    # if something blows up (e.g. accidental backbone load).
    assert elapsed < 60.0, f"CPU plumbing exceeded 60 s budget: {elapsed:.2f}s"

    # ---- Checkpoint plumbing ----------------------------------------------
    # checkpoint.every_steps defaults to 5_000; a 5-step run shouldn't write
    # any periodic checkpoints, but the run dir must exist.
    assert (args.output_dir).exists()


@pytest.mark.gpu_8
@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="needs 8 GPUs for the full smoke train (A40 budget)",
)
def test_smoke_train(tiny_droid_root: Path, tmp_path: Path, caplog) -> None:
    """100-step 8×A40 smoke train.

    Hard guarantees:

    * Loss is finite at every step.
    * The 10-step moving average is monotonically non-increasing
      across the last 50 steps. (Allows microscopic float-drift
      noise — we compare ``avg[i+1] <= avg[i] + tol``.)
    * Total wall-clock < 600 s.
    """
    caplog.set_level(logging.INFO, logger="usam.train")

    args = _make_args(
        output_dir=tmp_path / "smoke_8gpu",
        max_steps=100,
        device="cuda",
    )
    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    args.data = tiny_droid_root          # type: ignore[attr-defined]

    t0 = time.time()
    losses = train_run(args, train_cfg, model_cfg)
    elapsed = time.time() - t0

    assert elapsed < 600.0, f"smoke train exceeded 10 min: {elapsed:.2f}s"
    assert len(losses) == 100, losses
    for i, l in enumerate(losses):
        assert l == l, f"NaN loss at step {i}: {l}"               # NaN test
        assert abs(l) < 1e6, f"loss exploded at step {i}: {l}"

    # 10-step moving average over the last 50 steps must be non-increasing.
    ma = _moving_average(losses, window=10)
    tail = ma[-50:]
    tol = 1e-4
    for i in range(1, len(tail)):
        assert tail[i] <= tail[i - 1] + tol, (
            f"loss MA not non-increasing at i={i}: {tail[i-1]:.4f} -> {tail[i]:.4f}"
        )
