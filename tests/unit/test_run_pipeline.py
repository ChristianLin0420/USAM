# SPDX-License-Identifier: MIT
"""Unit tests for ``prep.run_pipeline``.

We exercise the orchestrator's control flow against monkey-patched stage
callouts so the test stays CPU-only and fast (no TFDS, no DINOv3, no DA3).
The contracts under test are:

* per-chunk ``.pipeline_complete`` marker is written atomically only after
  all 5 stages succeed
* SIGUSR1 between chunks → ``run()`` returns ``PREEMPT_EXIT_CODE`` (124)
* on resume the orchestrator skips chunks whose marker already exists
* a stage_2a return of ``(0, 0)`` terminates the loop with exit 0
* stage_5 validation fails → ``RuntimeError`` propagates and no marker
  is written
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest

from prep._base import PREEMPT_EXIT_CODE
from prep.run_pipeline import (
    DATASETS,
    DATASET_TO_EMBODIMENT,
    PipelineOrchestrator,
)


def _make_orch(tmp_path: Path, dataset: str = "droid") -> PipelineOrchestrator:
    cfg = {
        "fps_native": 15,
        "convert": {"cameras": ["head_rgb"]},
        "depth": {"target_hw": [192, 192], "max_range_mm": 5000, "fp16": True},
        "dino_cache": {"target_hw": [378, 378], "n_keep_tokens": 64, "cache_fps": 5, "fp16": True},
    }
    return PipelineOrchestrator(
        dataset=dataset,
        output_root=tmp_path,
        cfg=cfg,
        cfg_path=None,
        dinov3_ckpt=None,  # placeholder mode
        da3_ckpt=None,
    )


def _patch_stages(monkeypatch, orch: PipelineOrchestrator,
                  s2a_returns: list[tuple[int, int]] | None = None,
                  s5_fails: bool = False) -> dict:
    """Replace each per-stage callout with a recorder.

    ``s2a_returns`` is a list of ``(n_processed, n_skipped)`` tuples the
    fake stage_2a returns on successive invocations. Defaults to
    ``[(5, 0)]`` once then ``(0, 0)`` (i.e. one chunk of work, then EOF).
    """
    calls: dict[str, list[int]] = {
        "stage_2a": [], "stage_2c": [], "stage_3": [],
        "stage_4": [], "stage_5": [],
    }
    if s2a_returns is None:
        s2a_returns = [(5, 0)]

    def fake_2a(chunk):
        calls["stage_2a"].append(int(chunk))
        if s2a_returns:
            return s2a_returns.pop(0)
        return (0, 0)

    def fake_2c(chunk):
        calls["stage_2c"].append(int(chunk))

    def fake_3(chunk):
        calls["stage_3"].append(int(chunk))
        return 5

    def fake_4(chunk):
        calls["stage_4"].append(int(chunk))

    def fake_5(chunk):
        calls["stage_5"].append(int(chunk))
        summary = {"parquet_ok": 1, "parquet_fail": 1 if s5_fails else 0}
        return {"summary": summary, "fails": 1 if s5_fails else 0}

    monkeypatch.setattr(orch, "_run_stage_2a", fake_2a)
    monkeypatch.setattr(orch, "_run_stage_2c", fake_2c)
    monkeypatch.setattr(orch, "_run_stage_3", fake_3)
    monkeypatch.setattr(orch, "_run_stage_4", fake_4)
    monkeypatch.setattr(orch, "_run_stage_5", fake_5)
    return calls


def test_dataset_registry_matches_embodiment_keys() -> None:
    """Every dataset must map to a real embodiment."""
    for ds in DATASETS:
        assert ds in DATASET_TO_EMBODIMENT, f"{ds} missing from embodiment map"


def test_unknown_dataset_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        PipelineOrchestrator(
            dataset="nonexistent", output_root=tmp_path, cfg={},
        )


def test_marker_written_atomically_after_all_stages(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    calls = _patch_stages(monkeypatch, orch, s2a_returns=[(3, 0)])
    rc = orch.run()
    assert rc == 0
    # Stages ran in order, exactly once each for chunk 0.
    assert calls["stage_2a"] == [0, 1]  # 1 returns (0,0) → end
    assert calls["stage_2c"] == [0]
    assert calls["stage_3"] == [0]
    assert calls["stage_4"] == [0]
    assert calls["stage_5"] == [0]
    marker = orch.marker_path(0)
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["dataset"] == "droid"
    assert payload["chunk"] == 0
    assert payload["n_processed"] == 3


def test_zero_episodes_terminates_cleanly(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    _patch_stages(monkeypatch, orch, s2a_returns=[(0, 0)])
    rc = orch.run()
    assert rc == 0
    assert not orch.marker_path(0).exists(), "no marker for an empty chunk"


def test_validation_failure_blocks_marker(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    _patch_stages(monkeypatch, orch, s2a_returns=[(2, 0)], s5_fails=True)
    with pytest.raises(RuntimeError, match="validation reported"):
        orch.run()
    assert not orch.marker_path(0).exists(), "marker must not exist on validation failure"


def test_resume_skips_complete_chunks(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    # Pre-mark chunk 0 as complete.
    orch.write_marker(0, {"dataset": "droid", "chunk": 0, "stub": True})
    calls = _patch_stages(monkeypatch, orch, s2a_returns=[(4, 0)])
    rc = orch.run()
    assert rc == 0
    # Chunk 0 was skipped; chunk 1 processed; chunk 2 returned (0,0) → done.
    assert calls["stage_2a"] == [1, 2]
    assert calls["stage_3"] == [1]


def test_sigusr1_between_chunks_returns_124(tmp_path: Path, monkeypatch) -> None:
    """Send SIGUSR1 to our own process between chunks; run() must exit 124."""
    orch = _make_orch(tmp_path)
    sig_sent = {"v": False}

    def fake_2a(chunk):
        # Send the signal after the first chunk's stage_2a returns. The
        # orchestrator's check between chunks should pick it up.
        if not sig_sent["v"]:
            sig_sent["v"] = True
            return (2, 0)
        return (2, 0)

    def fake_2c(chunk):
        pass

    def fake_3(chunk):
        return 2

    def fake_4(chunk):
        # Fire SIGUSR1 mid-chunk; orchestrator should still finish the chunk
        # and then exit 124 between chunks.
        import os
        os.kill(os.getpid(), signal.SIGUSR1)
        # Give the signal a moment to land before we return.
        time.sleep(0.05)

    def fake_5(chunk):
        return {"summary": {}, "fails": 0}

    monkeypatch.setattr(orch, "_run_stage_2a", fake_2a)
    monkeypatch.setattr(orch, "_run_stage_2c", fake_2c)
    monkeypatch.setattr(orch, "_run_stage_3", fake_3)
    monkeypatch.setattr(orch, "_run_stage_4", fake_4)
    monkeypatch.setattr(orch, "_run_stage_5", fake_5)

    rc = orch.run()
    assert rc == PREEMPT_EXIT_CODE
    # Chunk 0 finished and got its marker before the preempt exit.
    assert orch.marker_path(0).exists()


def test_max_chunks_caps_attempts(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    orch.max_chunks = 2
    calls = _patch_stages(monkeypatch, orch, s2a_returns=[(1, 0), (1, 0), (1, 0)])
    rc = orch.run()
    assert rc == 0
    assert calls["stage_2a"] == [0, 1]  # capped before chunk 2
    assert orch.marker_path(0).exists()
    assert orch.marker_path(1).exists()
    assert not orch.marker_path(2).exists()


def test_start_chunk_skips_lower(tmp_path: Path, monkeypatch) -> None:
    orch = _make_orch(tmp_path)
    orch.start_chunk = 5
    calls = _patch_stages(monkeypatch, orch, s2a_returns=[(1, 0)])
    rc = orch.run()
    assert rc == 0
    assert calls["stage_2a"][0] == 5


def test_marker_write_is_atomic(tmp_path: Path) -> None:
    """Marker is written via tmp + replace; no partial file remains on crash."""
    orch = _make_orch(tmp_path)
    orch.write_marker(7, {"hello": "world"})
    marker = orch.marker_path(7)
    assert marker.exists()
    assert json.loads(marker.read_text())["hello"] == "world"
    # No leftover .tmp file.
    leftover = list(marker.parent.glob(".pipeline_complete.tmp*"))
    assert leftover == [], f"unexpected tmp leftovers: {leftover}"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _populate_chunk_artifacts(orch: PipelineOrchestrator, chunk: int,
                              n_eps: int = 2,
                              with_depth: bool = True,
                              with_done: bool = True) -> dict:
    """Materialize a fake chunk directory with both keepers and intermediates.

    Returns a dict describing what we wrote, so tests can assert what's
    expected to survive vs. be removed.
    """
    import numpy as np
    chunk_dir = orch.chunk_dir(chunk)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    written = {"keep": [], "remove": []}
    for i in range(n_eps):
        ep = chunk_dir / f"ep_{i:08d}"
        ep.mkdir(exist_ok=True)
        # Keepers: RGB / action / state / timestamps / meta
        for name in ("camera_head_rgb", "action_native", "action_canonical_ee",
                     "state", "timestamps"):
            path = ep / f"{name}.npy"
            np.save(path, np.zeros((4, 4), dtype=np.float32))
            written["keep"].append(path)
        meta = ep / "meta.json"
        meta.write_text(json.dumps({"episode_index": i, "instructions": {"level_1": ["task"]}}))
        written["keep"].append(meta)
        # Removable: depth npy + sidecar
        if with_depth:
            depth_npy = ep / "depth_head_rgb.npy"
            np.save(depth_npy, np.zeros((4, 4), dtype=np.uint16))
            written["remove"].append(depth_npy)
            depth_json = ep / "depth_head_rgb.json"
            depth_json.write_text(json.dumps({"low_quality": True, "source": "placeholder"}))
            written["remove"].append(depth_json)
    if with_done:
        done_dir = chunk_dir / "done"
        done_dir.mkdir(exist_ok=True)
        for i in range(n_eps):
            ok = done_dir / f"hash{i:02d}.ok"
            ok.write_text("{}")
            written["remove"].append(ok)
    return written


def test_cleanup_deletes_depth_npy_and_json(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    written = _populate_chunk_artifacts(orch, chunk=0)
    info = orch._cleanup_chunk(0)
    assert info["files_removed"] == len(written["remove"])
    assert info["bytes_freed"] > 0
    for path in written["keep"]:
        assert path.exists(), f"keeper deleted: {path}"
    for path in written["remove"]:
        assert not path.exists(), f"intermediate survived: {path}"


def test_cleanup_removes_done_dir(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    _populate_chunk_artifacts(orch, chunk=3, with_depth=False)
    done = orch.chunk_dir(3) / "done"
    assert done.exists()
    orch._cleanup_chunk(3)
    assert not done.exists()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    """Second cleanup call is a no-op (no errors on already-clean tree)."""
    orch = _make_orch(tmp_path)
    _populate_chunk_artifacts(orch, chunk=0)
    info1 = orch._cleanup_chunk(0)
    info2 = orch._cleanup_chunk(0)
    assert info1["files_removed"] > 0
    assert info2["files_removed"] == 0
    assert info2["bytes_freed"] == 0


def test_cleanup_tolerates_missing_chunk(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    info = orch._cleanup_chunk(999)  # never created
    assert info["files_removed"] == 0
    assert info["bytes_freed"] == 0


def test_no_cleanup_preserves_intermediates(tmp_path: Path, monkeypatch) -> None:
    """``cleanup=False`` keeps depth_*.npy on disk."""
    orch = _make_orch(tmp_path)
    orch.cleanup = False
    calls = _patch_stages(monkeypatch, orch, s2a_returns=[(2, 0)])
    # Pre-seed the chunk dir with what stage_2c WOULD have written, since
    # _run_stage_2c is monkey-patched away.
    written = _populate_chunk_artifacts(orch, chunk=0)
    rc = orch.run()
    assert rc == 0
    assert orch.marker_path(0).exists()
    for path in written["remove"]:
        assert path.exists(), f"intermediate missing with --no-cleanup: {path}"


def test_cleanup_runs_in_full_pipeline(tmp_path: Path, monkeypatch) -> None:
    """``cleanup=True`` (default) deletes intermediates and records bytes freed."""
    orch = _make_orch(tmp_path)
    _patch_stages(monkeypatch, orch, s2a_returns=[(2, 0)])
    written = _populate_chunk_artifacts(orch, chunk=0)
    rc = orch.run()
    assert rc == 0
    marker = json.loads(orch.marker_path(0).read_text())
    assert marker["cleanup_files_removed"] == len(written["remove"])
    assert marker["cleanup_bytes_freed"] > 0
    for path in written["remove"]:
        assert not path.exists(), f"intermediate survived cleanup: {path}"
    for path in written["keep"]:
        assert path.exists(), f"keeper deleted: {path}"
