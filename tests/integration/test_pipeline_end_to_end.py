# SPDX-License-Identifier: MIT
"""End-to-end pipeline test for USAM Phase A.

This test exercises the *control flow* of the prep pipeline — preemption,
graceful flush, requeue, and idempotent resume — without actually shelling
out to ``sbatch`` or talking to HF Hub. It is the regression gate for the
contract that ``CheckpointedJob`` ships in ``prep/_base.py`` and that
``prep/dispatch.py`` relies on.

What we cover
-------------

1. Generate ``tests/golden_data/tiny_droid`` via the seeded synthesizer (the
   conftest fixture already does this for unit tests).
2. Simulate a 5-episode chunk being processed by a ``CheckpointedJob`` —
   with a deliberately slow per-episode hook so we can land a SIGUSR1
   mid-run.
3. Fire ``SIGUSR1`` mid-job in a child process. Assert the child exits with
   ``PREEMPT_EXIT_CODE=124`` and that, before exit, every completed
   episode's marker file is present on disk.
4. Restart the same job (same chunk dir!) and verify it resumes, processes
   only the remaining episodes, and ends with ``set(produced) ==
   set(expected)`` and ``len(produced) == 5`` — no duplicates, no losses.
5. Run :class:`prep.dispatch.SlurmDispatcher` end-to-end with a
   bash-as-launcher mock so the DAG is exercised and the throttle is
   enforced.
6. Run :func:`prep.stage_5_validate.validate_source_outputs` over the
   resulting tree and assert the per-shard report passes the parquet
   columns gate.
7. Run :func:`prep.stage_6_upload.upload_source` with ``dry_run=True`` and
   a stub upload function (no HF API contact); assert the
   ``.upload_state.json`` file is materialized.

Marked ``@pytest.mark.slow`` so CI skips it by default. The team lead runs::

    /localhome/local-chrislin/miniconda3/envs/qwen3vl/bin/python -m pytest \
        tests/integration/test_pipeline_end_to_end.py -m slow -x -v
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import signal
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from prep._base import (
    PREEMPT_EXIT_CODE,
    CheckpointedJob,
    ConversionResult,
    EpisodeRef,
)
from prep.dispatch import ChunkStatus, DEFAULT_DAG, SlurmDispatcher
from prep.stage_5_validate import validate_source_outputs
from prep.stage_6_upload import compute_chunk_hash, plan_chunks, upload_source

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- #
# A synthetic CheckpointedJob that lets us steer the SIGUSR1 timing           #
# --------------------------------------------------------------------------- #


class _SlowSyntheticJob(CheckpointedJob):
    """5-episode synthetic job. Each ``convert_episode`` sleeps briefly
    so a parent process has a window to deliver SIGUSR1.

    The "shard" we write is a JSON file listing the episodes; that's
    plenty to enforce the per-episode-marker idempotency contract that
    matters for the preemption test.
    """

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        scratch_root: Path | None = None,
        n_episodes: int = 5,
        per_episode_seconds: float = 0.3,
        shard_size: int = 3,
    ) -> None:
        super().__init__(
            source="synth",
            stage="2a",
            chunk=chunk,
            output_root=output_root,
            scratch_root=scratch_root,
            shard_size=shard_size,
        )
        self.n_episodes = int(n_episodes)
        self.per_episode_seconds = float(per_episode_seconds)

    def list_episodes(self) -> Iterable[EpisodeRef]:
        for i in range(self.n_episodes):
            yield EpisodeRef(
                episode_id=f"synth::ep_{i:03d}",
                source="synth",
                raw_path=f"/dev/null/ep_{i}",
                extra={"i": i},
            )

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        # The sleep is what gives the parent process a chance to land
        # SIGUSR1 mid-loop. CheckpointedJob.run() polls _stop_requested
        # *between* episodes, not inside this call, so the in-flight
        # episode finishes cleanly even on signal.
        time.sleep(self.per_episode_seconds)
        return ConversionResult(
            episode_index=int(ref.extra["i"]),
            episode_id=ref.episode_id,
            payload={"i": int(ref.extra["i"]), "raw_path": ref.raw_path},
        )

    def write_shard(self, results: list[ConversionResult]) -> Path:
        shard_h = self.shard_hash(results)
        out = self.output_dir / f"shard-{shard_h}.json"
        out.write_text(json.dumps([r.payload for r in results], sort_keys=True))
        return out


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _run_synthetic_in_child(
    output_root: str,
    scratch_root: str,
    n_episodes: int,
    per_episode_seconds: float,
    chunk: int,
    return_pipe: Any,
) -> None:
    """Worker entry point for ``multiprocessing.Process`` runs."""
    job = _SlowSyntheticJob(
        chunk=chunk,
        output_root=Path(output_root),
        scratch_root=Path(scratch_root),
        n_episodes=n_episodes,
        per_episode_seconds=per_episode_seconds,
    )
    rc = 0
    try:
        rc = job.run()
    except SystemExit as exc:
        rc = int(exc.code) if exc.code is not None else 0
    return_pipe.send(int(rc))
    return_pipe.close()


def _collect_completed_markers(output_dir: Path) -> set[str]:
    """Return the set of episode_ids whose ``done/<hash>.ok`` exists."""
    done = output_dir / "done"
    if not done.exists():
        return set()
    out: set[str] = set()
    for marker in done.glob("*.ok"):
        try:
            data = json.loads(marker.read_text())
        except json.JSONDecodeError:
            continue
        eid = data.get("episode_id")
        if isinstance(eid, str):
            out.add(eid)
    return out


def _collect_shard_episodes(output_dir: Path) -> list[int]:
    """Return the list of integer ``i`` values present in shard files."""
    out: list[int] = []
    for shard in sorted(output_dir.glob("shard-*.json")):
        out.extend(int(r["i"]) for r in json.loads(shard.read_text()))
    return out


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_tiny_droid_fixture_materializes(tiny_droid_root: Path) -> None:
    """The synthesized fixture must exist before the rest of this file runs."""
    assert tiny_droid_root.exists()
    assert (tiny_droid_root / "meta" / "info.json").exists()
    info = json.loads((tiny_droid_root / "meta" / "info.json").read_text())
    assert info["source"] == "droid"
    assert info["embodiment"] == "droid_franka"


def test_sigusr1_round_trips_with_no_loss_or_dup(tmp_path: Path) -> None:
    """Inject SIGUSR1 mid-job; assert resumed run yields exactly 5 episodes.

    This is the core preemption gate. The first child runs until SIGUSR1
    arrives; the second child resumes from the per-episode markers.
    """
    output_root = tmp_path / "out"
    scratch_root = tmp_path / "scratch"
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    # ---- Child 1: start the job, fire SIGUSR1 mid-run ----
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p1 = mp.Process(
        target=_run_synthetic_in_child,
        args=(str(output_root), str(scratch_root), 5, 0.3, 0, child_conn),
        daemon=False,
    )
    p1.start()
    # Let it process at least one episode (~ 1 * 0.3 s) before we poke it.
    time.sleep(0.6)
    assert p1.is_alive(), "child died before we could signal it"
    os.kill(p1.pid, signal.SIGUSR1)
    p1.join(timeout=20)
    assert not p1.is_alive(), "child did not exit after SIGUSR1"
    assert parent_conn.poll(timeout=5), "child did not report exit code"
    rc1 = parent_conn.recv()
    parent_conn.close()
    assert rc1 == PREEMPT_EXIT_CODE, f"expected {PREEMPT_EXIT_CODE}, got {rc1}"

    # The chunk dir from the first run.
    chunk_dir = output_root / "synth" / "2a" / "chunk-000"
    assert chunk_dir.exists()
    completed_after_preempt = _collect_completed_markers(chunk_dir)
    assert 0 < len(completed_after_preempt) < 5, (
        f"expected partial progress, got {len(completed_after_preempt)} markers"
    )

    # ---- Child 2: resume; let it finish ----
    parent_conn2, child_conn2 = mp.Pipe(duplex=False)
    p2 = mp.Process(
        target=_run_synthetic_in_child,
        args=(str(output_root), str(scratch_root), 5, 0.3, 0, child_conn2),
        daemon=False,
    )
    p2.start()
    p2.join(timeout=30)
    assert not p2.is_alive(), "resumed child failed to terminate"
    assert parent_conn2.poll(timeout=5)
    rc2 = parent_conn2.recv()
    parent_conn2.close()
    assert rc2 == 0, f"resumed child exited {rc2}, expected 0"

    # ---- Assertions: the global hash-set on completed episodes ----
    expected = {f"synth::ep_{i:03d}" for i in range(5)}
    produced = _collect_completed_markers(chunk_dir)
    assert produced == expected, (
        f"missing or extra episode ids; expected={sorted(expected)} produced={sorted(produced)}"
    )
    assert len(produced) == 5

    # The shards' integer i list must contain every i exactly once.
    is_in_shards = _collect_shard_episodes(chunk_dir)
    assert sorted(is_in_shards) == [0, 1, 2, 3, 4]
    assert len(is_in_shards) == 5, "duplicate episode in shard files"


def test_dispatcher_schedules_dag_with_bash_launcher_mock(tmp_path: Path) -> None:
    """``SlurmDispatcher`` must respect DAG order and MAX_PENDING."""
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)

    # Each (stage, source, chunk) the launcher gets called for is recorded.
    # The mock returns 0 to signal "this stage's chunk is done".
    submission_log: list[tuple[str, str, int]] = []
    in_flight: list[tuple[str, str, int]] = []
    max_observed_in_flight = {"v": 0}

    def bash_launcher(stage: str, source: str, chunk: int, extra: list[str]) -> tuple[int, str]:
        # Record concurrency at submission time. The dispatcher's
        # _submitted_count() is incremented inside step() as records flip
        # into RUNNING; we observe it here by checking against in_flight.
        in_flight.append((stage, source, chunk))
        max_observed_in_flight["v"] = max(max_observed_in_flight["v"], len(in_flight))
        submission_log.append((stage, source, chunk))
        # Synchronous "job": pretend we ran instantly and returned 0.
        in_flight.pop()
        return 0, f"mock-{stage}-{source}-{chunk}"

    sources = ["droid", "agibot2026"]
    dispatcher = SlurmDispatcher(
        output_root=output_root,
        max_pending=2,  # tight cap so we can verify enforcement
        state_path=tmp_path / "dispatch_state.json",
        sources=sources,
        chunks_per_source={s: 2 for s in sources},
        launcher=bash_launcher,
        poll_seconds=0,
    )
    dispatcher.load_state()

    # Step until everything is done (or we run out of patience).
    for _ in range(200):
        n = dispatcher.step()
        if all(r.status is ChunkStatus.DONE for r in dispatcher._records.values()):
            break
        if n == 0:
            break

    # Every (source, stage, chunk) in DEFAULT_DAG must have been submitted exactly once.
    expected_keys = {
        (stage, source, chunk)
        for source in sources
        for chunk in range(2)
        for stage in DEFAULT_DAG
    }
    assert set(submission_log) == expected_keys, (
        f"missing or extra submissions; want={sorted(expected_keys)} got={sorted(submission_log)}"
    )
    # Each submission appears once.
    assert len(submission_log) == len(expected_keys), "duplicate submissions"

    # DAG order: for each (source, chunk), the order of stages submitted must
    # match DEFAULT_DAG.
    for source in sources:
        for chunk in range(2):
            seen = [s for (s, src, ch) in submission_log if src == source and ch == chunk]
            assert seen == DEFAULT_DAG, (
                f"DAG order violated for {source}/chunk {chunk}: {seen}"
            )

    # All records must be DONE.
    assert all(r.status is ChunkStatus.DONE for r in dispatcher._records.values())

    # State file must exist and round-trip.
    state_payload = json.loads((tmp_path / "dispatch_state.json").read_text())
    assert state_payload["max_pending"] == 2
    assert "records" in state_payload


def test_dispatcher_throttle_caps_in_flight(tmp_path: Path) -> None:
    """``MAX_PENDING`` is enforced; submission stops at the cap.

    The mock launcher returns ``PREEMPT_EXIT_CODE`` so the dispatcher
    flips records to REQUEUED (which counts toward ``_submitted_count``)
    instead of DONE, simulating jobs that stay in-flight indefinitely.
    """
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)

    n_calls = {"v": 0}

    def stuck_launcher(stage: str, source: str, chunk: int, extra: list[str]) -> tuple[int, str]:
        n_calls["v"] += 1
        return PREEMPT_EXIT_CODE, f"stuck-{n_calls['v']}"

    sources = ["droid", "agibot2026", "robomind"]
    dispatcher = SlurmDispatcher(
        output_root=output_root,
        max_pending=3,
        state_path=tmp_path / "dispatch_state.json",
        sources=sources,
        chunks_per_source={s: 4 for s in sources},  # 12 chunks per stage
        launcher=stuck_launcher,
        poll_seconds=0,
    )
    dispatcher.load_state()
    n = dispatcher.step()
    # Only 3 records should be in-flight after one step thanks to the cap.
    assert n == 3, f"throttle violated: submitted {n} jobs, expected ≤ 3"
    in_flight = sum(
        1 for r in dispatcher._records.values()
        if r.status is ChunkStatus.REQUEUED
    )
    assert in_flight == 3, f"expected 3 REQUEUED records, got {in_flight}"


def test_validation_runs_on_synthesized_tree(
    tmp_path: Path, tiny_droid_root: Path
) -> None:
    """Stage-5 validates the synthesized droid fixture's parquet shards."""
    # Mirror the fixture into an output_root layout: <out>/<source>/data/...
    out = tmp_path / "out"
    src_dir = out / "droid"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_droid_root / "data", src_dir / "data")

    report = validate_source_outputs(out, source="droid")
    assert report.summary.get("parquet_ok", 0) >= 1, report.summary
    # No mp4 / safetensors in this minimal tree, so those keys should be 0.
    assert report.summary.get("mp4_fail", 0) == 0
    assert report.summary.get("safetensors_fail", 0) == 0


def test_upload_dry_run_and_state(tmp_path: Path, tiny_droid_root: Path) -> None:
    """Stage-6 must plan chunks, hash them, and persist .upload_state.json."""
    out = tmp_path / "out"
    src_dir = out / "droid"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_droid_root / "data", src_dir / "data")

    plans = plan_chunks(src_dir)
    assert len(plans) >= 1, "no chunks planned for non-empty source"
    h = compute_chunk_hash(plans[0])
    assert isinstance(h, str) and len(h) >= 8

    # Stub upload function: must NOT be called when dry_run=True.
    n_calls = {"v": 0}

    def stub_upload(**kwargs: Any) -> None:
        n_calls["v"] += 1

    state = upload_source(
        source="droid",
        output_root=out,
        repo_prefix="test/usam-",
        dry_run=True,
        upload_fn=stub_upload,
    )
    assert n_calls["v"] == 0, "stub_upload was called despite dry_run=True"
    assert any(k.startswith("chunk-") for k in state)
    assert (src_dir / ".upload_state.json").exists()


def test_stage_0_download_droid_dry_run(tmp_path: Path) -> None:
    """The DROID downloader's ``--dry-run`` path must touch no network."""
    cfg_path = tmp_path / "droid.yaml"
    cfg_path.write_text(
        "source: droid\n"
        "download:\n"
        "  rlds_data_dir: gs://gresearch/robotics\n"
        "  karlp_droid_repo: KarlP/droid\n"
    )
    cache = tmp_path / "raw"

    from prep.stage_0_download import droid as droid_dl

    result = droid_dl.download(cfg_path, cache, dry_run=True)
    assert result.backend == "noop"
    assert (cache / "download_manifest.json").exists()
    manifest = json.loads((cache / "download_manifest.json").read_text())
    assert manifest["source"] == "droid"
    assert manifest["extra"]["dry_run"] is True


def test_stage_1_index_builds_manifest(tmp_path: Path) -> None:
    """The IndexJob must produce a manifest.parquet (or .jsonl fallback)."""
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    # Synthetic raw layout: 3 fake .h5 files for robomind.
    raw_root.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (raw_root / f"trajectory_{i:02d}.h5").write_bytes(b"\x00" * 16)

    from prep.stage_1_index import build_index_for_source

    manifest = build_index_for_source(
        source="robomind",
        raw_root=raw_root,
        output_root=out_root,
    )
    assert manifest.exists()
    # Either parquet (preferred) or jsonl (fallback). Both must list 3 episodes.
    if manifest.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            pytest.skip("pyarrow unavailable; cannot inspect parquet")
        rows = pq.read_table(str(manifest)).to_pylist()
    else:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    embodiments = {r["embodiment"] for r in rows}
    assert embodiments == {"robomind_tien_kung"}


def test_full_local_dag_smoke_no_network(tmp_path: Path) -> None:
    """Run the synthetic CheckpointedJob, then validate, then dry-run upload.

    This is the end-to-end smoke that proves the CheckpointedJob output
    layout, validation, and uploader all agree on shapes — without any
    Slurm or Hub IO.
    """
    output_root = tmp_path / "out"
    job = _SlowSyntheticJob(
        chunk=0,
        output_root=output_root,
        scratch_root=tmp_path / "scratch",
        n_episodes=5,
        per_episode_seconds=0.0,
    )
    rc = job.run()
    assert rc == 0

    # 5 markers, no duplicates.
    chunk_dir = output_root / "synth" / "2a" / "chunk-000"
    markers = _collect_completed_markers(chunk_dir)
    assert len(markers) == 5
    assert markers == {f"synth::ep_{i:03d}" for i in range(5)}

    # The synthetic job writes JSON shards (not parquet/mp4/safetensors), so
    # ``validate_source_outputs`` returns an "all-ok" empty summary — proves
    # validation walks the tree without exploding.
    report = validate_source_outputs(output_root, source="synth")
    assert report.summary == {
        "parquet_ok": 0, "parquet_fail": 0,
        "mp4_ok": 0, "mp4_fail": 0,
        "safetensors_ok": 0, "safetensors_fail": 0,
    }, report.summary

    # Plan a dry-run upload over the synthetic shards.
    n_calls = {"v": 0}

    def stub_upload(**kwargs: Any) -> None:
        n_calls["v"] += 1

    state = upload_source(
        source="synth",
        output_root=output_root,
        repo_prefix="test/usam-",
        dry_run=True,
        upload_fn=stub_upload,
    )
    assert n_calls["v"] == 0
    assert state, "state should be non-empty after planning"
