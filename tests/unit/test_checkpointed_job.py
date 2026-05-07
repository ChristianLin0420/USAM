# SPDX-License-Identifier: MIT
"""Unit tests for ``prep._base.CheckpointedJob``.

Covers:
* Per-episode idempotency: re-running the job over a chunk that's already
  done produces the same shard hash and does not call ``convert_episode``
  a second time.
* SIGUSR1 handling: a signal delivered mid-run causes the in-flight buffer
  to flush and the process to exit with code 124.
"""
from __future__ import annotations

import hashlib
import os
import signal
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from prep._base import CheckpointedJob, ConversionResult, EpisodeRef, PREEMPT_EXIT_CODE


class _MockJob(CheckpointedJob):
    """5-episode in-memory mock used by the tests below.

    Each ``write_shard`` call writes a deterministic line-per-episode file so
    we can hash it and assert idempotency. ``convert_episode`` records the
    set of episode ids it was called for, so we can verify the second run
    skips already-done work.
    """

    def __init__(self, output_root: Path, n_episodes: int = 5, slow_episode: int | None = None) -> None:
        super().__init__(
            source="mock",
            stage="unit_test",
            chunk=0,
            output_root=output_root,
            scratch_root=output_root / "scratch",
            shard_size=10,  # large enough that all 5 episodes fit in one shard
        )
        self.n_episodes = n_episodes
        self.calls: list[str] = []
        self._slow_episode = slow_episode
        self._signal_event: threading.Event | None = None

    def list_episodes(self) -> Iterable[EpisodeRef]:
        for i in range(self.n_episodes):
            yield EpisodeRef(episode_id=f"ep-{i:03d}", source="mock", raw_path=f"/dev/null/{i}")

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        self.calls.append(ref.episode_id)
        # If the test asked for a "slow" episode, deliver the signal here and
        # block briefly so the handler runs before we return.
        if self._slow_episode is not None and ref.episode_id == f"ep-{self._slow_episode:03d}":
            if self._signal_event is not None:
                self._signal_event.set()
            time.sleep(0.05)
        return ConversionResult(
            episode_index=int(ref.episode_id.split("-")[-1]),
            episode_id=ref.episode_id,
            payload={"value": ref.episode_id},
        )

    def write_shard(self, results: list[ConversionResult]) -> Path:
        # Filename embeds the canonical shard hash so two runs over the same
        # set of episodes produce the same file path.
        h = self.shard_hash(results)
        path = self.output_dir / f"file-{h}.txt"
        # Sort by episode_id so the on-disk content is also deterministic.
        lines = sorted(r.episode_id + "\n" for r in results)
        path.write_text("".join(lines))
        return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_idempotency_same_shard_hash(tmp_path: Path) -> None:
    """Running the job twice produces the same shard file (byte-identical)."""
    out = tmp_path / "out"

    job1 = _MockJob(out, n_episodes=5)
    rc1 = job1.run()
    assert rc1 == 0
    assert sorted(job1.calls) == [f"ep-{i:03d}" for i in range(5)]

    shards1 = sorted(job1.output_dir.glob("file-*.txt"))
    assert len(shards1) == 1
    digest1 = _file_sha256(shards1[0])

    # Second run: every episode is already in the done/ marker dir, so
    # ``convert_episode`` should not be called at all.
    job2 = _MockJob(out, n_episodes=5)
    rc2 = job2.run()
    assert rc2 == 0
    assert job2.calls == [], f"second run reprocessed: {job2.calls}"

    shards2 = sorted(job2.output_dir.glob("file-*.txt"))
    assert len(shards2) == 1, "second run must not produce a duplicate shard"
    digest2 = _file_sha256(shards2[0])

    assert digest1 == digest2, "shard digest changed between runs; idempotency broken"
    assert shards1[0].name == shards2[0].name, "shard filename must be identical between runs"


def test_episode_hash_is_deterministic_and_12_chars() -> None:
    """The exact hashing rule from docs/IMPLEMENTATION_PLAN.md §11.13."""
    h = CheckpointedJob.episode_hash("ep-000")
    expected = hashlib.sha256(b"ep-000").hexdigest()[:12]
    assert h == expected
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_sigusr1_causes_exit_124(tmp_path: Path) -> None:
    """Mid-run SIGUSR1 → flush + SystemExit(124)."""
    out = tmp_path / "out"

    job = _MockJob(out, n_episodes=20, slow_episode=2)
    job._signal_event = threading.Event()

    def deliver_signal() -> None:
        # Wait until the slow episode actually starts running, then signal.
        if job._signal_event is not None and job._signal_event.wait(timeout=5.0):
            os.kill(os.getpid(), signal.SIGUSR1)

    t = threading.Thread(target=deliver_signal, daemon=True)
    t.start()

    with pytest.raises(SystemExit) as exc_info:
        job.run()

    t.join(timeout=2.0)

    assert exc_info.value.code == PREEMPT_EXIT_CODE == 124

    # We should have processed at least the slow episode and at most all 20.
    assert 1 <= len(job.calls) <= 20
    # A partial shard must have been flushed for whatever was completed.
    shards = list(job.output_dir.glob("file-*.txt"))
    if job.calls:
        assert len(shards) == 1, f"expected one partial shard, got {shards}"
        # Every episode in the buffer should be marked done so a resume skips it.
        for ep_id in job.calls:
            marker = job.output_dir / "done" / f"{CheckpointedJob.episode_hash(ep_id)}.ok"
            assert marker.exists(), f"episode {ep_id} not marked done after partial flush"


def test_resume_after_preemption(tmp_path: Path) -> None:
    """After a preempt-and-resume cycle, total episodes processed == n_episodes."""
    out = tmp_path / "out"

    # First run: stop after the third episode.
    job1 = _MockJob(out, n_episodes=5, slow_episode=2)
    job1._signal_event = threading.Event()

    def deliver_signal() -> None:
        if job1._signal_event is not None and job1._signal_event.wait(timeout=5.0):
            os.kill(os.getpid(), signal.SIGUSR1)

    t = threading.Thread(target=deliver_signal, daemon=True)
    t.start()
    with pytest.raises(SystemExit) as exc_info:
        job1.run()
    t.join(timeout=2.0)
    assert exc_info.value.code == 124
    first_pass_calls = list(job1.calls)
    assert first_pass_calls, "first pass should have processed at least one episode"

    # Second run: should pick up where we left off.
    job2 = _MockJob(out, n_episodes=5)
    rc = job2.run()
    assert rc == 0
    # Every episode processed in either run must appear exactly once across
    # the union of calls — no duplicates, no losses.
    union = set(first_pass_calls) | set(job2.calls)
    assert union == {f"ep-{i:03d}" for i in range(5)}
    assert set(first_pass_calls).isdisjoint(set(job2.calls)), (
        f"episode reprocessed across resume: first={first_pass_calls} second={job2.calls}"
    )
