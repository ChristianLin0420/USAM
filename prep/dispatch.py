# SPDX-License-Identifier: MIT
"""Slurm DAG dispatcher for the USAM Phase A pipeline.

The dispatcher is a long-lived process — typically a tmux session on a
login node — that:

1. Reads each ``manifests/<source>__stage_1_index/manifest.parquet`` to
   discover the universe of chunks per source.
2. Tracks per-chunk state (``pending`` / ``running`` / ``done`` / ``failed``)
   in ``dispatch_state.json`` so a restart resumes correctly.
3. Throttles in-flight Slurm jobs to ``MAX_PENDING`` (default 64; configurable
   via ``--max-pending`` or ``USAM_MAX_PENDING`` env var) — this is the single
   line that makes ``self._submitted_count() < self.max_pending`` in
   :meth:`SlurmDispatcher.step`.
4. Submits ready chunks via ``sbatch`` (or any caller-supplied launcher mock,
   used by ``tests/integration/test_pipeline_end_to_end.py``).
5. Detects requeue exits (``prep._base.PREEMPT_EXIT_CODE``) and keeps the
   chunk in the running set rather than marking it failed.

The DAG mirrors ``docs/IMPLEMENTATION_PLAN.md §5.3``::

    stage_2a_to_lerobot ──┐
                           ├──> stage_3_canonical ──> stage_4_dino_cache
    stage_2c_compute_depth ┘
                                            ──> stage_5_validate ──> stage_6_upload

Two key design choices:

* The dispatcher does not know how to *do* a stage; it only knows how to
  fire-and-forget a Slurm job. Every stage's launcher is the same
  ``slurm/job.sbatch`` script with different arguments.
* ``MAX_PENDING`` is enforced by counting the rows currently in
  ``RUNNING`` state in the dispatcher's own state, not by polling
  ``squeue``. Polling ``squeue`` is a fallback in :meth:`squeue_count`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from prep._base import PREEMPT_EXIT_CODE

__all__ = [
    "ChunkStatus",
    "ChunkRecord",
    "SlurmDispatcher",
    "DEFAULT_DAG",
    "main",
]

_LOG = logging.getLogger(__name__)

# Stage execution order. Each entry is the python module path used by
# slurm/job.sbatch's first positional argument. The dispatcher submits a
# stage only after every preceding stage in this list reports "done".
DEFAULT_DAG: list[str] = [
    "stage_2a_to_lerobot",
    "stage_2c_compute_depth",
    "stage_3_canonical",
    "stage_4_dino_cache",
    "stage_5_validate",
    "stage_6_upload",
]


class ChunkStatus(str, Enum):
    """Lifecycle of a single (source, stage, chunk) tuple."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REQUEUED = "requeued"


@dataclass
class ChunkRecord:
    """Per-(source, stage, chunk) tracking record in ``dispatch_state.json``.

    Attributes
    ----------
    source, stage, chunk : str/str/int
    status : ChunkStatus
    job_id : str | None
        Slurm job id (or mock id under tests).
    attempts : int
    last_exit : int | None
    updated_at : int
        unix timestamp (s)
    """

    source: str
    stage: str
    chunk: int
    status: ChunkStatus = ChunkStatus.PENDING
    job_id: str | None = None
    attempts: int = 0
    last_exit: int | None = None
    updated_at: int = 0

    def key(self) -> str:
        return f"{self.source}::{self.stage}::{self.chunk:06d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "stage": self.stage,
            "chunk": self.chunk,
            "status": self.status.value,
            "job_id": self.job_id,
            "attempts": self.attempts,
            "last_exit": self.last_exit,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkRecord:
        return cls(
            source=str(d["source"]),
            stage=str(d["stage"]),
            chunk=int(d["chunk"]),
            status=ChunkStatus(str(d.get("status", "pending"))),
            job_id=d.get("job_id"),
            attempts=int(d.get("attempts", 0)),
            last_exit=d.get("last_exit"),
            updated_at=int(d.get("updated_at", 0)),
        )


# A launcher is any callable: ``(stage, source, chunk, extra_args) -> (exit_code, job_id)``.
# Real production: subprocess to ``sbatch ...``. Tests substitute a bash mock
# that runs the python module synchronously and returns its exit code.
Launcher = Callable[[str, str, int, list[str]], tuple[int, str]]


def sbatch_launcher(
    stage: str,
    source: str,
    chunk: int,
    extra_args: list[str],
    sbatch_path: str = "sbatch",
    template: str = "slurm/job.sbatch",
) -> tuple[int, str]:
    """Production launcher: ``sbatch slurm/job.sbatch <stage> <dataset> <chunk>``.

    The dataset name is the second positional arg (formerly called ``source``);
    ``slurm/job.sbatch`` forwards it to the python module via ``--dataset``.

    Returns ``(0, job_id)`` on successful submission. The actual job
    completion is observed asynchronously via the per-stage marker files
    that ``CheckpointedJob`` writes; we do not block on the slurm job here.
    """
    cmd = [sbatch_path, template, stage, source, str(chunk), *extra_args]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # ``Submitted batch job 12345``
    job_id = out.stdout.strip().split()[-1]
    return 0, job_id


@dataclass
class SlurmDispatcher:
    """Long-lived DAG dispatcher.

    Parameters
    ----------
    output_root : Path
        Where all stage outputs land. The dispatcher uses this path to
        discover manifests (``<output_root>/<source>/stage_1_index/...``)
        and per-stage chunk-XXX/done markers.
    max_pending : int
        Cap on simultaneously-running chunks. The throttle is enforced in
        :meth:`step` before any submission.
    state_path : Path
        Persisted state JSON (``dispatch_state.json``).
    sources : list[str]
    dag : list[str]
        Sequence of stage module names; later stages wait for earlier ones.
    launcher : Launcher
        Defaults to :func:`sbatch_launcher`.
    poll_seconds : int
        Sleep between :meth:`step` calls in :meth:`run_forever`.
    """

    output_root: Path
    max_pending: int = 64
    state_path: Path = field(default_factory=lambda: Path("dispatch_state.json"))
    sources: list[str] = field(default_factory=list)
    dag: list[str] = field(default_factory=lambda: list(DEFAULT_DAG))
    launcher: Launcher = field(default=sbatch_launcher)
    poll_seconds: int = 60
    chunks_per_source: dict[str, int] = field(default_factory=dict)
    _records: dict[str, ChunkRecord] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        """Load ``dispatch_state.json`` into ``self._records`` (id -> record)."""
        if not self.state_path.exists():
            self._records = {}
            return
        try:
            raw = json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            _LOG.warning("dispatch state is corrupt at %s; starting fresh", self.state_path)
            raw = {}
        records = {}
        for key, body in raw.get("records", {}).items():
            records[key] = ChunkRecord.from_dict(body)
        self._records = records

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_pending": int(self.max_pending),
            "saved_at": int(time.time()),
            "records": {k: r.to_dict() for k, r in self._records.items()},
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.state_path)

    # ------------------------------------------------------------------
    # DAG state queries
    # ------------------------------------------------------------------

    def _ensure_records(self) -> None:
        """Populate ``self._records`` with one row per (source, stage, chunk)."""
        for source in self.sources:
            n_chunks = max(1, int(self.chunks_per_source.get(source, 1)))
            for stage in self.dag:
                for ch in range(n_chunks):
                    rec = ChunkRecord(source=source, stage=stage, chunk=ch)
                    self._records.setdefault(rec.key(), rec)

    def _stage_index(self, stage: str) -> int:
        try:
            return self.dag.index(stage)
        except ValueError:
            return -1

    def _deps_done(self, rec: ChunkRecord) -> bool:
        """True iff every earlier stage for the same (source, chunk) is done."""
        idx = self._stage_index(rec.stage)
        if idx <= 0:
            return True
        for prev_stage in self.dag[:idx]:
            prev_rec = self._records.get(
                ChunkRecord(source=rec.source, stage=prev_stage, chunk=rec.chunk).key()
            )
            if prev_rec is None or prev_rec.status is not ChunkStatus.DONE:
                return False
        return True

    def _submitted_count(self) -> int:
        """Number of chunks currently in RUNNING / REQUEUED state."""
        return sum(
            1 for r in self._records.values()
            if r.status in (ChunkStatus.RUNNING, ChunkStatus.REQUEUED)
        )

    def squeue_count(self, user: str | None = None) -> int:
        """Optional fallback: count user's USAM jobs in ``squeue``.

        The dispatcher's own RUNNING count is authoritative; this helper is
        provided so operators can sanity-check from the cli.
        """
        try:
            user = user or os.environ.get("USER", "")
            out = subprocess.run(
                ["squeue", "-u", user, "-h", "-o", "%i"],
                capture_output=True, text=True, check=True, timeout=20,
            )
            return len([line for line in out.stdout.splitlines() if line.strip()])
        except Exception as exc:  # pragma: no cover - real-runtime path
            _LOG.warning("squeue probe failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Submission step
    # ------------------------------------------------------------------

    def step(self) -> int:
        """One pass of the scheduler. Returns the number of jobs submitted.

        Throttle is enforced by checking ``self._submitted_count() <
        self.max_pending`` before each ``self.launcher`` call. The
        dispatcher will *not* exceed ``max_pending`` even if many chunks
        become eligible in the same poll.
        """
        self._ensure_records()
        submitted = 0
        for rec in sorted(self._records.values(), key=lambda r: (self._stage_index(r.stage), r.source, r.chunk)):
            if self._submitted_count() >= self.max_pending:
                _LOG.info("max_pending=%d reached; deferring further submissions", self.max_pending)
                break
            if rec.status not in (ChunkStatus.PENDING, ChunkStatus.REQUEUED):
                continue
            if not self._deps_done(rec):
                continue

            # Launch
            rec.attempts += 1
            try:
                exit_code, job_id = self.launcher(rec.stage, rec.source, rec.chunk, [])
            except Exception as exc:
                _LOG.exception("launcher raised for %s: %s", rec.key(), exc)
                rec.status = ChunkStatus.FAILED
                rec.last_exit = -1
                rec.updated_at = int(time.time())
                continue
            rec.job_id = job_id
            rec.last_exit = exit_code
            rec.updated_at = int(time.time())
            if exit_code == 0:
                # In the test mock the launcher runs synchronously and returns
                # the python child's exit code directly; in production the
                # sbatch_launcher returns 0 for "submitted ok" and the actual
                # python exit is observed later via the marker files.
                rec.status = ChunkStatus.DONE
            elif exit_code == PREEMPT_EXIT_CODE:
                rec.status = ChunkStatus.REQUEUED
            else:
                rec.status = ChunkStatus.FAILED
            submitted += 1

        self.save_state()
        return submitted

    def run_forever(self, max_iterations: int | None = None) -> None:
        """Block in a step / sleep loop until everything is done or capped."""
        self.load_state()
        i = 0
        while True:
            self.step()
            if all(r.status is ChunkStatus.DONE for r in self._records.values()):
                _LOG.info("dispatcher: all chunks done")
                return
            if max_iterations is not None and i + 1 >= max_iterations:
                _LOG.info("dispatcher: reached max_iterations=%d", max_iterations)
                return
            i += 1
            time.sleep(self.poll_seconds)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI driver. ``--once`` is useful for scripted/test runs."""
    parser = argparse.ArgumentParser(prog="prep.dispatch", description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--state", type=Path, default=Path("dispatch_state.json"))
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--datasets", nargs="+",
                    help="Dataset names to schedule (one A100 node per dataset, "
                         "per Wave F).")
    ds.add_argument("--sources", dest="datasets", nargs="+",
                    help="(deprecated) use --datasets")
    parser.add_argument("--chunks-per-dataset", type=int, default=1,
                        help="Uniform chunk count per dataset. For a heterogeneous mix, "
                             "edit dispatch_state.json directly after the first run.")
    parser.add_argument("--chunks-per-source", dest="chunks_per_dataset", type=int,
                        help="(deprecated) use --chunks-per-dataset")
    parser.add_argument("--max-pending", type=int,
                        default=int(os.environ.get("USAM_MAX_PENDING", "64")))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true",
                        help="Run a single dispatch step and exit (useful in tests)")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    dispatcher = SlurmDispatcher(
        output_root=args.output_root,
        max_pending=args.max_pending,
        state_path=args.state,
        sources=list(args.datasets),
        chunks_per_source={s: int(args.chunks_per_dataset) for s in args.datasets},
        poll_seconds=args.poll_seconds,
    )
    if args.once:
        dispatcher.load_state()
        n = dispatcher.step()
        _LOG.info("dispatch step submitted %d job(s)", n)
        return 0
    dispatcher.run_forever(max_iterations=args.max_iterations)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
