# SPDX-License-Identifier: MIT
"""Stage 6: idempotent uploader.

Walks a source's output directory, splits its contents into chunk-shaped
sub-folders that satisfy the Hub's ``MAX_FILES_PER_CHUNK=1000`` and
``MAX_BYTES_PER_FILE=5 GiB`` limits, then calls
:func:`prep._hub.upload_chunk_final` on each one.

Idempotency
-----------

Each chunk gets a content hash recorded in
``<output_root>/<source>/.upload_state.json``. A subsequent run skips any
chunk whose hash has not changed since the last successful upload.

This module is also the entry point for the long-lived
``CommitScheduler`` daemon — see ``--watch`` below.

CRITICAL DEPLOYMENT NOTE
-------------------------

The ``CommitScheduler`` (``--watch``) MUST run on a **login node**, not
inside a Slurm batch job. Slurm jobs:

1. Are short-lived; the scheduler's "last commit" state would be lost
   on every requeue.
2. Should not waste their walltime on IO-bound uploads.
3. Mount ephemeral scratch which the scheduler would re-diff on every
   restart, forcing redundant network IO.

Run it under ``tmux`` or ``systemd --user`` on the login node instead.
``slurm/README.md`` documents the full operational pattern.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from prep._hub import (
    MAX_BYTES_PER_FILE,
    MAX_FILES_PER_CHUNK,
    make_commit_scheduler,
    upload_chunk_final,
    validate_chunk,
)

__all__ = [
    "ChunkPlan",
    "plan_chunks",
    "compute_chunk_hash",
    "upload_source",
    "watch_source",
    "main",
]

_LOG = logging.getLogger(__name__)

_STATE_FILENAME = ".upload_state.json"


@dataclass
class ChunkPlan:
    """One uploadable unit.

    Parameters
    ----------
    chunk_id : int
    folder : Path
        Local directory containing exactly the files for this chunk.
    files : list[Path]
        Concrete list of files (sorted) that belong to the chunk.
    total_bytes : int
    """

    chunk_id: int
    folder: Path
    files: list[Path] = field(default_factory=list)
    total_bytes: int = 0


def _is_payload_file(path: Path, root: Path) -> bool:
    """Skip bookkeeping files (matches ``prep._hub.validate_chunk``)."""
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    rel = path.relative_to(root).parts
    if "done" in rel:
        return False
    return True


def plan_chunks(
    source_root: Path,
    max_files: int = MAX_FILES_PER_CHUNK,
    max_bytes_per_file: int = MAX_BYTES_PER_FILE,
) -> list[ChunkPlan]:
    """Group a source's output files into chunks that satisfy Hub limits.

    The grouping walks ``source_root`` deterministically (sorted) so reruns
    produce identical plans. Files whose individual size already exceeds
    ``max_bytes_per_file`` are *not* split here — the validator
    (``prep._hub.validate_chunk``) refuses them at upload time and the
    operator must re-shard upstream. Stage_5 should have caught it already.
    """
    assert isinstance(source_root, Path), f"source_root must be a Path, got {type(source_root).__name__}"
    if not source_root.exists():
        return []

    plans: list[ChunkPlan] = []
    cur = ChunkPlan(chunk_id=0, folder=source_root)
    next_id = 1
    for path in sorted(source_root.rglob("*")):
        if not _is_payload_file(path, source_root):
            continue
        size = path.stat().st_size
        # If adding this file would blow the file-count cap, start a new chunk.
        if len(cur.files) >= max_files:
            plans.append(cur)
            cur = ChunkPlan(chunk_id=next_id, folder=source_root)
            next_id += 1
        cur.files.append(path)
        cur.total_bytes += size
    if cur.files:
        plans.append(cur)
    return plans


def compute_chunk_hash(plan: ChunkPlan) -> str:
    """Stable hash of (path, size, mtime) for every file in the chunk.

    Used to short-circuit re-uploads. We deliberately include size and
    mtime so a regenerated shard with identical bytes but a fresh mtime
    counts as "changed" — uploads are cheap relative to the cost of
    silently shipping a stale shard.
    """
    h = hashlib.sha256()
    for f in plan.files:
        st = f.stat()
        h.update(str(f).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(st.st_size).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _load_state(state_path: Path) -> dict[str, str]:
    if not state_path.exists():
        return {}
    try:
        return dict(json.loads(state_path.read_text()))
    except Exception:
        return {}


def _save_state(state_path: Path, state: dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(state_path)


def upload_source(
    source: str,
    output_root: Path,
    repo_prefix: str = "<org>/usam-",
    repo_id: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
    dry_run: bool = False,
    upload_fn=upload_chunk_final,
) -> dict[str, str]:
    """Upload every chunk under ``<output_root>/<source>``.

    Parameters
    ----------
    source : str
    output_root : Path
    repo_prefix : str
        Used iff ``repo_id`` is None: the target repo is
        ``f"{repo_prefix}{source}"``.
    repo_id : str | None
        Explicit repo id override.
    repo_type : str
        ``"dataset"`` (almost always).
    token : str | None
        HF token; falls back to env vars inside ``upload_chunk_final``.
    dry_run : bool
        Plan + log only; never call the network.
    upload_fn : callable
        Hook for tests to substitute the real ``upload_chunk_final``.

    Returns
    -------
    dict[str, str]
        Final upload state (chunk_id -> hash). Same content as the
        on-disk ``.upload_state.json``.
    """
    assert isinstance(source, str) and source, "source required"
    source_root = Path(output_root) / source
    repo_id = repo_id or f"{repo_prefix}{source}"

    state_path = source_root / _STATE_FILENAME
    state = _load_state(state_path)

    plans = plan_chunks(source_root)
    _LOG.info("source=%s plans=%d repo=%s dry_run=%s", source, len(plans), repo_id, dry_run)

    for plan in plans:
        ch = compute_chunk_hash(plan)
        key = f"chunk-{plan.chunk_id:04d}"
        if state.get(key) == ch:
            _LOG.info("skip %s: hash unchanged (%s)", key, ch)
            continue
        # Pre-flight validation re-runs the file-count + per-file size gate
        val = validate_chunk(plan.folder)
        if not val.ok:
            _LOG.error("chunk %s failed pre-flight: %s", key, val.errors)
            raise RuntimeError(f"chunk validation failed for {key}: {val.errors}")
        # Build allow_patterns from the plan's relative paths so the upload
        # only ships this chunk's files even though plan.folder == source_root.
        allow_patterns = [
            str(f.relative_to(plan.folder)).replace(os.sep, "/")
            for f in plan.files
        ]
        if dry_run:
            _LOG.info(
                "DRY RUN: would upload %s (%d files, %.2f GiB) to %s",
                key, len(plan.files), plan.total_bytes / (1024**3), repo_id,
            )
        else:
            upload_fn(
                folder=plan.folder,
                repo_id=repo_id,
                chunk_id=plan.chunk_id,
                repo_type=repo_type,
                token=token,
                allow_patterns=allow_patterns,
            )
        state[key] = ch
        _save_state(state_path, state)

    return state


def watch_source(
    source: str,
    output_root: Path,
    repo_prefix: str = "<org>/usam-",
    every_minutes: int = 10,
    repo_type: str = "dataset",
    token: str | None = None,
):
    """Long-lived ``CommitScheduler`` for one source. Login-node only.

    See the module docstring for why this MUST NOT run inside a Slurm job.
    """
    source_root = Path(output_root) / source
    source_root.mkdir(parents=True, exist_ok=True)
    repo_id = f"{repo_prefix}{source}"
    return make_commit_scheduler(
        repo_id=repo_id,
        folder=source_root,
        every_minutes=every_minutes,
        repo_type=repo_type,
        token=token,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI driver. ``--watch`` runs the long-lived CommitScheduler."""
    parser = argparse.ArgumentParser(prog="prep.stage_6_upload", description=__doc__)
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset",
                    help="Source name (one A100 node per dataset, per Wave F).")
    ds.add_argument("--source", dest="dataset",
                    help="(deprecated) use --dataset")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-prefix", default="<org>/usam-")
    parser.add_argument("--repo-id", default=None,
                        help="Explicit repo id; overrides --repo-prefix")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true",
                        help="Spawn the long-lived CommitScheduler (login node only). "
                             "Blocks forever; use Ctrl-C to stop.")
    parser.add_argument("--every-minutes", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="Accepted for parity; uploads are always resumable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.watch:
        scheduler = watch_source(
            source=args.dataset,
            output_root=args.output_root,
            repo_prefix=args.repo_prefix,
            every_minutes=args.every_minutes,
            repo_type=args.repo_type,
        )
        _LOG.info("CommitScheduler running; sleeping forever. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            _LOG.info("stopping CommitScheduler")
            try:
                scheduler.stop()  # type: ignore[union-attr]
            except Exception:
                pass
        return 0

    upload_source(
        source=args.dataset,
        output_root=args.output_root,
        repo_prefix=args.repo_prefix,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
