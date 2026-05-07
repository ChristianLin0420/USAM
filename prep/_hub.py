# SPDX-License-Identifier: MIT
"""HF Hub upload helpers for the USAM Phase A pipeline.

Two responsibilities:

1. Wrap ``huggingface_hub.CommitScheduler`` and ``upload_large_folder`` with
   USAM-specific defaults and pre-flight validation. Rejects any chunk that
   violates the per-chunk limits (≤ 1000 files, ≤ 5 GB per file) before a
   single byte goes over the wire — these limits are enforced server-side
   and a violation causes a half-uploaded chunk that's expensive to clean up.

2. Forbid the dangerous-by-default ``Dataset.push_to_hub`` path. We saw this
   blow up on the previous attempt at this pipeline; the function below
   raises a ``RuntimeError`` immediately with a pointer to the right API.

The actual upload daemon ``CommitScheduler`` runs on a **login node**, never
inside a Slurm job. See ``slurm/README.md`` §"Where the upload daemon lives".

# Imports

We do a delayed/optional import of ``huggingface_hub`` so that
``import prep._hub`` works on a node without HF Hub installed (e.g. unit
tests). The actual API calls require the dependency.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huggingface_hub import CommitScheduler

__all__ = [
    "MAX_FILES_PER_CHUNK",
    "MAX_BYTES_PER_FILE",
    "ChunkValidation",
    "validate_chunk",
    "make_commit_scheduler",
    "upload_chunk_final",
    "reject_push_to_hub",
]

logger = logging.getLogger(__name__)

# Hard limits enforced by the Hub. Keep slightly under the published
# ceilings to leave headroom for sidecar files (.gitattributes, etc.).
MAX_FILES_PER_CHUNK: int = 1000
MAX_BYTES_PER_FILE: int = 5 * 1024 * 1024 * 1024  # 5 GiB


@dataclass
class ChunkValidation:
    """Result of pre-flight chunk validation.

    Parameters
    ----------
    ok : bool
        True iff the folder satisfies all per-chunk limits.
    file_count : int
        Number of regular files (recursively) under the chunk dir.
    total_bytes : int
        Sum of file sizes in bytes.
    largest_file : Path | None
        Path of the largest file (None if the folder is empty).
    largest_bytes : int
        Size of ``largest_file`` in bytes.
    errors : list[str]
        Human-readable error messages; empty when ``ok=True``.
    """

    ok: bool
    file_count: int
    total_bytes: int
    largest_file: Path | None
    largest_bytes: int
    errors: list[str]


def validate_chunk(folder: Path) -> ChunkValidation:
    """Walk ``folder`` and check Hub per-chunk limits.

    Counts regular files only (skips dotfiles like ``.gitattributes`` and
    ``done/`` markers since those are bookkeeping, not data). Raises no
    exceptions; pack everything into the returned ``ChunkValidation``.

    Parameters
    ----------
    folder : Path
        Directory whose contents would be uploaded as a single chunk.

    Returns
    -------
    ChunkValidation
    """
    assert isinstance(folder, Path), f"folder must be a Path, got {type(folder).__name__}"
    folder = folder.resolve()
    errors: list[str] = []
    file_count = 0
    total_bytes = 0
    largest_file: Path | None = None
    largest_bytes = 0

    if not folder.exists():
        return ChunkValidation(
            ok=False,
            file_count=0,
            total_bytes=0,
            largest_file=None,
            largest_bytes=0,
            errors=[f"folder does not exist: {folder}"],
        )

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        # Skip bookkeeping files that are not part of the dataset payload.
        if path.name.startswith(".") or "done" in path.relative_to(folder).parts:
            continue
        size = path.stat().st_size
        file_count += 1
        total_bytes += size
        if size > largest_bytes:
            largest_bytes = size
            largest_file = path
        if size > MAX_BYTES_PER_FILE:
            errors.append(
                f"file exceeds {MAX_BYTES_PER_FILE} bytes "
                f"(got {size}): {path.relative_to(folder)}"
            )

    if file_count > MAX_FILES_PER_CHUNK:
        errors.append(
            f"chunk has {file_count} files; limit is {MAX_FILES_PER_CHUNK}. "
            f"Re-shard upstream or split the chunk."
        )

    return ChunkValidation(
        ok=not errors,
        file_count=file_count,
        total_bytes=total_bytes,
        largest_file=largest_file,
        largest_bytes=largest_bytes,
        errors=errors,
    )


def make_commit_scheduler(
    repo_id: str,
    folder: Path,
    every_minutes: int = 10,
    repo_type: str = "dataset",
    private: bool = False,
    token: str | None = None,
) -> CommitScheduler:
    """Construct a ``CommitScheduler`` watching ``folder``.

    The scheduler periodically diffs ``folder`` against the remote repo and
    commits new/changed files. It is meant to run as a long-lived process on
    a login node.

    Parameters
    ----------
    repo_id : str
        e.g. ``"<org>/usam-droid"``.
    folder : Path
        Local dir whose contents mirror the remote repo. Must exist.
    every_minutes : int
        Minutes between commit attempts. Default 10.
    repo_type : str
        "dataset" (almost always) or "model".
    private : bool
        Create the repo as private if it does not exist yet.
    token : str | None
        HF token. Defaults to the ``HUGGINGFACE_TOKEN``/``HF_TOKEN`` env var.

    Returns
    -------
    CommitScheduler
        Caller is responsible for keeping a reference to it. The scheduler
        thread starts on construction.

    Notes
    -----
    Per ``slurm/README.md``, this MUST NOT run inside a Slurm batch job; the
    daemon needs a stable network identity and survives across compute jobs.
    """
    assert isinstance(repo_id, str) and "/" in repo_id, f"repo_id must be 'org/name', got {repo_id!r}"
    folder = Path(folder)
    assert folder.exists(), f"folder must exist before scheduling: {folder}"

    from huggingface_hub import CommitScheduler  # local import — see module docstring

    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    logger.info("Starting CommitScheduler for %s -> %s every %d min", folder, repo_id, every_minutes)
    return CommitScheduler(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        every=every_minutes,
        private=private,
        token=token,
    )


def upload_chunk_final(
    folder: Path,
    repo_id: str,
    chunk_id: int,
    repo_type: str = "dataset",
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Final reconciliation upload for one chunk via ``upload_large_folder``.

    This is called by ``stage_6_upload`` after ``stage_5_validate`` passes
    and is the **only** Hub-write path the validator is allowed to use.

    Parameters
    ----------
    folder : Path
        Local chunk directory. Validated against ``MAX_FILES_PER_CHUNK`` and
        ``MAX_BYTES_PER_FILE`` before upload begins. ``RuntimeError`` if invalid.
    repo_id : str
        Target repo, e.g. ``"<org>/usam-droid"``.
    chunk_id : int
        Numeric chunk id (also encoded in the folder name); only used here
        for log messages.
    repo_type : str
        "dataset" or "model".
    token : str | None
        HF token; falls back to env vars.
    allow_patterns : list[str] | None
        Optional glob patterns restricting what gets uploaded.

    Raises
    ------
    RuntimeError
        If pre-flight validation fails.
    """
    assert isinstance(chunk_id, int) and chunk_id >= 0, f"chunk_id must be a non-negative int, got {chunk_id!r}"
    folder = Path(folder)
    val = validate_chunk(folder)
    if not val.ok:
        raise RuntimeError(
            f"Chunk {chunk_id} at {folder} failed Hub validation; refusing upload.\n"
            + "\n".join("  - " + e for e in val.errors)
        )

    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    from huggingface_hub import upload_large_folder  # local import

    logger.info(
        "upload_large_folder: %s (%d files, %.2f GiB) -> %s",
        folder,
        val.file_count,
        val.total_bytes / (1024**3),
        repo_id,
    )
    upload_large_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        allow_patterns=allow_patterns,
    )


def reject_push_to_hub(*_args: Any, **_kwargs: Any) -> None:
    """Sentinel that explicitly rejects the ``Dataset.push_to_hub`` path.

    The reviewer agent treats any call to ``Dataset.push_to_hub`` as a
    blocking review issue. This helper exists so that pipeline code that
    might be tempted to call it raises with a clear, actionable error
    pointing at ``upload_chunk_final`` instead.

    Raises
    ------
    RuntimeError
        Always.
    """
    raise RuntimeError(
        "Dataset.push_to_hub is forbidden in USAM. "
        "Use prep._hub.upload_chunk_final / make_commit_scheduler instead. "
        "See docs/IMPLEMENTATION_PLAN.md §5.5 for the rationale."
    )
