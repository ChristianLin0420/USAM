# SPDX-License-Identifier: MIT
"""CheckpointedJob: per-episode idempotent processing with SIGUSR1 graceful exit.

This module provides the ``CheckpointedJob`` abstract base class that all USAM
Phase A pipeline stages subclass. It enforces three invariants that the rest
of the pipeline relies on:

1. **Per-episode idempotency.** Output filenames embed a 12-char SHA-256 hash
   of the ``episode_id``. Re-running a job over an already-processed episode
   is a no-op: the existing file on disk wins and we skip the work.

2. **Crash-safe resume across Slurm preemption.** The job listens for
   ``SIGUSR1`` (Slurm's pre-walltime warning, see ``--signal=B:USR1@600`` in
   ``slurm/job.sbatch``). When the signal fires, the in-flight episode is
   allowed to finish writing its shard, the `_flush()` hook lets subclasses
   persist any in-memory accumulator, and the process exits with code
   ``124``. The bash wrapper interprets exit code 124 as "please requeue".

3. **Single source of truth for shard naming.** Every subclass writes its
   per-chunk output through ``write_shard``, which by default lives at
   ``<output_root>/<source>/<stage>/chunk-<NNN>/file-<hash>.<ext>``.

# Subclassing contract

Subclasses **must** implement these three methods:

    list_episodes(self) -> Iterable[EpisodeRef]
    convert_episode(self, ref: EpisodeRef) -> ConversionResult
    write_shard(self, results: list[ConversionResult]) -> Path

They **must not** override ``run()`` or the signal-handling internals. They
**may** override ``_flush()`` to persist a partial-shard buffer when
preemption hits; the default ``_flush()`` is a no-op.

Subclasses get these helpers for free:

* ``self.episode_hash(episode_id) -> str`` — 12-char hex digest, deterministic
* ``self.is_done(ref) -> bool`` — has the per-episode output already been
  produced (checked via the marker file ``done/<hash>.ok``)
* ``self.mark_done(ref) -> None`` — write that marker
* ``self.output_dir -> Path`` — pre-created destination for this chunk
* ``self.scratch_dir -> Path`` — local-only working area; never uploaded

# Signal handling

We register a single SIGUSR1 handler in ``run()``. It does **not** raise from
inside the signal handler (that breaks libraries that catch broad
exceptions). Instead it flips ``self._stop_requested = True`` and ``run()``
checks it at the top of every loop iteration. This gives the in-flight
``convert_episode`` call time to finish or, when it returns,
``_flush(partial=True)`` is called and we ``sys.exit(124)``.

# Imports

This file deliberately has zero heavy dependencies (``numpy``, ``torch``,
``pyarrow``, ``huggingface_hub`` are all forbidden here) so that
``import prep._base`` stays cheap and robust on a freshly-allocated Slurm
node before the science deps are loaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

__all__ = [
    "EpisodeRef",
    "ConversionResult",
    "CheckpointedJob",
    "PREEMPT_EXIT_CODE",
]

# Slurm wrapper interprets 124 as "please requeue". This constant is
# referenced by ``slurm/job.sbatch`` — do not change without updating both.
PREEMPT_EXIT_CODE: int = 124

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeRef:
    """Lightweight reference to a single source episode.

    Parameters
    ----------
    episode_id : str
        Source-unique stable identifier. MUST be deterministic across reruns
        (the same episode produces the same id every time). The 12-char
        hash derived from this id appears in output filenames, so changing
        it invalidates idempotency.
    source : str
        One of {"droid", "agibot2026", "rh20t", "robomind", "bridge",
        "oxe_auge"}.
    raw_path : str
        Absolute path on the local scratch filesystem to the raw episode
        (a TFDS shard, an HDF5 file, etc.). String, not Path, so the dataclass
        stays trivially serializable to JSON.
    extra : dict
        Free-form per-source metadata (camera serials, embodiment variant,
        etc.). The framework does not interpret it.
    """

    episode_id: str
    source: str
    raw_path: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Per-episode payload returned by ``convert_episode``.

    The exact field set is dictated by the canonical ConversionResult in
    ``docs/IMPLEMENTATION_PLAN.md`` §5.2 and is owned by ``data-engineer``;
    here we keep the dataclass intentionally permissive (everything is
    ``Any``) so subclasses can attach numpy/torch arrays without making
    ``prep._base`` import them.

    Parameters
    ----------
    episode_index : int
        Monotonically increasing index assigned by ``stage_1_index``.
    episode_id : str
        Mirrors ``EpisodeRef.episode_id`` so downstream stages can rederive
        the hash without consulting the manifest.
    payload : dict
        Source-specific tensors and metadata. ``data-engineer`` defines its
        schema in ``stage_2a_to_lerobot/<source>.py``. Treated as opaque here.
    """

    episode_index: int
    episode_id: str
    payload: dict[str, Any] = field(default_factory=dict)


class CheckpointedJob(ABC):
    """Abstract base for every per-chunk USAM Phase A pipeline job.

    See module docstring for the contract. The minimal subclass looks like::

        class DroidConverter(CheckpointedJob):
            def list_episodes(self):
                for shard in self.scratch_dir.glob("*.tfrecord"):
                    yield EpisodeRef(episode_id=shard.stem, source="droid",
                                     raw_path=str(shard))

            def convert_episode(self, ref):
                arr = decode_tfrecord(ref.raw_path)
                return ConversionResult(episode_index=...,
                                        episode_id=ref.episode_id,
                                        payload={"frames": arr})

            def write_shard(self, results):
                out = self.output_dir / f"file-{self.shard_hash(results)}.parquet"
                write_parquet(out, results)
                return out

    Parameters
    ----------
    source : str
        e.g. ``"droid"``.
    stage : str
        e.g. ``"stage_2a_to_lerobot"``.
    chunk : int
        0-indexed chunk identifier.
    output_root : Path
        Root directory for finalized outputs (a subdir per source/stage/chunk
        is created beneath it).
    scratch_root : Path | None
        Local-only working dir. Defaults to ``$SCRATCH/usam`` if set, else
        ``output_root / "_scratch"``.
    shard_size : int
        Number of episodes accumulated in memory before ``write_shard`` is
        called. Default 256 keeps RAM bounded.
    """

    def __init__(
        self,
        source: str,
        stage: str,
        chunk: int,
        output_root: Path,
        scratch_root: Path | None = None,
        shard_size: int = 256,
    ) -> None:
        assert isinstance(chunk, int) and chunk >= 0, f"chunk must be a non-negative int, got {chunk!r}"
        self.source = source
        self.stage = stage
        self.chunk = chunk
        self.shard_size = shard_size

        self.output_root = Path(output_root)
        self.output_dir = self.output_root / source / stage / f"chunk-{chunk:03d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scratch = scratch_root if scratch_root is not None else Path(os.environ.get("SCRATCH", str(self.output_root / "_scratch")))
        self.scratch_dir = Path(scratch) / source / stage / f"chunk-{chunk:03d}"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self._done_dir = self.output_dir / "done"
        self._done_dir.mkdir(parents=True, exist_ok=True)

        self._stop_requested: bool = False
        self._buffer: list[ConversionResult] = []
        self._previous_handler: Any = None

    # ------------------------------------------------------------------
    # Abstract API the subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def list_episodes(self) -> Iterable[EpisodeRef]:
        """Yield every ``EpisodeRef`` in the chunk this job is responsible for.

        Implementations should be lazy (a generator) so memory does not scale
        with chunk size. Order must be deterministic to keep idempotency
        meaningful — sort by ``episode_id`` if the source iteration order is
        not stable.
        """

    @abstractmethod
    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        """Convert one source episode into a ``ConversionResult``.

        MUST be pure with respect to ``ref``: the same ref always produces the
        same result. MUST NOT write to ``self.output_dir`` directly — that's
        ``write_shard``'s job. MAY write temporary files inside
        ``self.scratch_dir``.
        """

    @abstractmethod
    def write_shard(self, results: list[ConversionResult]) -> Path:
        """Persist a batch of converted episodes to one shard file.

        Implementations MUST embed a content-addressable hash in the filename
        (use ``self.shard_hash(results)`` for the canonical form) so that two
        concurrent runs cannot collide. MUST be the only place writes to
        ``self.output_dir`` happen. Returns the path to the written shard.
        """

    # ------------------------------------------------------------------
    # Idempotency helpers (free for subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def episode_hash(episode_id: str) -> str:
        """Return the 12-character SHA-256 prefix used for filename hashing.

        ``hashlib.sha256(episode_id.encode()).hexdigest()[:12]`` — verbatim,
        as specified by the implementation plan. 12 hex chars give 48 bits
        of collision resistance, enough for the ~100M episode upper bound.
        """
        assert isinstance(episode_id, str) and episode_id, "episode_id must be a non-empty string"
        return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12]

    def shard_hash(self, results: list[ConversionResult]) -> str:
        """Hash the sorted list of episode ids in a shard for filename use.

        Two runs that produce the same set of episodes (regardless of
        in-memory order) yield the same shard hash. This guarantees that
        rerunning a chunk emits a byte-identical filename, which is what
        ``upload_large_folder`` needs to deduplicate on the Hub.
        """
        assert isinstance(results, list), "results must be a list"
        ids = sorted(r.episode_id for r in results)
        h = hashlib.sha256(("\n".join(ids)).encode("utf-8")).hexdigest()[:12]
        return h

    def _marker_path(self, ref: EpisodeRef) -> Path:
        return self._done_dir / f"{self.episode_hash(ref.episode_id)}.ok"

    def is_done(self, ref: EpisodeRef) -> bool:
        """True iff ``ref`` was already processed in a previous run.

        Backed by an empty marker file at ``done/<hash>.ok``. We deliberately
        do not check the shard file itself because shards are aggregated
        across many episodes; the per-episode marker is the only correct
        unit of idempotency.
        """
        return self._marker_path(ref).exists()

    def mark_done(self, ref: EpisodeRef) -> None:
        """Atomically record that ``ref`` has been processed."""
        marker = self._marker_path(ref)
        tmp = marker.with_suffix(".ok.tmp")
        tmp.write_text(json.dumps({"episode_id": ref.episode_id, "source": self.source}))
        os.replace(tmp, marker)

    # ------------------------------------------------------------------
    # Optional flush hook
    # ------------------------------------------------------------------

    def _flush(self, partial: bool = False) -> Path | None:
        """Flush the in-memory buffer to a shard file. Returns the path written.

        Called automatically when ``len(self._buffer) >= self.shard_size`` and
        once more on graceful exit. ``partial=True`` indicates the flush is
        happening because of SIGUSR1 (so subclasses can mark the shard as
        partial in the manifest if they care).
        """
        if not self._buffer:
            return None
        results = self._buffer
        self._buffer = []
        path = self.write_shard(results)
        if partial:
            logger.info("Flushed partial shard (%d episodes) to %s", len(results), path)
        else:
            logger.info("Flushed shard (%d episodes) to %s", len(results), path)
        for r in results:
            # The marker is the source of truth for idempotency; only after
            # the shard write succeeds do we record the episodes as done.
            self.mark_done(EpisodeRef(episode_id=r.episode_id, source=self.source, raw_path=""))
        return path

    # ------------------------------------------------------------------
    # Signal plumbing
    # ------------------------------------------------------------------

    def _on_sigusr1(self, signum: int, frame: FrameType | None) -> None:
        # Keep the handler trivial; no IO, no exceptions. ``run()`` polls
        # ``self._stop_requested`` between episodes.
        self._stop_requested = True

    def _install_signal_handler(self) -> None:
        self._previous_handler = signal.signal(signal.SIGUSR1, self._on_sigusr1)

    def _restore_signal_handler(self) -> None:
        if self._previous_handler is not None:
            try:
                signal.signal(signal.SIGUSR1, self._previous_handler)
            except (ValueError, TypeError):
                # Happens when called from a non-main thread; safe to ignore.
                pass
            self._previous_handler = None

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Main loop. Returns 0 on success or raises ``SystemExit(124)`` on preempt.

        Subclasses normally do not override this. The loop:

        1. Iterate over ``list_episodes()``.
        2. Skip episodes whose marker exists (``is_done``).
        3. Between episodes, check ``self._stop_requested`` — if set, flush
           the partial buffer and exit 124.
        4. Call ``convert_episode``, append to buffer.
        5. When buffer reaches ``self.shard_size``, flush.
        6. After the loop, flush the tail buffer, restore the signal
           handler, return 0.
        """
        self._install_signal_handler()
        try:
            for ref in self.list_episodes():
                if self._stop_requested:
                    logger.warning("SIGUSR1 received before episode %s; flushing and exiting 124", ref.episode_id)
                    self._flush(partial=True)
                    sys.exit(PREEMPT_EXIT_CODE)
                if self.is_done(ref):
                    continue
                try:
                    result = self.convert_episode(ref)
                except Exception:
                    logger.exception("convert_episode failed for %s; re-raising", ref.episode_id)
                    self._flush(partial=True)
                    raise
                self._buffer.append(result)
                if len(self._buffer) >= self.shard_size:
                    self._flush(partial=False)
                if self._stop_requested:
                    logger.warning("SIGUSR1 received after episode %s; flushing and exiting 124", ref.episode_id)
                    self._flush(partial=True)
                    sys.exit(PREEMPT_EXIT_CODE)
            # Tail flush
            self._flush(partial=False)
            return 0
        finally:
            self._restore_signal_handler()
