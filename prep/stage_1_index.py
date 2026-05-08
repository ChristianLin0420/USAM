# SPDX-License-Identifier: MIT
"""Stage 1: episode-listing manifest builder.

For each source we enumerate raw episodes (one row per episode) and emit a
``manifest.parquet`` (or ``manifest.jsonl`` fallback if ``pyarrow`` is
absent) with columns:

    episode_id        : str   - source-unique stable identifier
    episode_idx       : int64 - dense, monotonically increasing
    source            : str
    embodiment        : str   - one of the embodiment.json keys
    raw_path          : str   - absolute path to the raw episode file/dir
    expected_n_frames : int64 - hint for downstream stages, may be 0 if unknown
    hash              : str   - 12-char sha256 prefix of episode_id

The manifest is the source-of-truth for stage_2a; downstream stages also
join against it on ``episode_idx`` and ``hash``.

CheckpointedJob subclass
------------------------

Manifest building is itself preemption-safe: if a chunk's worth of
enumeration is interrupted, the partial rows are flushed to a
``manifest.partial-<hash>.parquet`` and the next run continues from the
next un-listed episode. We use the standard ``CheckpointedJob`` plumbing
even though there is no per-episode "conversion" — each episode reference
is its own ``ConversionResult`` with the index payload.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prep._base import CheckpointedJob, ConversionResult, EpisodeRef

__all__ = ["IndexJob", "build_index_for_source", "main"]

_LOG = logging.getLogger(__name__)

INDEX_COLUMNS: list[str] = [
    "episode_id",
    "episode_idx",
    "source",
    "embodiment",
    "raw_path",
    "expected_n_frames",
    "hash",
]


# --------------------------------------------------------------------------- #
# Per-source enumerators                                                      #
# --------------------------------------------------------------------------- #
#
# Each enumerator yields ``(episode_id, embodiment, raw_path, n_frames)``
# tuples. They never touch the network: HF / TFDS reads from the raw cache
# directory only. A sentinel "noop" path is provided for unit tests so the
# indexer can run on a synthetic raw cache.

@dataclass(frozen=True)
class _RawEpisode:
    episode_id: str
    embodiment: str
    raw_path: str
    n_frames: int


def _enumerate_local_dir(
    source: str,
    embodiment: str,
    raw_root: Path,
    glob: str,
) -> Iterator[_RawEpisode]:
    """Generic enumerator: every file matching ``glob`` is one episode."""
    if not raw_root.exists():
        _LOG.warning("raw_root for %s missing at %s; manifest will be empty", source, raw_root)
        return
    for entry in sorted(raw_root.rglob(glob)):
        if not entry.is_file():
            continue
        yield _RawEpisode(
            episode_id=f"{source}::{entry.relative_to(raw_root)}",
            embodiment=embodiment,
            raw_path=str(entry.resolve()),
            n_frames=0,
        )


def _enumerate_droid(raw_root: Path) -> Iterator[_RawEpisode]:
    """DROID enumeration. Tries TFDS metadata; falls back to file scan."""
    try:
        import tensorflow_datasets as tfds  # type: ignore[import-not-found]
    except ImportError:
        tfds = None  # type: ignore[assignment]

    tfds_dir = raw_root / "tfds"
    if tfds is not None and tfds_dir.exists():
        try:
            builder = tfds.builder("droid", data_dir=str(tfds_dir))
            n_total = int(builder.info.splits["train"].num_examples)
            for i in range(n_total):
                yield _RawEpisode(
                    episode_id=f"droid::{i}",
                    embodiment="droid_franka",
                    raw_path=str(tfds_dir.resolve()),
                    n_frames=0,
                )
            return
        except Exception as exc:  # pragma: no cover
            _LOG.warning("TFDS DROID enumeration failed (%s); falling back to local scan", exc)
    yield from _enumerate_local_dir("droid", "droid_franka", raw_root, "*.tfrecord*")


def _enumerate_agibot(raw_root: Path) -> Iterator[_RawEpisode]:
    yield from _enumerate_local_dir("agibot2026", "agibot_g1", raw_root, "*.parquet")


def _enumerate_rh20t(raw_root: Path) -> Iterator[_RawEpisode]:
    if not raw_root.exists():
        return
    for cfg in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for traj in sorted(p for p in cfg.iterdir() if p.is_dir()):
            yield _RawEpisode(
                episode_id=f"rh20t::{cfg.name}::{traj.name}",
                embodiment="rh20t_franka",
                raw_path=str(traj.resolve()),
                n_frames=0,
            )


def _enumerate_robomind(raw_root: Path) -> Iterator[_RawEpisode]:
    yield from _enumerate_local_dir("robomind", "robomind_tien_kung", raw_root, "*.h5")


def _enumerate_bridge(raw_root: Path) -> Iterator[_RawEpisode]:
    yield from _enumerate_local_dir("bridge", "bridge_widowx", raw_root, "*.tfrecord*")


def _enumerate_oxe_auge(raw_root: Path) -> Iterator[_RawEpisode]:
    if not raw_root.exists():
        return
    for sub in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for shard in sorted(sub.rglob("*.tfrecord*")):
            yield _RawEpisode(
                episode_id=f"oxe_auge::{sub.name}::{shard.name}",
                embodiment="oxe_auge_generic",
                raw_path=str(shard.resolve()),
                n_frames=0,
            )


_ENUMERATORS = {
    "droid": _enumerate_droid,
    "agibot2026": _enumerate_agibot,
    "rh20t": _enumerate_rh20t,
    "robomind": _enumerate_robomind,
    "bridge": _enumerate_bridge,
    "oxe_auge": _enumerate_oxe_auge,
}


# --------------------------------------------------------------------------- #
# IndexJob                                                                    #
# --------------------------------------------------------------------------- #


class IndexJob(CheckpointedJob):
    """Build ``manifest.parquet`` for one ``(source, chunk)``.

    Parameters
    ----------
    source : str
    chunk : int
    output_root : Path
        Manifests land at ``output_root/<source>/stage_1_index/chunk-XXX/``.
    raw_root : Path
        Local raw-data cache produced by the matching ``stage_0_download``.
    enumerator : callable | None
        Override for testing; defaults to the per-source dispatch above.
    """

    def __init__(
        self,
        source: str,
        chunk: int,
        output_root: Path,
        raw_root: Path,
        enumerator: Any = None,
        scratch_root: Path | None = None,
        shard_size: int = 1024,
    ) -> None:
        super().__init__(
            source=source,
            stage="stage_1_index",
            chunk=chunk,
            output_root=Path(output_root),
            scratch_root=scratch_root,
            shard_size=shard_size,
        )
        self.raw_root = Path(raw_root)
        self._enumerator = enumerator if enumerator is not None else _ENUMERATORS.get(source)
        if self._enumerator is None:
            raise KeyError(f"no enumerator registered for source {source!r}")

    def list_episodes(self) -> Iterable[EpisodeRef]:
        """Yield one ``EpisodeRef`` per discovered raw episode."""
        for raw in self._enumerator(self.raw_root):
            yield EpisodeRef(
                episode_id=raw.episode_id,
                source=self.source,
                raw_path=raw.raw_path,
                extra={"embodiment": raw.embodiment, "expected_n_frames": int(raw.n_frames)},
            )

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        """Trivially package the ref into a ``ConversionResult`` row."""
        payload = {
            "episode_id": ref.episode_id,
            "source": self.source,
            "embodiment": str(ref.extra.get("embodiment", "")),
            "raw_path": ref.raw_path,
            "expected_n_frames": int(ref.extra.get("expected_n_frames", 0)),
            "hash": self.episode_hash(ref.episode_id),
        }
        return ConversionResult(episode_index=-1, episode_id=ref.episode_id, payload=payload)

    def write_shard(self, results: list[ConversionResult]) -> Path:
        """Write a manifest shard. Uses ``self.shard_hash(results)`` for naming."""
        rows: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            row = dict(r.payload)
            row["episode_idx"] = i  # reassigned at merge time below
            rows.append(row)
        shard_h = self.shard_hash(results)
        out = self.output_dir / f"manifest.shard-{shard_h}.parquet"
        _write_rows(out, rows)
        return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` as parquet, falling back to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        jsonl = path.with_suffix(".jsonl")
        with jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return
    if not rows:
        # An empty schema-only parquet trips pyarrow's type inference on some
        # versions; emit a sentinel JSON instead so the merge step can still
        # detect the file but skips it without raising.
        path.with_suffix(".empty.json").write_text(json.dumps({"rows": 0}))
        return
    cols: dict[str, list[Any]] = {col: [] for col in INDEX_COLUMNS}
    for row in rows:
        for col in INDEX_COLUMNS:
            cols[col].append(row.get(col))
    pq.write_table(pa.Table.from_pydict(cols), str(path))


def build_index_for_source(
    source: str,
    raw_root: Path,
    output_root: Path,
    enumerator: Any = None,
    chunk: int = 0,
) -> Path:
    """Top-level helper: run a single ``IndexJob`` and merge its shards.

    Returns the path to the canonical ``manifest.parquet`` (or ``.jsonl``
    fallback) for the source.
    """
    assert isinstance(source, str) and source, "source must be a non-empty str"
    job = IndexJob(
        source=source,
        chunk=chunk,
        output_root=Path(output_root),
        raw_root=Path(raw_root),
        enumerator=enumerator,
    )
    job.run()
    return _merge_manifest_shards(job.output_dir, source=source)


def _merge_manifest_shards(chunk_dir: Path, source: str) -> Path:
    """Concatenate ``manifest.shard-*.parquet`` into one manifest, dense-indexed.

    Idempotent: rerunning over the same chunk dir yields the same merged
    file content (modulo the wall-clock; rows are sorted by episode_id).
    """
    shards = sorted(chunk_dir.glob("manifest.shard-*.parquet"))
    if not shards:
        # Maybe we fell back to JSONL.
        shards = sorted(chunk_dir.glob("manifest.shard-*.jsonl"))
    if not shards:
        # Empty manifest still useful — write a sentinel JSON so the
        # dispatcher can distinguish "no episodes" from "stage didn't run".
        out = chunk_dir / "manifest.empty.json"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": 0, "source": source}))
        return out

    rows: list[dict[str, Any]] = []
    for shard in shards:
        if shard.suffix == ".jsonl":
            for line in shard.read_text().splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        else:
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError:
                continue
            tbl = pq.read_table(str(shard))
            rows.extend(tbl.to_pylist())

    # Dedup by episode_id, sort, and re-index.
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen[str(row["episode_id"])] = row
    merged = sorted(seen.values(), key=lambda r: str(r["episode_id"]))
    for i, row in enumerate(merged):
        row["episode_idx"] = i
        row["source"] = row.get("source", source)

    out = chunk_dir / "manifest.parquet"
    _write_rows(out, merged)
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """``python -m prep.stage_1_index --source droid --raw /scratch/.../raw --out /scratch/.../manifests``."""
    parser = argparse.ArgumentParser(prog="prep.stage_1_index")
    parser.add_argument("--source", required=True, choices=sorted(_ENUMERATORS.keys()))
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="kept for parity with other stages; IndexJob is always resumable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    manifest = build_index_for_source(args.source, args.raw, args.out, chunk=args.chunk)
    print(f"manifest written: {manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
