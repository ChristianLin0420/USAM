# SPDX-License-Identifier: MIT
"""Roll per-episode staged artifacts into a USAM-LeRobot v2.1 dataset.

Reads from ``<staged_root>/<dataset>/chunk-NNN/ep_*/`` (output of stage_2a's
per-episode staging) and ``<dino_cache_root>/<dataset>/<camera>/<modality>/``
(output of stage_4), produces:

    <out_root>/meta/info.json
    <out_root>/meta/episodes.parquet
    <out_root>/data/chunk-NNN/file-XXX.parquet  (frame-level rows)
    <out_root>/features/<camera>/<modality>/chunk-NNN/file-NNN.safetensors  (symlinks)

This is the input layout consumed by :class:`usam.dataloader.usam_lerobot.USAMLeRobotDataset`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)


def _episode_rows(ep_dir: Path) -> list[dict]:
    """Materialize per-frame parquet rows for one staged episode."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    action_native = np.load(ep_dir / "action_native.npy")
    action_canonical_ee = np.load(ep_dir / "action_canonical_ee.npy")
    state = np.load(ep_dir / "state.npy")
    timestamps = np.load(ep_dir / "timestamps.npy")

    T = action_native.shape[0]
    assert state.shape[0] == T, f"state rows {state.shape[0]} vs action {T}"
    assert timestamps.shape[0] == T
    assert action_canonical_ee.shape[0] == T

    ep_idx = int(meta["episode_index"])
    embodiment = str(meta["embodiment"])
    action_mask = list(meta["action_mask"])
    state_mask = list(meta["state_mask"])
    l1 = list(meta["instructions"].get("level_1") or [""] * T)
    l2 = list(meta["instructions"].get("level_2") or [""] * T)
    l3 = list(meta["instructions"].get("level_3") or [""] * T)
    if len(l1) < T:
        l1 = l1 + [l1[-1] if l1 else ""] * (T - len(l1))
    if len(l2) < T:
        l2 = l2 + [""] * (T - len(l2))
    if len(l3) < T:
        l3 = l3 + [""] * (T - len(l3))

    rows = []
    for t in range(T):
        rows.append(
            {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": float(timestamps[t]),
                "embodiment": embodiment,
                "proprio": state[t].astype(np.float32).tolist(),
                "action_native": action_native[t].astype(np.float32).tolist(),
                "action_canonical_ee": action_canonical_ee[t].astype(np.float32).tolist(),
                "action_mask": action_mask,
                "state_mask": state_mask,
                "level_1": l1[t],
                "level_2": l2[t],
                "level_3": l3[t],
                "subtask_label": False,
            }
        )
    return rows


def _link_features(dino_cache_root: Path, out_root: Path, dataset: str) -> None:
    """Symlink ``dino_cache_root/<dataset>/<cam>/<mod>/...`` into ``out_root/features/<cam>/<mod>/...``."""
    src_base = dino_cache_root / dataset
    if not src_base.exists():
        raise FileNotFoundError(src_base)
    for cam_dir in sorted(p for p in src_base.iterdir() if p.is_dir()):
        for mod_dir in sorted(p for p in cam_dir.iterdir() if p.is_dir()):
            for chunk_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
                tgt = out_root / "features" / cam_dir.name / mod_dir.name / chunk_dir.name
                tgt.mkdir(parents=True, exist_ok=True)
                for shard in chunk_dir.iterdir():
                    if not shard.is_file():
                        continue
                    link = tgt / shard.name
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(shard.resolve())


def assemble(
    staged_root: Path,
    dino_cache_root: Path,
    out_root: Path,
    dataset: str,
    chunk: int,
    fps: int,
    fps_features: int,
    min_episode_length: int = 32,
) -> None:
    chunk_dir = staged_root / dataset / f"chunk-{chunk:03d}"
    assert chunk_dir.exists(), f"missing staged chunk dir {chunk_dir}"

    raw_ep_dirs = sorted(p for p in chunk_dir.glob("ep_*") if p.is_dir())
    # Drop short episodes that can't form a full (history + action_chunk + future) window.
    ep_dirs: list[Path] = []
    skipped = 0
    for p in raw_ep_dirs:
        T = int(np.load(p / "timestamps.npy").shape[0])
        if T < min_episode_length:
            skipped += 1
            continue
        ep_dirs.append(p)
    _LOG.info(
        "assembling %d episodes from %s (skipped %d short < %d frames)",
        len(ep_dirs), chunk_dir, skipped, min_episode_length,
    )
    if not ep_dirs:
        raise SystemExit(2)

    # ---- frame-level rows ------------------------------------------------
    # Preserve the original ``episode_index`` from each staged episode's
    # meta.json. The DINO-cache safetensors are keyed by this original
    # index (``ep_<NNNNNNNN>``); renumbering to a contiguous 0..N-1 here
    # silently breaks the feature lookup for any source whose chunk uses
    # a stride other than 1 (Bridge chunk-0 picks every 256th episode →
    # episode_index ∈ {0, 256, 512, ...}). Earlier versions renumbered
    # and that bug showed up only on Bridge / OXE / AgiBot, never DROID.
    all_rows: list[dict] = []
    episode_meta: list[dict] = []
    for ep in ep_dirs:
        rows = _episode_rows(ep)
        all_rows.extend(rows)
        meta = json.loads((ep / "meta.json").read_text())
        ep_idx = int(meta["episode_index"])
        episode_meta.append(
            {
                "episode_index": ep_idx,
                "length": len(rows),
                "chunk": chunk,
                "file": 0,
                "embodiment": str(meta["embodiment"]),
            }
        )
        # Rows already carry episode_index from `_episode_rows`; nothing to patch.

    # ---- write data/chunk-NNN/file-XXX.parquet ---------------------------
    data_dir = out_root / "data" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "file-000.parquet"
    tbl = pa.Table.from_pylist(all_rows)
    pq.write_table(tbl, str(file_path))
    _LOG.info("wrote %d rows to %s", len(all_rows), file_path)

    # ---- meta/info.json --------------------------------------------------
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": int(fps),
        "fps_features": int(fps_features),
        "source": dataset,
        "embodiment": str(episode_meta[0]["embodiment"]),
        "n_episodes": len(episode_meta),
        "n_frames_per_episode": -1,  # variable length
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    _LOG.info("wrote %s", meta_dir / "info.json")

    # ---- meta/episodes.parquet -------------------------------------------
    eps_tbl = pa.Table.from_pylist(episode_meta)
    pq.write_table(eps_tbl, str(meta_dir / "episodes.parquet"))
    _LOG.info("wrote %s (%d episodes)", meta_dir / "episodes.parquet", len(episode_meta))

    # ---- features symlinks ------------------------------------------------
    _link_features(dino_cache_root, out_root, dataset)
    _LOG.info("linked features under %s", out_root / "features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prep.stage_2a_to_lerobot._assemble")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--dino-cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--fps-features", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    assemble(
        staged_root=args.staged_root,
        dino_cache_root=args.dino_cache_root,
        out_root=args.out_root,
        dataset=args.dataset,
        chunk=args.chunk,
        fps=args.fps,
        fps_features=args.fps_features,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
