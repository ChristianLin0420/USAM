# SPDX-License-Identifier: MIT
"""Sharding correctness for stage_4_dino_cache.encode_chunk_multigpu.

These tests do NOT need a real DINOv3 encoder — they exercise the
multi-process episode partitioning by passing dinov3_ckpt=None (which
takes stage_4's placeholder/zero-tensor path) and asserting:

* Each rank writes exactly one shard file named file-{rank:03d}.safetensors.
* The union of episode_idxs across all rank shards equals the input set.
* Episode partition is disjoint (no episode appears in two shards).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _build_synthetic_chunk(chunk_dir: Path, num_episodes: int, num_frames: int = 8) -> None:
    """Materialize ``num_episodes`` ep_*/ subdirs with the on-disk layout
    that ``encode_chunk`` expects for one (camera, modality) combo."""
    for ep_idx in range(num_episodes):
        ep_dir = chunk_dir / f"ep_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # camera_head_rgb.npy: [T, H, W, 3] uint8
        np.save(ep_dir / "camera_head_rgb.npy",
                np.zeros((num_frames, 32, 32, 3), dtype=np.uint8))
        (ep_dir / "meta.json").write_text(json.dumps({"episode_index": ep_idx}))


def test_sharding_partitions_episodes_disjoint(tmp_path: Path) -> None:
    """world_size=4 across 10 episodes ⇒ 4 shard files, partition disjoint."""
    from prep.stage_4_dino_cache import encode_chunk_multigpu

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_synthetic_chunk(staged, num_episodes=10)

    encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=None,           # placeholder path — no GPU needed
        source_fps=30,
        world_size=4,
    )
    # Filter to the (cam, mod) combo we built.
    rgb_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(rgb_dir.glob("file-*.safetensors"))
    assert [s.name for s in shards] == [
        "file-000.safetensors", "file-001.safetensors",
        "file-002.safetensors", "file-003.safetensors",
    ]
    # Read the safetensors back; key names encode episode_index.
    from usam.dataloader.feature_cache import read_feature_shard

    seen: set[int] = set()
    for shard in shards:
        ep_to_tensor = read_feature_shard(shard)
        for ep_idx in ep_to_tensor.keys():
            ep_int = int(ep_idx)
            assert ep_int not in seen, f"duplicate ep_idx={ep_int} across shards"
            seen.add(ep_int)
    assert seen == set(range(10))


def test_sharding_with_world_size_larger_than_episodes(tmp_path: Path) -> None:
    """world_size=8 across 3 episodes ⇒ first 3 ranks write shards, others empty."""
    from prep.stage_4_dino_cache import encode_chunk_multigpu

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_synthetic_chunk(staged, num_episodes=3)

    encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=None,
        source_fps=30,
        world_size=8,
    )
    rgb_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(rgb_dir.glob("file-*.safetensors"))
    # At most 3 shards because there are 3 episodes; ranks 3..7 produce nothing.
    assert 1 <= len(shards) <= 3
    from usam.dataloader.feature_cache import read_feature_shard

    seen: set[int] = set()
    for shard in shards:
        seen |= {int(k) for k in read_feature_shard(shard).keys()}
    assert seen == {0, 1, 2}
