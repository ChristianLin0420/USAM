# SPDX-License-Identifier: MIT
"""Unit tests for the USAM-LeRobot dataloader and feature cache.

Loads the ``tiny_droid`` fixture (real or synthesized via
``tests/conftest.py``), checks shapes / dtypes / dict keys, and verifies that
the mmap cache reader returns identical tensors to a full-load path. We also
sanity-check that two ``FeatureCache`` instances opened on the same shard
serve identical data — a proxy for the "two workers, no copy" requirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from usam.dataloader.feature_cache import FeatureCache
from usam.dataloader.usam_lerobot import USAMLeRobotDataset


def test_dataset_loads_and_returns_expected_keys(tiny_droid_root: Path) -> None:
    ds = USAMLeRobotDataset(
        repo_id_or_path=tiny_droid_root,
        use_cached_features=True,
        modalities=["rgb", "depth"],
        cameras=["head_rgb"],
        history_frames=2,
        future_frames=1,
        action_chunk=4,
        fps_features=5,
        fps_action=15,
    )
    assert len(ds) >= 3, f"expected at least one sample per episode, got {len(ds)}"
    sample = ds[0]
    required_scalar_keys = {
        "proprio",
        "state_mask",
        "action_chunk",
        "action_native",
        "action_mask",
        "instruction",
        "task_id",
        "noise_level",
        "subtask_label",
        "episode_index",
        "embodiment",
    }
    for k in required_scalar_keys:
        assert k in sample, f"missing key {k}"

    # Dtype/shape contract ------------------------------------------------
    assert sample["proprio"].shape == (50,), sample["proprio"].shape
    assert sample["proprio"].dtype == torch.float32
    assert sample["action_chunk"].shape == (4, 7), sample["action_chunk"].shape
    assert sample["action_chunk"].dtype == torch.float32
    assert sample["action_native"].shape == (4, 32), sample["action_native"].shape
    assert sample["action_mask"].shape == (32,)
    assert sample["state_mask"].shape == (50,)
    assert isinstance(sample["instruction"], str)
    assert isinstance(sample["task_id"], int)
    assert isinstance(sample["subtask_label"], bool)

    # At least the rgb modality should have produced features.
    assert "rgb_dino_seq" in sample, "rgb feature window missing"
    rgb = sample["rgb_dino_seq"]
    assert rgb.dtype == torch.float16, rgb.dtype
    # T_total = history_frames + future_frames = 3
    assert rgb.shape[0] == 3, rgb.shape
    # n_keep_tokens + 1 (CLS) = 65 in the synthetic fixture
    assert rgb.shape[1] == 65, rgb.shape
    assert rgb.shape[2] == 768, rgb.shape


def test_feature_cache_mmap_returns_consistent_tensors(tiny_droid_root: Path) -> None:
    cache_root = tiny_droid_root / "features" / "head_rgb"
    cache_a = FeatureCache(cache_root, "rgb")
    cache_b = FeatureCache(cache_root, "rgb")
    assert len(cache_a) == len(cache_b)
    assert len(cache_a) >= 1

    ep_idx = next(iter(cache_a.episodes()))
    full_a = cache_a.get(int(ep_idx))
    full_b = cache_b.get(int(ep_idx))
    assert full_a.dtype == torch.float16
    assert full_a.shape == full_b.shape
    assert torch.equal(full_a, full_b), "two readers disagree on the same shard"


def test_feature_cache_partial_read_matches_full_read(tiny_droid_root: Path) -> None:
    cache = FeatureCache(tiny_droid_root / "features" / "head_rgb", "rgb")
    ep_idx = int(next(iter(cache.episodes())))
    full = cache.get(ep_idx)
    n = full.shape[0]
    # Pick a non-trivial subset: first frame, last frame, and one in the middle.
    indices = torch.tensor([0, n - 1, n // 2], dtype=torch.long)
    partial = cache.get(ep_idx, indices)
    assert partial.shape == (3, full.shape[1], full.shape[2]), partial.shape
    assert torch.equal(partial[0], full[0])
    assert torch.equal(partial[1], full[n - 1])
    assert torch.equal(partial[2], full[n // 2])


def test_feature_cache_uses_mmap_safe_open(tiny_droid_root: Path) -> None:
    """Spot-check that the on-disk shard is readable via the safetensors
    ``device='cpu'`` mmap path that ``FeatureCache`` relies on."""
    shard = tiny_droid_root / "features" / "head_rgb" / "rgb" / "chunk-000" / "file-000.safetensors"
    assert shard.exists(), f"expected feature shard at {shard}"
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert any(k.startswith("ep_") for k in keys), f"no episode keys in {shard}"


def test_use_cached_features_false_streaming_is_blocked(tiny_droid_root: Path) -> None:
    """Phase 1 explicitly does not wire streaming-decode; the loader must
    refuse rather than silently fall back."""
    ds = USAMLeRobotDataset(
        repo_id_or_path=tiny_droid_root,
        use_cached_features=False,
        modalities=["rgb"],
        cameras=["head_rgb"],
        history_frames=1,
        future_frames=0,
        action_chunk=2,
        fps_features=5,
        fps_action=15,
        streaming_encoder=None,
    )
    with pytest.raises(NotImplementedError):
        _ = ds[0]
