# SPDX-License-Identifier: MIT
"""Real-DINOv3-weights smoke for stage_4_dino_cache.

Skipped unless ``USAM_DINOV3_CKPT`` is set in the environment, e.g.::

    USAM_DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m \
        pytest tests/integration/test_dino_cache_real_weights.py

Designed to run inside the prep Docker image where the gated weights
are baked at /opt/dinov3-cache; passing the HF model id resolves
locally because TRANSFORMERS_OFFLINE=1 is set in the image.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

DINOV3_CKPT = os.environ.get("USAM_DINOV3_CKPT")

pytestmark = pytest.mark.skipif(
    not DINOV3_CKPT,
    reason="USAM_DINOV3_CKPT not set; needs a real DINOv3 checkpoint to run",
)


def _build_two_episode_chunk(chunk_dir: Path, num_frames: int = 4) -> None:
    """Two episodes of ``num_frames`` random RGB frames each, 64×64."""
    rng = np.random.default_rng(0)
    for ep_idx in range(2):
        ep_dir = chunk_dir / f"ep_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "camera_head_rgb.npy",
                rng.integers(0, 256, (num_frames, 64, 64, 3), dtype=np.uint8))
        (ep_dir / "meta.json").write_text(json.dumps({"episode_index": ep_idx}))


def test_real_weights_produces_nonzero_shards(tmp_path: Path) -> None:
    """End-to-end: load real DINOv3, encode 2 episodes, assert non-zero output."""
    import torch
    from prep.stage_4_dino_cache import (
        DinoCacheConfig,
        encode_chunk_multigpu,
    )
    from usam.dataloader.feature_cache import read_feature_shard

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_two_episode_chunk(staged)

    world_size = max(1, min(2, torch.cuda.device_count()))
    cfg = DinoCacheConfig(
        target_hw=(448, 448),
        n_keep_tokens=64,
        embed_dim=1024 if "vitl16" in (DINOV3_CKPT or "") else 768,
        batch_size=2,
        cache_fps=5,
    )
    encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=Path(DINOV3_CKPT),
        source_fps=10,
        world_size=world_size,
        config=cfg,
    )
    chunk_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(chunk_dir.glob("file-*.safetensors"))
    assert shards, "no shards written"

    saw_nonzero = False
    for shard in shards:
        ep_to_tensor = read_feature_shard(shard)
        for ep_idx, t in ep_to_tensor.items():
            # [T, n_keep+1, embed_dim]
            assert t.dim() == 3, t.shape
            assert t.shape[1] == cfg.n_keep_tokens + 1, t.shape
            assert t.shape[2] == cfg.embed_dim, t.shape
            assert t.dtype == torch.float16
            if t.abs().sum() > 0:
                saw_nonzero = True
    assert saw_nonzero, "all shards are zero — encoder forward did not run"
