# SPDX-License-Identifier: MIT
"""Stage 4: fp16 Tri-DINO feature caching (Phase 1 — DROID only).

For each chunk of staged RGB / depth / flow frames, runs
:meth:`usam.encoders.tri_dino.TriDINOTower.extract_features` and writes the
result as memory-mapped safetensors shards consumed by
:class:`usam.dataloader.feature_cache.FeatureCache`.

We **do not** import the encoder at module load: the model-architect's
``usam.encoders.tri_dino`` is being written in parallel. The encoder is loaded
inside :func:`encode_chunk` so simply importing this module never crashes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from usam.dataloader.feature_cache import write_feature_shard

_LOG = logging.getLogger(__name__)


@dataclass
class DinoCacheConfig:
    """Hyperparameters for DINO feature caching.

    Attributes
    ----------
    target_hw : tuple[int, int]
        Inference resolution. ViT-B/14 at 378x378 yields a 27x27=729 patch
        grid (the plan colloquially calls this "384²" but 384 is not divisible
        by 14; 378 keeps the canonical 729-token invariant).
    n_keep_tokens : int
        Number of patch tokens kept per frame after pooling. Default 64
        (downsampled by 3x3 average) plus the [CLS] -> 65 total.
    batch_size : int
    cache_fps : int
        Target output fps. We stride raw frames so the cache is at this fps.
    fp16 : bool
        Always True at runtime; arg kept for parity with the other stages.
    """

    target_hw: tuple[int, int] = (378, 378)
    n_keep_tokens: int = 64
    batch_size: int = 16
    cache_fps: int = 5
    fp16: bool = True


def _load_tri_dino(ckpt_path: Path, dinov3_arch: str = "vit_b_14"):
    """Lazy-load :class:`usam.encoders.tri_dino.TriDINOTower` and place on cuda.

    The model-architect's :class:`TriDINOTower` accepts a :class:`TriDinoConfig`
    dataclass; we wrap the path argument here so callers don't need to import
    that dataclass themselves.
    """
    try:
        from usam.encoders.tri_dino import TriDinoConfig, TriDINOTower
    except ImportError as e:
        raise RuntimeError(
            "usam.encoders.tri_dino is not yet importable; this stage requires "
            "the model-architect's TriDINOTower."
        ) from e
    cfg = TriDinoConfig(dinov3_ckpt=str(ckpt_path), dinov3_arch=dinov3_arch)
    model = TriDINOTower(cfg)
    model.eval().cuda()
    return model


def _stride_to_cache_fps(num_frames: int, source_fps: int, cache_fps: int) -> List[int]:
    """Pick frame indices so the cache lands at exactly ``cache_fps``."""
    assert source_fps > 0 and cache_fps > 0
    stride = max(int(round(source_fps / cache_fps)), 1)
    return list(range(0, num_frames, stride))


def encode_chunk(
    staged_chunk_dir: Path,
    output_root: Path,
    modalities: Iterable[str] = ("rgb", "depth", "flow"),
    cameras: Iterable[str] = ("head_rgb", "wrist_rgb"),
    dinov3_ckpt: Optional[Path] = None,
    source_fps: int = 30,
    config: DinoCacheConfig | None = None,
) -> List[Path]:
    """Encode one chunk's worth of staged frames into per-modality safetensors.

    Output layout (per modality, per camera):

        ``<output_root>/<camera>/<modality>/chunk-XXX/file-YYY.safetensors``

    Returns the list of shard paths actually written.

    The encoder argument is loaded lazily; if ``dinov3_ckpt`` is ``None`` we
    write zero-tensor placeholders of the correct shape. This is what the
    Phase 1 unit test exercises — the real encoder runs only on T1 hosts.
    """
    cfg = config or DinoCacheConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    cams = list(cameras)
    mods = list(modalities)

    encoder = _load_tri_dino(dinov3_ckpt) if dinov3_ckpt is not None else None
    if encoder is None:
        _LOG.warning(
            "no DINOv3 checkpoint provided; stage_4 will write zero-tensor shards "
            "(structural smoke-test mode)"
        )

    written: List[Path] = []
    for cam in cams:
        for mod in mods:
            shard_features: Dict[int, torch.Tensor] = {}
            for ep_dir in sorted(staged_chunk_dir.glob("ep_*")):
                ep_meta_path = ep_dir / "meta.json"
                if not ep_meta_path.exists():
                    continue
                import json

                ep_idx = int(json.loads(ep_meta_path.read_text())["episode_index"])

                # Pick the on-disk file for this (cam, mod) combo
                file_for_modality = {
                    "rgb": ep_dir / f"camera_{cam}.npy",
                    "depth": ep_dir / f"depth_{cam}.npy",
                    "flow": ep_dir / f"flow_{cam}.npy",
                }[mod]
                if not file_for_modality.exists():
                    continue
                arr = np.load(file_for_modality)
                idxs = _stride_to_cache_fps(arr.shape[0], source_fps, cfg.cache_fps)
                if encoder is None:
                    feats = torch.zeros(
                        (len(idxs), cfg.n_keep_tokens + 1, 768), dtype=torch.float16
                    )
                else:  # pragma: no cover - real-runtime path
                    feats = _encode_modality(encoder, arr[idxs], mod, cfg)
                shard_features[ep_idx] = feats

            if not shard_features:
                continue
            chunk_dir = output_root / cam / mod / f"chunk-{0:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            shard_path = chunk_dir / "file-000.safetensors"
            write_feature_shard(shard_path, shard_features)
            written.append(shard_path)
    return written


def _encode_modality(
    encoder, arr: np.ndarray, modality: str, cfg: DinoCacheConfig
) -> torch.Tensor:  # pragma: no cover
    """Run the encoder on a [T, ...] array, returning ``[T, n_keep+1, D]`` fp16."""
    import torch

    out_chunks: List[torch.Tensor] = []
    T = arr.shape[0]
    for start in range(0, T, cfg.batch_size):
        end = min(start + cfg.batch_size, T)
        x = arr[start:end]
        if modality == "rgb":
            t = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous().float() / 255.0
        elif modality == "depth":
            t = torch.as_tensor(x).unsqueeze(1).contiguous().float() / 1000.0
        elif modality == "flow":
            t = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous().float()
        else:
            raise ValueError(f"unknown modality {modality}")
        with torch.no_grad():
            feats = encoder.extract_features(
                t.cuda(), modality=modality, n_keep_tokens=cfg.n_keep_tokens
            )
        out_chunks.append(feats.detach().cpu().to(torch.float16))
    return torch.cat(out_chunks, dim=0)


__all__ = ["DinoCacheConfig", "encode_chunk"]
