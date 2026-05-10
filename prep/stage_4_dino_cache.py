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
        Inference resolution. ViT-B/16 (or ViT-L/16) at 448x448 yields a
        28x28 = 784 patch grid; the cache keeps the first ``n_keep_tokens``
        patches plus [CLS].
    n_keep_tokens : int
        Number of patch tokens kept per frame. Default 64; with [CLS] prepended
        the on-disk shard's per-frame token dimension is 65.
    embed_dim : int
        Hidden dim of the encoder. 768 for ViT-B/16, 1024 for ViT-L/16. Used
        only by the placeholder (encoder=None) path; the real path reads
        the dim off the encoder.
    batch_size : int
    cache_fps : int
        Target output fps. We stride raw frames so the cache is at this fps.
    fp16 : bool
        Always True at runtime; arg kept for parity with the other stages.
    """

    target_hw: tuple[int, int] = (448, 448)
    n_keep_tokens: int = 64
    embed_dim: int = 768
    batch_size: int = 16
    cache_fps: int = 5
    fp16: bool = True


def _load_tri_dino(ckpt_path: Path, dinov3_arch: str = "vit_b_16"):
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
    shard_id: int = 0,
) -> List[Path]:
    """Encode one chunk's worth of staged frames into per-modality safetensors.

    Output layout (per modality, per camera):

        ``<output_root>/<camera>/<modality>/chunk-XXX/file-YYY.safetensors``

    Returns the list of shard paths actually written.

    The encoder argument is loaded lazily; if ``dinov3_ckpt`` is ``None`` we
    write zero-tensor placeholders of the correct shape. This is what the
    Phase 1 unit test exercises — the real encoder runs only on T1 hosts.

    ``shard_id`` controls the output filename suffix: rank ``r`` writes
    ``file-{r:03d}.safetensors`` so multi-rank workers don't clobber each
    other when sharing one ``output_root``.
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
                        (len(idxs), cfg.n_keep_tokens + 1, cfg.embed_dim),
                        dtype=torch.float16,
                    )
                else:  # pragma: no cover - real-runtime path
                    feats = _encode_modality(encoder, arr[idxs], mod, cfg)
                shard_features[ep_idx] = feats

            if not shard_features:
                continue
            chunk_dir = output_root / cam / mod / f"chunk-{0:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            shard_path = chunk_dir / f"file-{shard_id:03d}.safetensors"
            write_feature_shard(shard_path, shard_features)
            written.append(shard_path)
    return written


def _encode_modality(
    encoder, arr: np.ndarray, modality: str, cfg: DinoCacheConfig
) -> torch.Tensor:  # pragma: no cover
    """Run the encoder on a [T, ...] array, returning ``[T, n_keep+1, D]`` fp16.

    Frames are bilinearly resized to ``cfg.target_hw`` before the encoder
    call so the encoder always sees its expected input resolution
    (default 448x448 for DINOv3-ViT-{B,L}/16). This decouples data
    staging HW from encoder HW; staged data may be at e.g. 378x378.
    """
    import torch
    import torch.nn.functional as F

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
            # Magnitude-scaling correction is deferred; the flow
            # patch_embed is retrained via Phase A.5 adapter pretraining,
            # so absolute flow magnitudes here are not load-bearing.
            t = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous().float()
        else:
            raise ValueError(f"unknown modality {modality}")
        if t.shape[-2:] != tuple(cfg.target_hw):
            t = F.interpolate(t, size=cfg.target_hw, mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = encoder.extract_features(
                t.cuda(), modality=modality, n_keep_tokens=cfg.n_keep_tokens
            )
        out_chunks.append(feats.detach().cpu().to(torch.float16))
    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# Multi-GPU sharding wrapper
# ---------------------------------------------------------------------------
def _shard_episodes_by_rank(
    staged_chunk_dir: Path, world_size: int, rank: int
) -> Path:
    """Return a transient view of ``staged_chunk_dir`` whose ``ep_*`` symlinks
    are filtered to the episodes owned by ``rank``.

    We build a per-rank scratch directory under ``staged_chunk_dir.parent /
    f"_shard_view_{rank}_of_{world_size}"`` and symlink only the episode dirs
    where ``ep_idx % world_size == rank``. This lets us reuse the existing
    single-process ``encode_chunk`` unmodified — it just sees fewer episodes.
    """
    import json as _json
    import shutil

    view_root = staged_chunk_dir.parent / f"_shard_view_{rank}_of_{world_size}"
    view_root.mkdir(parents=True, exist_ok=True)
    # Clear any stale entries from a previous run.
    for old in view_root.glob("ep_*"):
        if old.is_symlink():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    for ep_dir in sorted(staged_chunk_dir.glob("ep_*")):
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        ep_idx = int(_json.loads(meta_path.read_text())["episode_index"])
        if ep_idx % world_size != rank:
            continue
        (view_root / ep_dir.name).symlink_to(ep_dir.resolve(), target_is_directory=True)
    return view_root


def _encode_chunk_worker(
    rank: int,
    world_size: int,
    staged_chunk_dir: str,
    output_root: str,
    modalities: tuple[str, ...],
    cameras: tuple[str, ...],
    dinov3_ckpt: Optional[str],
    source_fps: int,
    config_kwargs: dict,
) -> None:
    """torch.multiprocessing.spawn entry point.

    Pinned to one GPU; loads its own DINOv3; processes only the episodes
    owned by ``rank``; writes ``file-{rank:03d}.safetensors`` per (cam, mod).
    """
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format=f"[rank {rank}] %(asctime)s %(message)s")
    log = _logging.getLogger(__name__)

    # Pin to our GPU. spawn launches us with all GPUs visible, so we have
    # to set_device explicitly. CUDA_VISIBLE_DEVICES is not honored after
    # the parent process has already initialized CUDA in some PyTorch builds.
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.set_device(rank)

    view_dir = _shard_episodes_by_rank(
        Path(staged_chunk_dir), world_size=world_size, rank=rank
    )

    # If our shard has no episodes, return immediately. encode_chunk would
    # also be a no-op but logging here makes the rank diagnostics clearer.
    if not list(view_dir.glob("ep_*")):
        log.info("no episodes assigned; exiting")
        return

    cfg = DinoCacheConfig(**config_kwargs)
    written = encode_chunk(
        staged_chunk_dir=view_dir,
        output_root=Path(output_root),
        modalities=modalities,
        cameras=cameras,
        dinov3_ckpt=Path(dinov3_ckpt) if dinov3_ckpt else None,
        source_fps=source_fps,
        config=cfg,
        shard_id=rank,
    )
    log.info("wrote %d shards", len(written))


def encode_chunk_multigpu(
    staged_chunk_dir: Path,
    output_root: Path,
    modalities: Iterable[str] = ("rgb", "depth", "flow"),
    cameras: Iterable[str] = ("head_rgb", "wrist_rgb"),
    dinov3_ckpt: Optional[Path] = None,
    source_fps: int = 30,
    world_size: int = 0,
    config: DinoCacheConfig | None = None,
) -> None:
    """Run :func:`encode_chunk` sharded across ``world_size`` GPUs via
    ``torch.multiprocessing.spawn``.

    * ``world_size=0`` (default) auto-detects ``torch.cuda.device_count()``
      and falls back to 1 if no CUDAs are visible.
    * Each rank handles episodes ``ep_idx % world_size == rank``.
    * Each rank writes ``file-{rank:03d}.safetensors`` per (cam, mod).

    We do NOT return the list of shards because spawn's children write to
    disk autonomously; callers should glob ``output_root/.../chunk-*/file-*.safetensors``.
    """
    import torch as _torch
    if world_size <= 0:
        world_size = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
    cfg = config or DinoCacheConfig()
    config_kwargs = dict(
        target_hw=cfg.target_hw,
        n_keep_tokens=cfg.n_keep_tokens,
        embed_dim=cfg.embed_dim,
        batch_size=cfg.batch_size,
        cache_fps=cfg.cache_fps,
        fp16=cfg.fp16,
    )

    if world_size == 1:
        # Single-process path: avoid mp.spawn so the placeholder smoke
        # tests (no CUDA) and CI runners stay simple.
        _encode_chunk_worker(
            rank=0,
            world_size=1,
            staged_chunk_dir=str(staged_chunk_dir),
            output_root=str(output_root),
            modalities=tuple(modalities),
            cameras=tuple(cameras),
            dinov3_ckpt=str(dinov3_ckpt) if dinov3_ckpt else None,
            source_fps=source_fps,
            config_kwargs=config_kwargs,
        )
        return

    import torch.multiprocessing as mp
    mp.spawn(
        _encode_chunk_worker,
        args=(
            world_size,
            str(staged_chunk_dir),
            str(output_root),
            tuple(modalities),
            tuple(cameras),
            str(dinov3_ckpt) if dinov3_ckpt else None,
            source_fps,
            config_kwargs,
        ),
        nprocs=world_size,
        join=True,
    )


__all__ = ["DinoCacheConfig", "encode_chunk", "encode_chunk_multigpu"]
