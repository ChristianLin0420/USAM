# SPDX-License-Identifier: MIT
"""Stage 2c: depth precompute (Depth-Anything-V3 mono fallback path).

DROID has stereo ZED frames in ``droid_raw``; we use the **mono DA3 fallback**
path so the dataloader/feature-cache stack can run end-to-end without
depending on ZED runtime libs. The stereo path will be swapped in later.

Outputs ``depth_<cam>.npy`` (uint16, mm) per episode and an HEVC ``.mp4`` for
visualization. The mp4 encode step lives in pipeline-engineer's
``prep/_video.py``; we only emit the npy + a small JSON sidecar that flags
``low_quality=True`` so downstream consumers know the depth is monocular.

Model
-----
Default checkpoint is ``depth-anything/DA3MONO-LARGE`` from HuggingFace Hub
(the Depth-Anything-V3 mono-depth large preset). The DA3 ``DepthAnything3``
class loads via ``PyTorchModelHubMixin.from_pretrained`` and exposes a
high-level ``.inference(images, ...)`` method that returns a ``Prediction``
object whose ``.depth`` attribute is a ``[N, H, W]`` numpy array of metric
depth in meters.

We convert that to ``uint16 mm`` (clipped to ``max_range_mm``) for storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

_LOG = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Hyperparameters for DA3 inference."""

    target_hw: tuple[int, int] = (192, 192)
    batch_size: int = 64  # bumped from 16; DA3 at bs=64 ≈ 5 GB VRAM
    fp16: bool = True
    # max range used to clip metric depth (meters -> mm).
    max_range_mm: int = 5000
    # DA3 processing resolution; the upstream default is 504.
    process_res: int = 504


def _load_dav3(ckpt: Optional[str] = "depth-anything/DA3MONO-LARGE"):  # pragma: no cover - lazy
    """Lazy-load a frozen Depth-Anything-V3 checkpoint.

    Parameters
    ----------
    ckpt : str | None
        HuggingFace Hub model id (e.g. ``"depth-anything/DA3MONO-LARGE"``) or
        a local directory containing ``config.json`` + ``model.safetensors``.
        ``None`` returns ``None`` (smoke-test placeholder path).
    """
    if ckpt is None:
        return None
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "depth-anything-3 is required for stage_2c_compute_depth at runtime "
            "(install via `pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git` "
            "or clone to /opt/da3 and add it to PYTHONPATH)"
        ) from e
    import torch

    model = DepthAnything3.from_pretrained(str(ckpt))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def compute_depth_for_chunk(
    staged_chunk_dir: Path,
    output_dir: Path,
    cameras: Iterable[str],
    dav3_ckpt: Optional[str] = "depth-anything/DA3MONO-LARGE",
    config: DepthConfig | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> List[Path]:
    """Run DA3 monocular depth on a chunk of staged RGB frames.

    Parameters
    ----------
    dav3_ckpt : str | None
        HF Hub model id or local path. Defaults to ``depth-anything/DA3MONO-LARGE``.
        Pass ``None`` to bypass model loading (unit-test smoke path).
    rank, world_size : int
        For multi-worker parallelism: this process handles episodes whose
        index in the sorted ``ep_*`` list satisfies ``i % world_size == rank``.
        Defaults (0, 1) keep single-process behaviour bit-identical.

    Returns
    -------
    list[Path]
        Paths of the per-episode depth files written.
    """
    cfg = config or DepthConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    if not staged_chunk_dir.exists():
        _LOG.warning("staged chunk dir %s does not exist; skipping", staged_chunk_dir)
        return written

    model = _load_dav3(dav3_ckpt) if dav3_ckpt is not None else None
    if model is None:
        _LOG.warning(
            "no DA3 checkpoint provided; stage_2c will run as a structural smoke-test only"
        )

    cams = list(cameras)
    all_eps = sorted(staged_chunk_dir.glob("ep_*"))
    my_eps = [ep for i, ep in enumerate(all_eps) if i % world_size == rank]
    for ep_dir in my_eps:
        for cam in cams:
            rgb_npy = ep_dir / f"camera_{cam}.npy"
            if not rgb_npy.exists():
                continue
            rgb = np.load(rgb_npy)
            T, _, _, _ = rgb.shape
            depth_path = output_dir / ep_dir.name / f"depth_{cam}.npy"
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if model is None:
                placeholder = np.zeros((T, *cfg.target_hw), dtype=np.uint16)
                np.save(depth_path, placeholder)
                source_tag = "placeholder"
            else:  # pragma: no cover
                depth = _run_dav3(model, rgb, cfg)
                np.save(depth_path, depth.astype(np.uint16))
                source_tag = "dav3_mono"
            (depth_path.parent / f"depth_{cam}.json").write_text(
                json.dumps({"low_quality": True, "source": source_tag})
            )
            written.append(depth_path)
    return written


def _depth_worker(
    rank: int,
    world_size: int,
    staged_chunk_dir: str,
    output_dir: str,
    cameras: tuple[str, ...],
    dav3_ckpt: Optional[str],
    config_kwargs: dict,
    num_physical_gpus: int,
) -> None:
    """torch.multiprocessing.spawn entry point. One DA3 instance per rank,
    pinned to ``rank % num_physical_gpus`` so multiple workers can share a GPU.
    DA3MONO-LARGE is ~4 GB at inference; 2-3 instances fit on a 46 GB A40.
    """
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format=f"[rank {rank}] %(asctime)s %(message)s")
    import torch as _torch
    if _torch.cuda.is_available():
        n_phys = num_physical_gpus if num_physical_gpus > 0 else _torch.cuda.device_count()
        _torch.cuda.set_device(rank % max(n_phys, 1))
    compute_depth_for_chunk(
        staged_chunk_dir=Path(staged_chunk_dir),
        output_dir=Path(output_dir),
        cameras=cameras,
        dav3_ckpt=dav3_ckpt,
        config=DepthConfig(**config_kwargs),
        rank=rank,
        world_size=world_size,
    )


def compute_depth_multigpu(
    staged_chunk_dir: Path,
    output_dir: Path,
    cameras: Iterable[str],
    dav3_ckpt: Optional[str] = "depth-anything/DA3MONO-LARGE",
    config: DepthConfig | None = None,
    world_size: int = 0,
    workers_per_gpu: int = 1,
) -> None:
    """Run :func:`compute_depth_for_chunk` sharded across ``world_size`` workers
    via ``torch.multiprocessing.spawn``.

    Same oversubscription idiom as :func:`prep.stage_4_dino_cache.encode_chunk_multigpu`:
    ``world_size = num_physical_gpus * workers_per_gpu`` by default. Each rank
    processes episodes ``i % world_size == rank`` and loads its own DA3 instance
    pinned to ``rank % num_physical_gpus``.
    """
    import torch as _torch
    num_physical_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    if world_size <= 0:
        world_size = max(num_physical_gpus * max(workers_per_gpu, 1), 1)
    cfg = config or DepthConfig()
    config_kwargs = dict(
        target_hw=cfg.target_hw,
        batch_size=cfg.batch_size,
        fp16=cfg.fp16,
        max_range_mm=cfg.max_range_mm,
        process_res=cfg.process_res,
    )

    if world_size == 1:
        # Single-process path: avoid mp.spawn so placeholder/smoke tests stay simple.
        _depth_worker(
            rank=0,
            world_size=1,
            staged_chunk_dir=str(staged_chunk_dir),
            output_dir=str(output_dir),
            cameras=tuple(cameras),
            dav3_ckpt=dav3_ckpt,
            config_kwargs=config_kwargs,
            num_physical_gpus=num_physical_gpus,
        )
        return

    import torch.multiprocessing as mp
    mp.spawn(
        _depth_worker,
        args=(
            world_size,
            str(staged_chunk_dir),
            str(output_dir),
            tuple(cameras),
            dav3_ckpt,
            config_kwargs,
            num_physical_gpus,
        ),
        nprocs=world_size,
        join=True,
    )


def _run_dav3(model, rgb: np.ndarray, cfg: DepthConfig) -> np.ndarray:  # pragma: no cover
    """Batched DA3 monocular inference.

    DA3 exposes a high-level ``inference()`` method that batches internally
    and returns a ``Prediction`` object. The depth is metric (meters); we
    clip and convert to uint16 mm.
    """
    import cv2  # type: ignore

    T, H, W, _ = rgb.shape
    out = np.zeros((T, *cfg.target_hw), dtype=np.uint16)
    target_h, target_w = cfg.target_hw
    for start in range(0, T, cfg.batch_size):
        end = min(start + cfg.batch_size, T)
        # DA3's inference() accepts a list of np.ndarray HxWx3 RGB uint8 images.
        imgs = [rgb[i] for i in range(start, end)]
        prediction = model.inference(
            imgs,
            process_res=cfg.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth_m = np.asarray(prediction.depth)  # [N, H', W'] float meters
        # Resize each frame to target_hw, clip to range, convert to uint16 mm.
        for k in range(depth_m.shape[0]):
            dm = depth_m[k]
            if dm.shape != (target_h, target_w):
                dm = cv2.resize(dm.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_mm = np.clip(dm * 1000.0, 0, cfg.max_range_mm).clip(0, np.iinfo(np.uint16).max)
            out[start + k] = depth_mm.astype(np.uint16)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``python -m prep.stage_2c_compute_depth --dataset droid --chunk 0``.

    The CLI is a thin wrapper over :func:`compute_depth_for_chunk`. The
    ``--dataset`` flag selects the source name used to build paths (one
    A100 node per dataset, per Wave F).
    """
    import argparse
    import os as _os

    parser = argparse.ArgumentParser(
        prog="prep.stage_2c_compute_depth", description=__doc__
    )
    scratch_default = Path(_os.environ.get("USAM_SCRATCH", "/scratch/usam"))
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument(
        "--dataset",
        choices=("droid", "bridge", "agibot2026", "robomind"),
        help="Source name (one A100 node per dataset).",
    )
    ds.add_argument(
        "--source",
        dest="dataset",
        choices=("droid", "bridge", "agibot2026", "robomind"),
        help="(deprecated) use --dataset",
    )
    parser.add_argument("--chunk", required=True, type=int)
    parser.add_argument(
        "--staged-root",
        type=Path,
        default=scratch_default / "staged",
        help="Root containing <dataset>/chunk-NNN/ep_*/ directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=scratch_default / "depth",
        help="Root where per-episode depth files are written.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["head_rgb"],
        help="Canonical camera keys to process.",
    )
    parser.add_argument(
        "--dav3-ckpt",
        type=str,
        default="depth-anything/DA3MONO-LARGE",
        help="HF model id or local path for the DA3 checkpoint "
        "(default: depth-anything/DA3MONO-LARGE). Pass an empty string to "
        "skip model load (placeholder/smoke mode).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for parity; depth precompute is always per-episode idempotent.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0,
        help="0 = auto-detect torch.cuda.device_count() * workers_per_gpu.",
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=1,
        help="Inference workers per physical GPU. >1 oversubscribes the GPU "
             "(DA3MONO-LARGE is ~4 GB at inference; A40 fits 2-3 workers).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Frames per DA3 inference call (default 64; bumped from 16).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    chunk_dir = args.staged_root / args.dataset / f"chunk-{args.chunk:03d}"
    ckpt = args.dav3_ckpt if args.dav3_ckpt else None
    cfg = DepthConfig(batch_size=int(args.batch_size))

    # Resolve world_size: explicit --num-gpus overrides; otherwise auto = phys * workers_per_gpu.
    if args.num_gpus > 0:
        world_size = args.num_gpus
    else:
        import torch as _torch
        _phys = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
        world_size = max(_phys * args.workers_per_gpu, 1)
    compute_depth_multigpu(
        staged_chunk_dir=chunk_dir,
        output_dir=args.output_root / args.dataset / f"chunk-{args.chunk:03d}",
        cameras=args.cameras,
        dav3_ckpt=ckpt,
        config=cfg,
        world_size=world_size,
        workers_per_gpu=args.workers_per_gpu,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())


__all__ = ["DepthConfig", "compute_depth_for_chunk", "compute_depth_multigpu"]
