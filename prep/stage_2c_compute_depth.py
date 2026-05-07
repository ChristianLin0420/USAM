# SPDX-License-Identifier: MIT
"""Stage 2c: depth precompute (Phase 1 — DROID-only DAv2 path).

DROID has stereo ZED frames in ``droid_raw``; Phase 1 uses the **mono DAv2
fallback** path so the dataloader/feature-cache stack can be exercised
end-to-end without depending on ZED runtime libs. The stereo path will be
swapped in during Phase 2.

Outputs ``depth_<cam>.npy`` (uint16, mm) per episode and an HEVC ``.mp4`` for
visualization. The mp4 encode step lives in pipeline-engineer's
``prep/_video.py``; we only emit the npy + a small JSON sidecar that flags
``low_quality=True`` so downstream consumers know the depth is monocular.
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
    """Hyperparameters for DAv2 inference."""

    target_hw: tuple[int, int] = (192, 192)
    batch_size: int = 16
    fp16: bool = True
    # max range used to scale relative depth to uint16 mm
    max_range_mm: int = 5000


def _load_dav2(ckpt_path: Path):  # pragma: no cover - lazy
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        model = DepthAnythingV2(encoder="vitl")
        import torch

        sd = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model.eval().cuda()
        return model
    except ImportError as e:
        raise RuntimeError(
            "depth-anything-v2 is required for stage_2c_compute_depth at runtime"
        ) from e


def compute_depth_for_chunk(
    staged_chunk_dir: Path,
    output_dir: Path,
    cameras: Iterable[str],
    dav2_ckpt: Optional[Path] = None,
    config: DepthConfig | None = None,
) -> List[Path]:
    """Run DAv2 monocular depth on a chunk of staged RGB frames.

    Parameters mirror :func:`prep.stage_2b_compute_flow.compute_flow_for_chunk`.

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

    model = _load_dav2(dav2_ckpt) if dav2_ckpt is not None else None
    if model is None:
        _LOG.warning(
            "no DAv2 checkpoint provided; stage_2c will run as a structural smoke-test only"
        )

    cams = list(cameras)
    for ep_dir in sorted(staged_chunk_dir.glob("ep_*")):
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
            else:  # pragma: no cover
                depth = _run_dav2(model, rgb, cfg)
                np.save(depth_path, depth.astype(np.uint16))
            (depth_path.parent / f"depth_{cam}.json").write_text(
                json.dumps({"low_quality": True, "source": "dav2_mono"})
            )
            written.append(depth_path)
    return written


def _run_dav2(model, rgb: np.ndarray, cfg: DepthConfig) -> np.ndarray:  # pragma: no cover
    """Batched fp16 DAv2 inference."""
    import torch

    T, H, W, _ = rgb.shape
    out = np.zeros((T, *cfg.target_hw), dtype=np.uint16)
    for start in range(0, T, cfg.batch_size):
        end = min(start + cfg.batch_size, T)
        x = torch.as_tensor(rgb[start:end]).permute(0, 3, 1, 2).contiguous().float() / 255.0
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            rel_depth = model(x.cuda())
        rel = rel_depth.detach().cpu().numpy()
        rel_norm = (rel - rel.min()) / max(1e-6, (rel.max() - rel.min()))
        depth_mm = (rel_norm * cfg.max_range_mm).clip(0, np.iinfo(np.uint16).max)
        out[start:end] = depth_mm.astype(np.uint16)
    return out


__all__ = ["DepthConfig", "compute_depth_for_chunk"]
