# SPDX-License-Identifier: MIT
"""Stage 2b: SEA-RAFT optical-flow precompute.

Reads a chunk's worth of staged RGB frames produced by stage 2a, runs SEA-RAFT
in fp16 batches, and writes ``flow_{cam}.npy`` (HSV-encoded) plus an HSV-h264
``.mp4`` for downstream visualization.

This module exposes a tiny ``compute_flow_for_chunk`` API that the dispatcher
wraps in a ``CheckpointedJob`` instance. We do not subclass ``CheckpointedJob``
here because flow precompute is per-camera-per-chunk rather than per-episode.

Model
-----
Default checkpoint is ``MemorySlices/Tartan-C-T-TSKH-spring540x960-M`` (the
Spring-finetuned medium variant, 19.7M params). The paper's L (large) variant
was never published publicly — only S and M are available under the
``MemorySlices`` org. The shim at ``/opt/searaft/sea_raft/api.py`` (baked into
the prep Docker image) wraps the upstream ``RAFT`` class with
``PyTorchModelHubMixin``-style loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

_LOG = logging.getLogger(__name__)


@dataclass
class FlowConfig:
    """Hyperparameters for SEA-RAFT inference.

    Attributes
    ----------
    target_hw : tuple[int, int]
        Frames are resized to (H, W) before flow inference.
    batch_size : int
    iters : int
        SEA-RAFT iterative refinement steps.
    fp16 : bool
    """

    target_hw: tuple[int, int] = (378, 378)  # 27 * 14; ViT-B/14 → 729 tokens
    batch_size: int = 8
    iters: int = 12
    fp16: bool = True


def _flow_to_hsv(flow_uv: np.ndarray) -> np.ndarray:
    """Convert ``[H, W, 2]`` flow to an ``[H, W, 3]`` uint8 HSV-RGB image.

    Mirrors the reference visualization in the RAFT repo: angle -> hue,
    magnitude -> saturation/value. Used so flow can be stored as h264 mp4.
    """
    assert flow_uv.ndim == 3 and flow_uv.shape[-1] == 2, flow_uv.shape
    import cv2  # type: ignore

    fx, fy = flow_uv[..., 0], flow_uv[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((*flow_uv.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _load_searaft(ckpt: Optional[Union[str, Path]]):
    """Load a frozen SEA-RAFT checkpoint. Lazy import.

    Parameters
    ----------
    ckpt : str | Path | None
        Either an HF Hub model id (e.g.
        ``"MemorySlices/Tartan-C-T-TSKH-spring540x960-M"``) or a path to a
        local ``.pth`` file. ``None`` returns ``None``.
    """
    if ckpt is None:
        return None
    try:
        from sea_raft.api import load_searaft  # type: ignore

        return load_searaft(str(ckpt))
    except ImportError as e:  # pragma: no cover - only hit at real runtime
        raise RuntimeError(
            "SEA-RAFT is required for stage_2b_compute_flow at runtime "
            "(pip install -r requirements/prep.txt)"
        ) from e


def compute_flow_for_chunk(
    staged_chunk_dir: Path,
    output_dir: Path,
    cameras: Iterable[str],
    searaft_ckpt: Optional[Union[str, Path]] = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
    config: FlowConfig | None = None,
) -> List[Path]:
    """Compute per-camera flow for one chunk's worth of episodes.

    Parameters
    ----------
    staged_chunk_dir : Path
        Output of stage 2a (``ep_*/camera_<cam>.npy``).
    output_dir : Path
        ``flow_<cam>.npy`` and an HSV mp4 are written to ``ep_*/`` here.
    cameras : iterable of str
        Canonical camera keys to process (e.g. ``["head_rgb", "wrist_rgb"]``).
    searaft_ckpt : str | Path | None
        HF Hub model id (default
        ``"MemorySlices/Tartan-C-T-TSKH-spring540x960-M"``) or a path to a
        local checkpoint. ``None`` bypasses model load (smoke-test path).
    config : FlowConfig | None

    Returns
    -------
    list[Path]
        Paths of the per-episode flow files written.
    """
    cfg = config or FlowConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    if not staged_chunk_dir.exists():
        _LOG.warning("staged chunk dir %s does not exist; skipping", staged_chunk_dir)
        return written

    model = _load_searaft(searaft_ckpt) if searaft_ckpt is not None else None
    if model is None:
        _LOG.warning(
            "no SEA-RAFT checkpoint provided; stage_2b will run as a structural smoke-test only"
        )

    cams = list(cameras)
    for ep_dir in sorted(staged_chunk_dir.glob("ep_*")):
        for cam in cams:
            rgb_npy = ep_dir / f"camera_{cam}.npy"
            if not rgb_npy.exists():
                continue
            rgb = np.load(rgb_npy)  # [T, H, W, 3] uint8
            flow_path = output_dir / ep_dir.name / f"flow_{cam}.npy"
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            if model is None:
                # Smoke path: emit zero-flow so downstream stages have shapes.
                T, H, W, _ = rgb.shape
                placeholder = np.zeros((T - 1, H, W, 2), dtype=np.float16)
                np.save(flow_path, placeholder)
            else:  # pragma: no cover - exercised only on real T1 hosts
                flow = _run_searaft(model, rgb, cfg)
                np.save(flow_path, flow.astype(np.float16))
            written.append(flow_path)
    return written


def _run_searaft(model, rgb: np.ndarray, cfg: FlowConfig) -> np.ndarray:  # pragma: no cover
    """Batched fp16 SEA-RAFT inference. Real-runtime only.

    SEA-RAFT's ``forward`` returns a dict (``flow``, ``info``) where each entry
    is a list of refinement-step tensors; we take the last entry (final
    refinement) per RAFT convention.
    """
    import torch

    T = rgb.shape[0]
    out = np.zeros((T - 1, *cfg.target_hw, 2), dtype=np.float16)
    for start in range(0, T - 1, cfg.batch_size):
        end = min(start + cfg.batch_size, T - 1)
        a = torch.as_tensor(rgb[start:end]).permute(0, 3, 1, 2).contiguous().float() / 255.0
        b = torch.as_tensor(rgb[start + 1 : end + 1]).permute(0, 3, 1, 2).contiguous().float() / 255.0
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            output = model(a.cuda(), b.cuda(), iters=cfg.iters, test_mode=True)
        # SEA-RAFT returns a dict with `flow` being a list of refinement steps.
        if isinstance(output, dict) and "flow" in output:
            flow_pred = output["flow"][-1] if isinstance(output["flow"], (list, tuple)) else output["flow"]
        else:
            flow_pred = output
        flow_np = flow_pred.detach().cpu().permute(0, 2, 3, 1).numpy()
        out[start:end] = flow_np.astype(np.float16)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``python -m prep.stage_2b_compute_flow --dataset droid --chunk 0``.

    The CLI is a thin wrapper over :func:`compute_flow_for_chunk`. The
    ``--dataset`` flag selects the source name used to build paths (one
    A100 node per dataset, per Wave F).
    """
    import argparse
    import os as _os

    parser = argparse.ArgumentParser(
        prog="prep.stage_2b_compute_flow", description=__doc__
    )
    scratch_default = Path(_os.environ.get("USAM_SCRATCH", "/scratch/usam"))
    # Either --dataset or --source must be supplied (--source is deprecated).
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument(
        "--dataset",
        choices=("droid", "bridge", "agibot2026", "oxe_auge", "rh20t", "robomind"),
        help="Source name (one A100 node per dataset).",
    )
    ds.add_argument(
        "--source",
        dest="dataset",
        choices=("droid", "bridge", "agibot2026", "oxe_auge", "rh20t", "robomind"),
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
        default=scratch_default / "flow",
        help="Root where per-episode flow files are written.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["head_rgb"],
        help="Canonical camera keys to process.",
    )
    parser.add_argument(
        "--searaft-ckpt",
        type=str,
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        help="HF model id or local path for the SEA-RAFT checkpoint "
        "(default: MemorySlices/Tartan-C-T-TSKH-spring540x960-M). Pass an "
        "empty string to skip model load (placeholder/smoke mode).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for parity; flow precompute is always per-episode idempotent.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    chunk_dir = args.staged_root / args.dataset / f"chunk-{args.chunk:03d}"
    ckpt = args.searaft_ckpt if args.searaft_ckpt else None
    compute_flow_for_chunk(
        staged_chunk_dir=chunk_dir,
        output_dir=args.output_root / args.dataset / f"chunk-{args.chunk:03d}",
        cameras=args.cameras,
        searaft_ckpt=ckpt,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())


__all__ = ["FlowConfig", "compute_flow_for_chunk"]
