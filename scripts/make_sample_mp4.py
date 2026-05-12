# SPDX-License-Identifier: MIT
"""Generate sample_<dataset>.mp4 from one episode: RGB | depth | DINO-RGB PCA | DINO-depth PCA.

Layout discovery (per dataset name `D`):
  staged:     data/usam/<D>/output/staged/<src>/chunk-000/ep_<hash>/
              {camera_head_rgb.npy, depth_head_rgb.npy, meta.json}
  dino cache: data/usam/<D>/output/dino_cache/<src>/<cam>/{rgb,depth}/chunk-000/file-*.safetensors
  lerobot:    data/usam/<D>/output/lerobot/meta/info.json  (for fps + source name)

Where `<src>` is the canonical source key (info.json::source). For agibot the
disk dataset name is `agibot` but the source is `agibot2026`. Robomind's
episode_index is the hash-mod-(2^31-1) so the dino key may be longer than 8
digits — :08d handles both.

Usage:
  python scripts/make_sample_mp4.py            # all 4 datasets
  python scripts/make_sample_mp4.py droid bridge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from safetensors import safe_open


DATA_ROOT = Path("/localhome/local-chrislin/USAM/data/usam")
OUT_ROOT = Path("/localhome/local-chrislin/USAM")
ALL_DATASETS = ("droid", "agibot", "bridge", "robomind")

PANEL = 256
LABEL_H = 28


def colorize_depth(d: np.ndarray) -> np.ndarray:
    """uint16 depth -> BGR uint8 (TURBO colormap, ignoring zeros)."""
    valid = d > 0
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def dino_pca_video(features: torch.Tensor) -> np.ndarray:
    """(Tf, N, D) fp16 -> (Tf, g, g, 3) uint8 BGR via global PCA on patch tokens."""
    Tf, N, D = features.shape
    n_patch = N - 1
    g = int(round(n_patch ** 0.5))
    assert g * g == n_patch, f"expected square patch grid, got {n_patch}"

    x = features[:, 1:, :].float()
    flat = x.reshape(-1, D)
    flat = flat - flat.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    proj = (flat @ vh[:3].T).reshape(Tf, g, g, 3)
    lo = proj.amin(dim=(0, 1, 2), keepdim=True)
    hi = proj.amax(dim=(0, 1, 2), keepdim=True)
    norm = ((proj - lo) / (hi - lo + 1e-6)).clamp(0, 1).numpy()
    return (norm * 255.0).astype(np.uint8)[..., ::-1].copy()  # RGB -> BGR


def fit_panel(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Letterbox-resize into a (size, size) panel."""
    h, w = img_bgr.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def label_strip(panels: list[np.ndarray], labels: list[str], panel: int) -> np.ndarray:
    band = np.zeros((LABEL_H, panel * len(panels), 3), dtype=np.uint8)
    for i, text in enumerate(labels):
        cv2.putText(
            band, text, (i * panel + 8, LABEL_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return np.vstack([band, np.hstack(panels)])


def find_dino_shard_for_episode(dino_root: Path, ep_idx: int) -> Optional[Path]:
    """Return the shard containing ``ep_{ep_idx:08d}`` (None if not found)."""
    ep_key = f"ep_{ep_idx:08d}"
    for shard in sorted(dino_root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            if ep_key in f.keys():
                return shard
    return None


def resolve_sample_episode(dataset: str) -> Tuple[Path, int, str, int, Path, Path]:
    """Pick the first episode that has RGB, depth, and a dino-cache entry.

    Returns (staged_ep_dir, ep_idx, source, native_fps, dino_rgb_shard, dino_depth_shard).
    """
    base = DATA_ROOT / dataset
    info = json.loads((base / "output/lerobot/meta/info.json").read_text())
    source = info["source"]
    fps = int(info["fps"])
    fps_features = int(info["fps_features"])

    staged_chunk = base / f"output/staged/{source}/chunk-000"
    dino_rgb_dir = base / f"output/dino_cache/{source}/head_rgb/rgb/chunk-000"
    dino_depth_dir = base / f"output/dino_cache/{source}/head_rgb/depth/chunk-000"

    ep_dirs = sorted(
        p for p in staged_chunk.glob("ep_*")
        if p.is_dir() and not p.name.endswith(".done")
    )
    for ep in ep_dirs:
        cam_path = ep / "camera_head_rgb.npy"
        depth_path = ep / "depth_head_rgb.npy"
        meta_path = ep / "meta.json"
        if not (cam_path.exists() and depth_path.exists() and meta_path.exists()):
            continue
        ep_idx = int(json.loads(meta_path.read_text())["episode_index"])
        rgb_shard = find_dino_shard_for_episode(dino_rgb_dir, ep_idx)
        depth_shard = find_dino_shard_for_episode(dino_depth_dir, ep_idx)
        if rgb_shard is not None and depth_shard is not None:
            return ep, ep_idx, source, fps, fps_features, rgb_shard, depth_shard
    raise FileNotFoundError(
        f"no episode under {staged_chunk} has both RGB/depth on disk and a dino-cache entry"
    )


def render_one(dataset: str, max_frames: Optional[int] = None) -> Path:
    print(f"\n=== {dataset} ===")
    ep_dir, ep_idx, source, fps, fps_features, rgb_shard, depth_shard = resolve_sample_episode(dataset)
    feature_stride = max(1, fps // fps_features)

    print(f"[load] staged ep:  {ep_dir.name} (episode_index={ep_idx}, source={source})")
    rgb = np.load(ep_dir / "camera_head_rgb.npy", mmap_mode="r")
    depth = np.load(ep_dir / "depth_head_rgb.npy", mmap_mode="r")
    T = rgb.shape[0]
    assert depth.shape[0] == T, f"frame mismatch {rgb.shape[0]} vs {depth.shape[0]}"
    if max_frames is not None and T > max_frames:
        print(f"  truncating from T={T} to {max_frames}")
        T = max_frames
    print(f"  T={T}  rgb={rgb.shape}  depth={depth.shape}  fps={fps}/{fps_features}")

    ep_key = f"ep_{ep_idx:08d}"
    print(f"[load] dino:       key={ep_key}  rgb={rgb_shard.name}  depth={depth_shard.name}")
    with safe_open(str(rgb_shard), framework="pt") as f:
        feat_rgb = f.get_tensor(ep_key)
    with safe_open(str(depth_shard), framework="pt") as f:
        feat_depth = f.get_tensor(ep_key)
    print(f"  feat_rgb={tuple(feat_rgb.shape)}  feat_depth={tuple(feat_depth.shape)}")

    print("[pca] computing per-episode DINO PCA")
    pca_rgb = dino_pca_video(feat_rgb)
    pca_depth = dino_pca_video(feat_depth)

    print("[depth] colorize")
    depth_bgr = np.stack([colorize_depth(np.asarray(depth[t])) for t in range(T)], axis=0)

    out_path = OUT_ROOT / f"sample_{dataset}.mp4"
    H_out, W_out = LABEL_H + PANEL, PANEL * 4
    print(f"[write] {out_path}  res={W_out}x{H_out}  fps={fps}  frames={T}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W_out, H_out))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed for {out_path}")

    Tf = pca_rgb.shape[0]
    for t in range(T):
        rgb_bgr = cv2.cvtColor(np.asarray(rgb[t]), cv2.COLOR_RGB2BGR)
        tf = min(t // feature_stride, Tf - 1)
        frame = label_strip(
            [
                fit_panel(rgb_bgr, PANEL),
                fit_panel(depth_bgr[t], PANEL),
                fit_panel(pca_rgb[tf], PANEL),
                fit_panel(pca_depth[tf], PANEL),
            ],
            [f"{dataset} RGB t={t}", f"DEPTH t={t}", f"DINO-RGB tf={tf}", f"DINO-DEPTH tf={tf}"],
            PANEL,
        )
        writer.write(frame)
    writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"[done] {out_path} ({size_mb:.2f} MB, {T} frames @ {fps} fps)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(ALL_DATASETS),
                    help=f"Subset of {ALL_DATASETS}; default is all four.")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Cap per-episode frame count (default 600 — keeps agibot < ~30s).")
    args = ap.parse_args()

    unknown = [d for d in args.datasets if d not in ALL_DATASETS]
    if unknown:
        sys.exit(f"unknown datasets: {unknown}; valid = {list(ALL_DATASETS)}")

    for d in args.datasets:
        try:
            render_one(d, max_frames=args.max_frames)
        except Exception as e:
            print(f"[ERROR] {d}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
