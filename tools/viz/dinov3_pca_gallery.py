"""DINOv3 + depth visualization gallery for a USAM prep chunk.

For each (episode, camera) in /workspace/output/staged, samples N frames
evenly and saves a 3-column side-by-side image:

    [   RGB   |  DINOv3 PCA  |   Depth viz   ]

  * RGB:   the original frame, resized to 448×448 (the DINOv3 input
           resolution).
  * DINO:  PCA of the 1024-D patch tokens to 3 channels → 28×28 grid →
           upscaled to 448×448 with nearest-neighbour. Coherent color
           regions on object/gripper/table boundaries indicate the
           encoder is producing semantically meaningful patch features.
  * Depth: per-frame min-max normalized + viridis colormap, upscaled to
           448×448. Black = far, yellow = near (DA3MONO-LARGE outputs
           metric mm depth, so smaller = nearer).

Output lands under /workspace/output/viz/dinov3_chunk0/ with an
index.html that groups frames by (episode, camera).
"""
from __future__ import annotations

import logging
import sys
import warnings
from html import escape
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
warnings.filterwarnings("ignore")

STAGED = Path("/workspace/output/staged")
DEPTH_OUT = Path("/workspace/output/depth")
VIZ_OUT = Path("/workspace/output/viz/dinov3_chunk0")
FRAMES_PER_EP_CAM = 10  # 10 frames per (episode, camera)
TARGET_HW = (448, 448)


# ---------------------------------------------------------------------------
# Per-modality visualizers
# ---------------------------------------------------------------------------
def pca_3channel(patch_tokens: torch.Tensor) -> np.ndarray:
    """[num_patches, D] -> [H_grid, W_grid, 3] uint8 RGB heatmap.

    Three top PCA components mapped to R/G/B; each channel normalized to
    [0, 255] independently for visual contrast.
    """
    feats = patch_tokens.float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    _u, _s, v = torch.svd_lowrank(feats, q=3)
    pca = feats @ v
    pca = pca - pca.amin(dim=0, keepdim=True)
    pca = pca / pca.amax(dim=0, keepdim=True).clamp(min=1e-6)
    grid = int(pca.shape[0] ** 0.5)
    assert grid * grid == pca.shape[0], f"non-square grid: {pca.shape[0]}"
    pca = pca.reshape(grid, grid, 3)
    return (pca * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def depth_viz(depth_uint16: np.ndarray) -> np.ndarray:
    """[H, W] uint16 (mm) -> [H, W, 3] uint8 RGB (viridis colormap).

    Per-frame 2%–98% percentile clip + min-max normalize → OpenCV's
    COLORMAP_VIRIDIS so near objects pop in bright yellow, far in dark
    purple. Zero-valued (no-return) pixels get clipped to the dark end.
    """
    d = depth_uint16.astype(np.float32)
    if d.max() > 0:
        nonzero = d[d > 0]
        if nonzero.size > 0:
            lo, hi = np.percentile(nonzero, (2, 98))
            d = np.clip(d, lo, hi)
            d = (d - lo) / max(hi - lo, 1e-3)
    d_uint8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(d_uint8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    VIZ_OUT.mkdir(parents=True, exist_ok=True)

    print("loading DINOv3 ViT-L/16 (offline)…", flush=True)
    from usam.encoders.tri_dino import TriDinoConfig, TriDINOTower

    cfg = TriDinoConfig(
        dinov3_ckpt="facebook/dinov3-vitl16-pretrain-lvd1689m",
        dinov3_arch="vit_l_16",
        image_size=448,
        patch_size=16,
        embed_dim=1024,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tower = TriDINOTower(cfg).to(device).eval()
    print(f"loaded; num_patches={tower.num_patches} num_register_tokens={tower.num_register_tokens}", flush=True)

    gallery_entries: list[tuple[str, str]] = []  # (label, relpath)

    for ep_dir in sorted(STAGED.glob("ep_*")):
        ep_id = ep_dir.name
        for cam in ("head_rgb", "wrist_rgb"):
            rgb_path = ep_dir / f"camera_{cam}.npy"
            if not rgb_path.exists():
                continue
            depth_path = DEPTH_OUT / ep_id / f"depth_{cam}.npy"

            arr_rgb = np.load(rgb_path)            # [T, H, W, 3] uint8
            arr_depth = np.load(depth_path) if depth_path.exists() else None

            T = arr_rgb.shape[0]
            idxs = np.linspace(0, T - 1, FRAMES_PER_EP_CAM, dtype=int)
            out_dir = VIZ_OUT / ep_id / cam
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  {ep_id}/{cam}: T={T} -> {len(idxs)} frames", flush=True)

            for i, t in enumerate(idxs):
                # ----- RGB
                rgb = arr_rgb[t]
                rgb_448 = cv2.resize(rgb, TARGET_HW[::-1], interpolation=cv2.INTER_AREA)

                # ----- DINOv3 PCA
                x = torch.from_numpy(rgb_448).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                x = x.to(device)
                with torch.no_grad():
                    out = tower(x, modality="rgb")
                patch_tokens = out[0, 1 + tower.num_register_tokens:, :]
                pca_grid = pca_3channel(patch_tokens)
                pca_448 = cv2.resize(pca_grid, TARGET_HW[::-1], interpolation=cv2.INTER_NEAREST)

                # ----- Depth
                if arr_depth is not None:
                    d_viz = depth_viz(arr_depth[t])
                else:
                    d_viz = np.zeros((*TARGET_HW, 3), dtype=np.uint8)
                depth_448 = cv2.resize(d_viz, TARGET_HW[::-1], interpolation=cv2.INTER_AREA)

                # ----- 3-column composite (448 x 1344 x 3)
                sxs = np.concatenate([rgb_448, pca_448, depth_448], axis=1)

                sxs_p = out_dir / f"frame_{i:02d}_t{t:04d}_3col.png"
                cv2.imwrite(str(sxs_p), cv2.cvtColor(sxs, cv2.COLOR_RGB2BGR))
                rel = sxs_p.relative_to(VIZ_OUT).as_posix()
                gallery_entries.append((f"{ep_id} / {cam} / t={t}", rel))
                print(f"    [{i}/{FRAMES_PER_EP_CAM}] t={t} -> {rel}", flush=True)

    # ---- HTML gallery ----
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>USAM prep chunk-0 visualization</title>",
            "<style>",
            "body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:20px;}",
            "h1{color:#aef;} h2{color:#fea;margin-top:30px;border-bottom:1px solid #555;padding-bottom:4px;}",
            ".panels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;",
            "        max-width:1380px;margin:6px 0 16px 0;color:#aaa;font-size:0.85em;}",
            "img{display:block;margin:6px 0 4px 0;max-width:100%;width:100%;}",
            ".row{margin:18px 0;border:1px solid #333;padding:8px;border-radius:4px;background:#222;}",
            ".label{color:#bbb;font-size:0.9em;margin-bottom:4px;}",
            "</style></head><body>",
            "<h1>USAM prep chunk-0 — RGB · DINOv3 PCA · Depth</h1>",
            "<p>Each row shows one frame's three panels in order:",
            " <b>RGB</b> (original 448×448 input) ·",
            " <b>DINOv3 PCA</b> (1024-D patch tokens projected to 3 channels, 28×28 grid) ·",
            " <b>Depth</b> (DA3MONO-LARGE, viridis colormap; near=yellow, far=purple).</p>"]
    last_section = None
    for label, rel in gallery_entries:
        section = label.rsplit(" / ", 1)[0]
        if section != last_section:
            html.append(f"<h2>{escape(section)}</h2>")
            last_section = section
        html.append(
            f'<div class="row"><div class="label">{escape(label)}</div>'
            f'<img src="{escape(rel)}" alt="{escape(label)}">'
            f'<div class="panels"><span>RGB</span><span>DINOv3 PCA</span><span>Depth (viridis)</span></div>'
            f'</div>'
        )
    html.append("</body></html>")
    (VIZ_OUT / "index.html").write_text("\n".join(html))
    print(f"\nwrote {len(gallery_entries)} 3-column frames + index.html under {VIZ_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
