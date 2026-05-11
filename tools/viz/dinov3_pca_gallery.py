"""Wave I: PCA visualization of DINOv3 patch features over real DROID frames.

For each of ~10 representative frames from each episode/camera:
  * Save the original RGB frame at 448×448 (the encoder input resolution).
  * Run DINOv3 ViT-L/16, extract patch tokens (drop CLS + register tokens),
    reshape to a 28×28 patch grid.
  * PCA the 1024-D patch features to 3 channels → normalize to [0, 255] → save
    as a side-by-side image alongside the RGB.

Output lands at /workspace/output/viz/dinov3_chunk0/:
  ep_<hash>/<cam>/frame_<NNN>_rgb.png
  ep_<hash>/<cam>/frame_<NNN>_pca.png
  ep_<hash>/<cam>/frame_<NNN>_sxs.png   (side-by-side)
  index.html                            (browsable gallery)
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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
warnings.filterwarnings("ignore")

STAGED = Path("/workspace/output/staged")
VIZ_OUT = Path("/workspace/output/viz/dinov3_chunk0")
FRAMES_PER_EP_CAM = 10  # 10 frames per (episode, camera) combo


def pca_3channel(patch_tokens: torch.Tensor) -> np.ndarray:
    """patch_tokens: [num_patches, D] -> [H_grid, W_grid, 3] uint8 visualisation.

    Uses torch.pca_lowrank for speed; normalises each channel to [0, 255]
    independently for the cleanest visual contrast.
    """
    # patch_tokens is [N=784, D=1024]. PCA to 3 components.
    feats = patch_tokens.float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    # Use svd_lowrank (faster than torch.pca_lowrank wrapper).
    u, s, v = torch.svd_lowrank(feats, q=3)
    pca = feats @ v  # [N, 3]
    pca = pca - pca.amin(dim=0, keepdim=True)
    pca = pca / pca.amax(dim=0, keepdim=True).clamp(min=1e-6)
    grid = int(pca.shape[0] ** 0.5)
    assert grid * grid == pca.shape[0], f"non-square grid: {pca.shape[0]} patches"
    pca = pca.reshape(grid, grid, 3)
    return (pca * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()


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

    gallery_entries: list[tuple[str, str, str]] = []  # (label, rgb_relpath, pca_relpath)

    for ep_dir in sorted(STAGED.glob("ep_*")):
        ep_id = ep_dir.name
        for cam in ("head_rgb", "wrist_rgb"):
            rgb_path = ep_dir / f"camera_{cam}.npy"
            if not rgb_path.exists():
                continue
            arr = np.load(rgb_path)  # [T, H, W, 3] uint8
            T = arr.shape[0]
            idxs = np.linspace(0, T - 1, FRAMES_PER_EP_CAM, dtype=int)
            out_dir = VIZ_OUT / ep_id / cam
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  {ep_id}/{cam}: T={T} -> {len(idxs)} frames", flush=True)

            for i, t in enumerate(idxs):
                frame_uint8 = arr[t]  # [H_src, W_src, 3] uint8, e.g., 180x320x3
                # Resize to 448x448 (encoder input resolution).
                rgb_448 = cv2.resize(frame_uint8, (448, 448), interpolation=cv2.INTER_AREA)
                # To tensor [1, 3, 448, 448] float32 in [0, 1].
                x = torch.from_numpy(rgb_448).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                x = x.to(device)

                with torch.no_grad():
                    out = tower(x, modality="rgb")  # [1, 1+R+P, D]
                # Drop CLS + register tokens; keep all 784 patch tokens.
                patch_tokens = out[0, 1 + tower.num_register_tokens:, :]  # [784, 1024]
                pca_grid = pca_3channel(patch_tokens)  # [28, 28, 3] uint8
                # Upscale PCA to 448x448 for side-by-side with RGB.
                pca_448 = cv2.resize(pca_grid, (448, 448), interpolation=cv2.INTER_NEAREST)
                # Side-by-side composite (448 x 896 x 3).
                sxs = np.concatenate([rgb_448, pca_448], axis=1)

                rgb_p = out_dir / f"frame_{i:02d}_t{t:04d}_rgb.png"
                pca_p = out_dir / f"frame_{i:02d}_t{t:04d}_pca.png"
                sxs_p = out_dir / f"frame_{i:02d}_t{t:04d}_sxs.png"
                cv2.imwrite(str(rgb_p), cv2.cvtColor(rgb_448, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(pca_p), cv2.cvtColor(pca_448, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sxs_p), cv2.cvtColor(sxs, cv2.COLOR_RGB2BGR))
                rel = sxs_p.relative_to(VIZ_OUT).as_posix()
                gallery_entries.append((f"{ep_id} / {cam} / t={t}", rel, rel))
                print(f"    [{i}/{FRAMES_PER_EP_CAM}] t={t} -> {rel}", flush=True)

    # HTML gallery — one row per side-by-side image.
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>USAM DINOv3 chunk-0 visualization</title>",
            "<style>body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:20px;}",
            "h1{color:#aef;} h2{color:#fea;margin-top:30px;border-bottom:1px solid #555;padding-bottom:4px;}",
            "img{display:block;margin:6px 0 14px 0;max-width:100%;}",
            ".row{margin:18px 0;} .label{color:#999;font-size:0.9em;}",
            "</style></head><body>",
            "<h1>USAM DINOv3 ViT-L/16 — chunk-0 patch-feature visualization</h1>",
            "<p>Left half: 448×448 RGB input. Right half: PCA(patch_tokens to 3 channels), 28×28 grid upscaled to 448×448. "
            "Coherent regions of similar color in the PCA view indicate semantically grouped patches.</p>"]
    last_section = None
    for label, _rgb_rel, sxs_rel in gallery_entries:
        section = label.rsplit(" / ", 1)[0]   # ep_xxx / cam
        if section != last_section:
            html.append(f"<h2>{escape(section)}</h2>")
            last_section = section
        html.append(f'<div class="row"><div class="label">{escape(label)}</div>'
                    f'<img src="{escape(sxs_rel)}" alt="{escape(label)}"></div>')
    html.append("</body></html>")
    (VIZ_OUT / "index.html").write_text("\n".join(html))
    print(f"\nwrote {len(gallery_entries)} side-by-side frames + index.html under {VIZ_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
