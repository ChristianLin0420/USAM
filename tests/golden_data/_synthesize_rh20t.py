# SPDX-License-Identifier: MIT
"""Synthesize a tiny RH20T stand-in fixture.

RH20T uses pose-finite-diff canonicalization (``rh20t_franka`` embodiment),
runs at 10 Hz, and carries an optional 6-D force-torque channel. The
synthesizer pre-computes finite-diff velocities so cols 0..6 of
``action_native`` are the canonical EE-velocity stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from usam.dataloader.feature_cache import write_feature_shard


N_EPISODES = 3
N_FRAMES = 24
ACTION_DIM_NATIVE = 7
ACTION_DIM_PADDED = 32
STATE_DIM_PADDED = 50
N_KEEP_TOKENS = 64
DINO_DIM = 768
EMBODIMENT = "rh20t_franka"
FPS_NATIVE = 10
FPS_FEATURES = 5
CAMERAS = ("head_rgb",)


def _make_action_native(rng: np.random.Generator, T: int) -> np.ndarray:
    """Synthetic finite-diff velocities — within canonical bounds."""
    a = np.zeros((T, ACTION_DIM_PADDED), dtype=np.float32)
    a[:, 0:3] = rng.uniform(-0.3, 0.3, size=(T, 3)).astype(np.float32)
    a[:, 3:6] = rng.uniform(-0.5, 0.5, size=(T, 3)).astype(np.float32)
    a[:, 6] = rng.uniform(0.0, 1.0, size=(T,)).astype(np.float32)
    return a


def _make_state(rng: np.random.Generator, T: int) -> np.ndarray:
    s = np.zeros((T, STATE_DIM_PADDED), dtype=np.float32)
    s[:, :7] = rng.uniform(-3.14, 3.14, size=(T, 7)).astype(np.float32)
    # Force-torque columns 7:13 (RH20T's distinguishing channel).
    s[:, 7:13] = rng.uniform(-2.0, 2.0, size=(T, 6)).astype(np.float32)
    return s


def _episode_rows(ep_idx: int, rng: np.random.Generator) -> Dict[str, list]:
    action_native = _make_action_native(rng, N_FRAMES)
    action_canonical = action_native[:, :7].copy()
    state = _make_state(rng, N_FRAMES)

    am = np.zeros((ACTION_DIM_PADDED,), dtype=bool)
    am[:ACTION_DIM_NATIVE] = True
    sm = np.zeros((STATE_DIM_PADDED,), dtype=bool)
    sm[:13] = True

    rows = []
    for t in range(N_FRAMES):
        rows.append(
            {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": float(t) / FPS_NATIVE,
                "embodiment": EMBODIMENT,
                "proprio": state[t].tolist(),
                "action_native": action_native[t].tolist(),
                "action_canonical_ee": action_canonical[t].tolist(),
                "action_mask": am.tolist(),
                "state_mask": sm.tolist(),
                "level_1": "wipe the table",
                "level_2": "",
                "level_3": "",
                "subtask_label": False,
            }
        )
    cols: Dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols


def _merge(per_ep: list[Dict[str, list]]) -> Dict[str, list]:
    out: Dict[str, list] = {k: [] for k in per_ep[0].keys()}
    for ep_cols in per_ep:
        for k in out:
            out[k].extend(ep_cols[k])
    return out


def synthesize_tiny_rh20t(out_root: Path, seed: int = 0x12087) -> Path:
    """Materialize a tiny RH20T fixture."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    info = {
        "codebase_version": "v2.1",
        "fps": FPS_NATIVE,
        "fps_features": FPS_FEATURES,
        "source": "rh20t",
        "embodiment": EMBODIMENT,
        "n_episodes": N_EPISODES,
        "n_frames_per_episode": N_FRAMES,
        "cameras": list(CAMERAS),
    }
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    per_ep_cols = []
    episodes_meta = []
    for ep_idx in range(N_EPISODES):
        per_ep_cols.append(_episode_rows(ep_idx, rng))
        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "length": N_FRAMES,
                "chunk": 0,
                "file": 0,
                "embodiment": EMBODIMENT,
            }
        )
    merged = _merge(per_ep_cols)
    parquet_dir = out_root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to materialize the tiny_rh20t fixture"
        ) from e

    pq.write_table(pa.Table.from_pydict(merged), str(parquet_dir / "file-000.parquet"))
    pq.write_table(
        pa.Table.from_pylist(episodes_meta), str(out_root / "meta" / "episodes.parquet")
    )

    n_feat_frames = max(
        (N_FRAMES + (FPS_NATIVE // FPS_FEATURES) - 1) // (FPS_NATIVE // FPS_FEATURES),
        4,
    )
    for cam in CAMERAS:
        for mod in ("rgb", "depth"):
            feats: Dict[int, torch.Tensor] = {}
            for ep_idx in range(N_EPISODES):
                t = torch.zeros(
                    (n_feat_frames, N_KEEP_TOKENS + 1, DINO_DIM), dtype=torch.float16
                )
                t[:, 0, 0] = float(ep_idx)
                t[:, 0, 1] = {"rgb": 0, "depth": 1}[mod]
                feats[ep_idx] = t
            shard = (
                out_root
                / "features"
                / cam
                / mod
                / "chunk-000"
                / "file-000.safetensors"
            )
            write_feature_shard(shard, feats)

    return out_root


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "tiny_rh20t")
    args = p.parse_args()
    out = synthesize_tiny_rh20t(args.out)
    print(f"wrote tiny_rh20t fixture to {out}")
