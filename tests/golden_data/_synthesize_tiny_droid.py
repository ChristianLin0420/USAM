# SPDX-License-Identifier: MIT
"""Synthesize a tiny stand-in for ``tests/golden_data/tiny_droid``.

Used by ``tests/conftest.py`` when the real LFS fixture is missing. Produces
3 fake DROID-style episodes with 30 frames each in the USAM-LeRobot v2.1
on-disk layout, plus a matching fp16 DINO feature shard. The fixture is
deliberately *small* (a few KB) so unit tests run in <1 s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from usam.dataloader.feature_cache import write_feature_shard


N_EPISODES = 3
N_FRAMES = 30
ACTION_DIM_NATIVE = 7
ACTION_DIM_PADDED = 32
STATE_DIM_PADDED = 50
N_KEEP_TOKENS = 64
DINO_DIM = 768
EMBODIMENT = "droid_franka"
FPS_NATIVE = 15
FPS_FEATURES = 5


def _make_action_native(rng: np.random.Generator, T: int) -> np.ndarray:
    """Generate a DROID-shaped action chunk inside canonical bounds."""
    a = np.zeros((T, ACTION_DIM_PADDED), dtype=np.float32)
    # linear velocity in [-0.5, 0.5] m/s
    a[:, 0:3] = rng.uniform(-0.5, 0.5, size=(T, 3)).astype(np.float32)
    # angular velocity in [-1.0, 1.0] rad/s
    a[:, 3:6] = rng.uniform(-1.0, 1.0, size=(T, 3)).astype(np.float32)
    # gripper in [0, 1]
    a[:, 6] = rng.uniform(0.0, 1.0, size=(T,)).astype(np.float32)
    return a


def _make_state(rng: np.random.Generator, T: int) -> np.ndarray:
    s = np.zeros((T, STATE_DIM_PADDED), dtype=np.float32)
    s[:, :7] = rng.uniform(-3.14, 3.14, size=(T, 7)).astype(np.float32)
    return s


def _episode_rows(ep_idx: int, rng: np.random.Generator) -> Dict[str, list]:
    """Return parquet-ready columns for one episode."""
    from prep.stage_3_canonical import canonicalize_action

    action_native = _make_action_native(rng, N_FRAMES)
    action_canonical = canonicalize_action(action_native[:, :ACTION_DIM_NATIVE], EMBODIMENT)
    state = _make_state(rng, N_FRAMES)

    am = np.zeros((ACTION_DIM_PADDED,), dtype=bool)
    am[:ACTION_DIM_NATIVE] = True
    sm = np.zeros((STATE_DIM_PADDED,), dtype=bool)
    sm[:7] = True

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
                "level_1": "pick up the object",
                "level_2": "",
                "level_3": "",
                "subtask_label": False,
            }
        )
    # Convert to columnar form
    cols: Dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols


def _merge_columns(per_ep: list[Dict[str, list]]) -> Dict[str, list]:
    out: Dict[str, list] = {k: [] for k in per_ep[0].keys()}
    for ep_cols in per_ep:
        for k in out:
            out[k].extend(ep_cols[k])
    return out


def synthesize_tiny_droid(out_root: Path, seed: int = 0xD0DD) -> Path:
    """Materialize a tiny DROID-style fixture at ``out_root``.

    Returns ``out_root`` for convenience.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- meta ------------------------------------------------------------
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": FPS_NATIVE,
        "fps_features": FPS_FEATURES,
        "source": "droid",
        "embodiment": EMBODIMENT,
        "n_episodes": N_EPISODES,
        "n_frames_per_episode": N_FRAMES,
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # ---- per-episode rows + parquet --------------------------------------
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
    merged = _merge_columns(per_ep_cols)
    parquet_dir = out_root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "file-000.parquet"

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to materialize the tiny_droid synthetic fixture; "
            "the dataloader unit tests cannot run without it."
        ) from e

    tbl = pa.Table.from_pydict(merged)
    pq.write_table(tbl, str(parquet_path))

    # ---- episodes.parquet ------------------------------------------------
    eps_path = out_root / "meta" / "episodes.parquet"
    eps_tbl = pa.Table.from_pylist(episodes_meta)
    pq.write_table(eps_tbl, str(eps_path))

    # ---- features cache --------------------------------------------------
    # Layout: features/<camera>/<modality>/chunk-XXX/file-YYY.safetensors
    # We write only the head-camera RGB cache for the smoke test; depth
    # is a zero-tensor copy so the loader contracts hold.
    n_feat_frames = (N_FRAMES + (FPS_NATIVE // FPS_FEATURES) - 1) // (FPS_NATIVE // FPS_FEATURES)
    n_feat_frames = max(n_feat_frames, 4)  # need at least history_frames
    cam = "head_rgb"
    for mod in ("rgb", "depth"):
        feats: Dict[int, torch.Tensor] = {}
        for ep_idx in range(N_EPISODES):
            t = torch.zeros((n_feat_frames, N_KEEP_TOKENS + 1, DINO_DIM), dtype=torch.float16)
            # tag the [CLS] token with episode id + modality so tests can
            # distinguish episodes / modalities without relying on randomness
            t[:, 0, 0] = float(ep_idx)
            t[:, 0, 1] = {"rgb": 0, "depth": 1}[mod]
            feats[ep_idx] = t
        shard = out_root / "features" / cam / mod / "chunk-000" / "file-000.safetensors"
        write_feature_shard(shard, feats)

    return out_root


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "tiny_droid")
    args = p.parse_args()
    out = synthesize_tiny_droid(args.out)
    print(f"wrote tiny_droid fixture to {out}")
