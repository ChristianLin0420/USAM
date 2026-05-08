# SPDX-License-Identifier: MIT
"""Synthesize a tiny AgiBot 2026 stand-in fixture.

The synthesizer produces an AgiBot-shaped dataset in the USAM-LeRobot v2.1
on-disk layout, sized to match what ``tests/golden_data/build_fixtures.py``
generates in ``--use-mocks`` mode. The schema mirrors
``tests/golden_data/_synthesize_tiny_droid`` but with AgiBot-specific
embodiment ("agibot_g1"), instruction-segment level_2/level_3 fields, and
two cameras (``head_rgb``, ``wrist_rgb_left``) so the dataloader's
multi-camera path is exercised.

Design constraints
------------------
* No external downloads; deterministic seeded RNG.
* Output footprint: ≤ 5 MB on disk (synthetic features are tiny).
* Idempotent: re-running over an existing ``out_root`` overwrites cleanly.
* Embodiment is ``agibot_g1`` — registered in ``prep/embodiment.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from usam.dataloader.feature_cache import write_feature_shard


N_EPISODES = 5
N_FRAMES = 24
ACTION_DIM_NATIVE = 24  # AgiBot G1 native action dim
ACTION_DIM_PADDED = 32
STATE_DIM_PADDED = 50
N_KEEP_TOKENS = 64
DINO_DIM = 768
EMBODIMENT = "agibot_g1"
FPS_NATIVE = 15
FPS_FEATURES = 5
CAMERAS = ("head_rgb", "wrist_rgb_left")


def _make_action_native(rng: np.random.Generator, T: int) -> np.ndarray:
    """Return AgiBot-shaped action_native with first 7 cols == canonical EE.

    AgiBot's converter contract pre-fills cols 0..6 with the canonical
    EE-velocity stream and pads to 32 (the global maximum). We only need
    plausible values inside the canonical bounds — the dataloader doesn't
    validate AgiBot-specific schemas at read time.
    """
    a = np.zeros((T, ACTION_DIM_PADDED), dtype=np.float32)
    a[:, 0:3] = rng.uniform(-0.5, 0.5, size=(T, 3)).astype(np.float32)
    a[:, 3:6] = rng.uniform(-1.0, 1.0, size=(T, 3)).astype(np.float32)
    a[:, 6] = rng.uniform(0.0, 1.0, size=(T,)).astype(np.float32)
    a[:, 7:ACTION_DIM_NATIVE] = rng.uniform(
        -0.1, 0.1, size=(T, ACTION_DIM_NATIVE - 7)
    ).astype(np.float32)
    return a


def _make_state(rng: np.random.Generator, T: int) -> np.ndarray:
    s = np.zeros((T, STATE_DIM_PADDED), dtype=np.float32)
    s[:, :21] = rng.uniform(-3.14, 3.14, size=(T, 21)).astype(np.float32)
    return s


def _episode_rows(ep_idx: int, rng: np.random.Generator) -> Dict[str, list]:
    action_native = _make_action_native(rng, N_FRAMES)
    action_canonical = action_native[:, :7].copy()
    state = _make_state(rng, N_FRAMES)

    am = np.zeros((ACTION_DIM_PADDED,), dtype=bool)
    am[:ACTION_DIM_NATIVE] = True
    sm = np.zeros((STATE_DIM_PADDED,), dtype=bool)
    sm[:21] = True

    # Three-level instruction segments — AgiBot's distinguishing feature.
    rows = []
    for t in range(N_FRAMES):
        # First half = "approach"; second half = "grasp" — gives the
        # subtask classifier a positive-edge transition to learn.
        is_grasp = t >= N_FRAMES // 2
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
                "level_1": "pick the red cube and place it on the saucer",
                "level_2": "grasp" if is_grasp else "approach",
                "level_3": "close" if is_grasp else "x+",
                "subtask_label": bool(is_grasp and t == N_FRAMES // 2),
            }
        )
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


def synthesize_tiny_agibot(out_root: Path, seed: int = 0xA61B0) -> Path:
    """Materialize a tiny AgiBot 2026 fixture at ``out_root``."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    info = {
        "codebase_version": "v2.1",
        "fps": FPS_NATIVE,
        "fps_features": FPS_FEATURES,
        "source": "agibot2026",
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
    merged = _merge_columns(per_ep_cols)
    parquet_dir = out_root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to materialize the tiny_agibot fixture"
        ) from e

    pq.write_table(pa.Table.from_pydict(merged), str(parquet_dir / "file-000.parquet"))
    pq.write_table(
        pa.Table.from_pylist(episodes_meta), str(out_root / "meta" / "episodes.parquet")
    )

    # ---- features cache for both cameras + all three modalities ----------
    n_feat_frames = max(
        (N_FRAMES + (FPS_NATIVE // FPS_FEATURES) - 1) // (FPS_NATIVE // FPS_FEATURES),
        4,
    )
    for cam in CAMERAS:
        for mod in ("rgb", "depth", "flow"):
            feats: Dict[int, torch.Tensor] = {}
            for ep_idx in range(N_EPISODES):
                t = torch.zeros(
                    (n_feat_frames, N_KEEP_TOKENS + 1, DINO_DIM), dtype=torch.float16
                )
                t[:, 0, 0] = float(ep_idx)
                t[:, 0, 1] = {"rgb": 0, "depth": 1, "flow": 2}[mod]
                t[:, 0, 2] = {"head_rgb": 0, "wrist_rgb_left": 1}[cam]
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
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "tiny_agibot")
    args = p.parse_args()
    out = synthesize_tiny_agibot(args.out)
    print(f"wrote tiny_agibot fixture to {out}")
