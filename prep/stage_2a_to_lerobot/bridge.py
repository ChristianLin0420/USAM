# SPDX-License-Identifier: MIT
"""BridgeData V2 (RLDS) -> USAM-LeRobot v2.1 converter.

Source: ``gs://gresearch/robotics/bridge`` via ``tfds.load("bridge", ...)``.

BridgeData V2 ships at 5 Hz with WidowX 250 6-DoF + gripper episodes. The
native action is a 7-D ``[dx, dy, dz, drx, dry, drz, gripper]`` delta-pose
that maps cleanly to the canonical EE-velocity passthrough rule (the dt is
constant across all frames so the deltas double as velocities). Cameras are
``image_0`` (over-the-shoulder, head_rgb) and ``image_2`` (wrist) when
present; ``image_1`` is auxiliary and skipped.

This module describes the conversion but never runs downloads.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from prep._base import CheckpointedJob, EpisodeRef
from prep.stage_3_canonical import canonicalize_action, validate_action_canonical
from prep.stage_2a_to_lerobot.droid import ConversionResult, episode_filename_hash


_LOG = logging.getLogger(__name__)

BRIDGE_EMBODIMENT: str = "bridge_widowx"
BRIDGE_FPS: int = 5
BRIDGE_NATIVE_ACTION_DIM: int = 7
BRIDGE_CAMERA_MAP: Dict[str, str] = {
    # Egocentric-only mixture: drop wrist_rgb. The model is trained on
    # head_rgb across all Tier-1 sources to keep the visual prior
    # consistent and match the egocentric human datasets (Ego4D etc.)
    # that we're mixing in. See docs/IMPLEMENTATION_PLAN.md §5.
    "image_0": "head_rgb",
}


class BridgeConverter(CheckpointedJob):
    """BridgeData V2 -> USAM-LeRobot v2.1 converter.

    Parameters
    ----------
    chunk : int
    output_root : Path
    rlds_data_dir : str
        TFDS data dir (e.g. ``"gs://gresearch/robotics"`` or local mirror).
    version : str
    """

    SOURCE: str = "bridge"
    STAGE: str = "stage_2a_to_lerobot"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        rlds_data_dir: str = "gs://gresearch/robotics",
        version: str = "v0.1",
    ) -> None:
        super().__init__(
            source=self.SOURCE,
            stage=self.STAGE,
            chunk=chunk,
            output_root=Path(output_root),
        )
        assert isinstance(chunk, int) and chunk >= 0
        self.rlds_data_dir = rlds_data_dir
        self.version = version

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        try:
            import tensorflow_datasets as tfds  # type: ignore[import-not-found]
        except ImportError:
            _LOG.warning("tensorflow_datasets not installed; Bridge enumeration empty")
            return iter([])

        # The actual BridgeData V2 RLDS is registered as `bridge_data_v2`
        # in TFDS; the legacy `bridge` builder points at v1 (different
        # feature schema, missing `episode_metadata/episode_id`) and trips
        # an `InvalidArgumentError` on first iteration.
        builder = tfds.builder("bridge_data_v2", data_dir=self.rlds_data_dir)
        n_total = int(builder.info.splits["train"].num_examples)
        # Stable shard assignment: every 256th episode goes to chunk N.
        my_indices = [i for i in range(n_total) if i % 256 == self.chunk]
        refs: List[EpisodeRef] = []
        for ep_idx in my_indices:
            ep_id = f"bridge_ep_{ep_idx:08d}"
            refs.append(
                EpisodeRef(
                    episode_id=ep_id,
                    source=self.SOURCE,
                    raw_path=self.rlds_data_dir,
                    extra={"episode_index": ep_idx},
                )
            )
        return iter(refs)

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        try:
            import tensorflow_datasets as tfds  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "Bridge conversion requires tensorflow + tensorflow_datasets at runtime"
            ) from e

        ep_idx = int(ref.extra["episode_index"])
        # The actual BridgeData V2 RLDS is registered as `bridge_data_v2`
        # in TFDS; the legacy `bridge` builder points at v1 (different
        # feature schema, missing `episode_metadata/episode_id`) and trips
        # an `InvalidArgumentError` on first iteration.
        builder = tfds.builder("bridge_data_v2", data_dir=self.rlds_data_dir)
        ds = builder.as_dataset(split=f"train[{ep_idx}:{ep_idx + 1}]")
        for tf_episode in ds.take(1):
            return self._tf_episode_to_result(tf_episode, ref)
        raise FileNotFoundError(f"no Bridge episode at index {ep_idx}")

    def _tf_episode_to_result(self, tf_episode, ref: EpisodeRef) -> ConversionResult:
        """Pure transform: a TFDS episode dict -> ConversionResult."""
        steps = list(tf_episode["steps"].as_numpy_iterator())
        T = len(steps)
        assert T > 0, f"empty Bridge episode {ref.episode_id}"

        # Cameras.
        cameras: Dict[str, np.ndarray] = {}
        for src_key, dst_key in BRIDGE_CAMERA_MAP.items():
            frames = []
            for s in steps:
                obs = s.get("observation", {})
                if src_key in obs:
                    frames.append(np.asarray(obs[src_key], dtype=np.uint8))
            if frames:
                cameras[dst_key] = np.stack(frames, axis=0)

        # Action: 7-D delta-pose + gripper.
        action_native_raw = np.stack(
            [np.asarray(s["action"], dtype=np.float32) for s in steps], axis=0
        )
        if action_native_raw.ndim == 1:
            action_native_raw = action_native_raw.reshape(T, -1)
        if action_native_raw.shape[1] > BRIDGE_NATIVE_ACTION_DIM:
            action_native_raw = action_native_raw[:, :BRIDGE_NATIVE_ACTION_DIM]
        action_native = np.zeros((T, 32), dtype=np.float32)
        action_native[:, :BRIDGE_NATIVE_ACTION_DIM] = action_native_raw
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[:BRIDGE_NATIVE_ACTION_DIM] = True

        # Proprio: WidowX exposes joint positions + EE pose under
        # ``observation/state``. Pad to 50.
        state_raw_list = []
        for s in steps:
            obs = s.get("observation", {})
            if "state" in obs:
                state_raw_list.append(np.asarray(obs["state"], dtype=np.float32))
            else:
                state_raw_list.append(np.zeros((7,), dtype=np.float32))
        state_raw = np.stack(state_raw_list, axis=0)
        if state_raw.ndim == 1:
            state_raw = state_raw.reshape(T, -1)
        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(state_raw.shape[1], 50)
        state[:, :d_state] = state_raw[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        # Bridge's deltas double as velocities (constant dt) — the canonical
        # rule is ee_velocity_passthrough so we pre-clip to the documented
        # bounds. Bridge's typical |delta| is < 0.05 so clipping is a safety net.
        clipped = action_native_raw.copy()
        clipped[:, 0:3] = np.clip(clipped[:, 0:3], -2.0, 2.0)
        clipped[:, 3:6] = np.clip(clipped[:, 3:6], -math.pi, math.pi)
        clipped[:, 6] = np.clip(clipped[:, 6], 0.0, 1.0)
        action_native[:, :BRIDGE_NATIVE_ACTION_DIM] = clipped

        action_canonical_ee = canonicalize_action(action_native, BRIDGE_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        # Language: BridgeData carries ``language_instruction`` per step.
        rlds_instr = ""
        first = steps[0]
        if "language_instruction" in first:
            v = first["language_instruction"]
            if isinstance(v, (bytes, bytearray)):
                rlds_instr = v.decode("utf-8", errors="ignore")
            else:
                rlds_instr = str(v)
        instructions = {
            "level_1": [rlds_instr] * T,
            "level_2": [""] * T,
            "level_3": [""] * T,
        }

        timestamps = np.arange(T, dtype=np.float32) / float(BRIDGE_FPS)

        return ConversionResult(
            episode_index=int(ref.extra["episode_index"]),
            embodiment=BRIDGE_EMBODIMENT,
            fps=BRIDGE_FPS,
            cameras=cameras,
            depth={},
            state=state,
            state_mask=state_mask,
            action_native=action_native,
            action_mask=action_mask,
            action_canonical_ee=action_canonical_ee,
            instructions=instructions,
            force_torque=None,
            timestamps=timestamps,
            raw_meta={
                "bridge_episode_id": ref.episode_id,
                "rlds_instruction": rlds_instr,
            },
        )

    # ----- shard writing ---------------------------------------------------

    def write_shard(self, results: List[ConversionResult]) -> Path:
        assert len(results) > 0, "empty shard"
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("pyarrow required for shard writing") from e

        rows = []
        for r in results:
            T = r.action_canonical_ee.shape[0]
            for t in range(T):
                rows.append(
                    {
                        "episode_index": r.episode_index,
                        "frame_index": t,
                        "timestamp": float(r.timestamps[t]),
                        "embodiment": r.embodiment,
                        "proprio": r.state[t].tolist(),
                        "action_native": r.action_native[t].tolist(),
                        "action_canonical_ee": r.action_canonical_ee[t].tolist(),
                        "action_mask": r.action_mask.tolist(),
                        "state_mask": r.state_mask.tolist(),
                        "level_1": r.instructions["level_1"][t],
                        "level_2": r.instructions["level_2"][t],
                        "level_3": r.instructions["level_3"][t],
                        "subtask_label": False,
                    }
                )

        table = pa.Table.from_pylist(rows)
        h = self.shard_hash(results)
        out = self.output_dir / f"file-{h}.parquet"
        if not out.exists():
            pq.write_table(table, str(out))
        return out


__all__ = [
    "BridgeConverter",
    "BRIDGE_EMBODIMENT",
    "BRIDGE_FPS",
    "BRIDGE_NATIVE_ACTION_DIM",
    "BRIDGE_CAMERA_MAP",
]
