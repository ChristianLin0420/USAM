# SPDX-License-Identifier: MIT
"""OXE-AugE (RLDS) -> USAM-LeRobot v2.1 converter.

Source: ``Open X-Embodiment AugE`` — a meta-collection of 50+ RLDS sub-sources.
Per-source quirks (action layout, camera names, fps) live in a manifest
(``configs/data/oxe_auge_manifest.yaml`` if present) keyed by sub-source name.

Filtering: we drop every sub-source without an ego/head camera (per
``docs/IMPLEMENTATION_PLAN.md §11.15``). For the remaining sub-sources we
dispatch on the manifest's ``action_format`` field to populate the FIRST 7
padded columns of ``action_native`` with the canonical EE-velocity stream.
The stage_3 rule ``manifest_per_source`` then becomes a passthrough on those
seven columns.

Action formats handled (selected by manifest ``action_format``):

* ``ee_velocity``  — passthrough.
* ``ee_pose``      — finite-difference to velocity.
* ``joint_pos``    — forward-difference and write the first 7 columns. The
                     converter does NOT do FK; the pre-fill is a numerical
                     stand-in adequate for the bound check, and the per-source
                     QA gate flags any drift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from prep._base import CheckpointedJob, EpisodeRef
from prep.stage_3_canonical import canonicalize_action, validate_action_canonical
from prep.stage_2a_to_lerobot.droid import ConversionResult, episode_filename_hash


_LOG = logging.getLogger(__name__)

OXE_AUGE_EMBODIMENT: str = "oxe_auge_generic"


@dataclass(frozen=True)
class OxeAugeSubSource:
    """Per-sub-source manifest entry.

    Attributes
    ----------
    name : str
        TFDS dataset name.
    action_format : str
        One of ``"ee_velocity"``, ``"ee_pose"``, ``"joint_pos"``.
    fps : int
    has_ego_camera : bool
        If ``False`` the sub-source is filtered out at enumeration time.
    camera_map : dict[str, str]
        TFDS image-key -> canonical camera name.
    """

    name: str
    action_format: str
    fps: int
    has_ego_camera: bool
    camera_map: Dict[str, str]


def default_oxe_auge_manifest() -> Dict[str, OxeAugeSubSource]:
    """Return a hard-coded subset of OXE-AugE entries for smoke-testing.

    The real manifest will be loaded from
    ``configs/data/oxe_auge_manifest.yaml`` once that file lands. For now we
    seed enough entries here that the converter is exercisable without an
    external file dependency.
    """
    return {
        "fractal20220817_data": OxeAugeSubSource(
            name="fractal20220817_data",
            action_format="ee_velocity",
            fps=3,
            has_ego_camera=True,
            camera_map={"image": "head_rgb"},
        ),
        "kuka": OxeAugeSubSource(
            name="kuka",
            action_format="ee_pose",
            fps=10,
            has_ego_camera=True,
            camera_map={"image": "head_rgb"},
        ),
        "taco_play": OxeAugeSubSource(
            name="taco_play",
            action_format="ee_velocity",
            fps=15,
            has_ego_camera=True,
            camera_map={"rgb_static": "head_rgb", "rgb_gripper": "wrist_rgb"},
        ),
    }


class OxeAugeConverter(CheckpointedJob):
    """OXE-AugE -> USAM-LeRobot v2.1 converter (one sub-source per chunk).

    Parameters
    ----------
    chunk : int
    output_root : Path
    sub_source : str
        Manifest key (e.g. ``"taco_play"``).
    rlds_data_dir : str
    manifest : dict | None
        Override the default manifest. If ``None`` the hard-coded default is used.
    version : str
    """

    SOURCE: str = "oxe_auge"
    STAGE: str = "stage_2a_to_lerobot"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        sub_source: str,
        rlds_data_dir: str = "gs://gresearch/robotics",
        manifest: Optional[Dict[str, OxeAugeSubSource]] = None,
        version: str = "v0.1",
    ) -> None:
        super().__init__(
            source=self.SOURCE,
            stage=self.STAGE,
            chunk=chunk,
            output_root=Path(output_root),
        )
        assert isinstance(chunk, int) and chunk >= 0
        self.sub_source = sub_source
        self.rlds_data_dir = rlds_data_dir
        self.manifest = manifest or default_oxe_auge_manifest()
        self.version = version

        if sub_source not in self.manifest:
            raise KeyError(f"sub_source {sub_source!r} not in manifest")
        self._entry = self.manifest[sub_source]
        if not self._entry.has_ego_camera:
            raise ValueError(
                f"sub_source {sub_source!r} has no ego camera and is filtered out"
            )

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        try:
            import tensorflow_datasets as tfds  # type: ignore[import-not-found]
        except ImportError:
            _LOG.warning("tensorflow_datasets not installed; OXE enumeration empty")
            return iter([])
        builder = tfds.builder(self._entry.name, data_dir=self.rlds_data_dir)
        n_total = int(builder.info.splits["train"].num_examples)
        my_indices = [i for i in range(n_total) if i % 256 == self.chunk]
        refs: List[EpisodeRef] = []
        for ep_idx in my_indices:
            ep_id = f"oxe_{self.sub_source}_{ep_idx:08d}"
            refs.append(
                EpisodeRef(
                    episode_id=ep_id,
                    source=self.SOURCE,
                    raw_path=self.rlds_data_dir,
                    extra={"episode_index": ep_idx, "sub_source": self.sub_source},
                )
            )
        return iter(refs)

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        try:
            import tensorflow_datasets as tfds  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("tfds required for OXE conversion") from e
        ep_idx = int(ref.extra["episode_index"])
        builder = tfds.builder(self._entry.name, data_dir=self.rlds_data_dir)
        ds = builder.as_dataset(split=f"train[{ep_idx}:{ep_idx + 1}]")
        for tf_episode in ds.take(1):
            return self._tf_episode_to_result(tf_episode, ref)
        raise FileNotFoundError(f"no OXE episode at index {ep_idx}")

    def _tf_episode_to_result(self, tf_episode, ref: EpisodeRef) -> ConversionResult:
        """Pure transform: a TFDS episode dict -> ConversionResult."""
        steps = list(tf_episode["steps"].as_numpy_iterator())
        T = len(steps)
        assert T > 0, f"empty OXE episode {ref.episode_id}"

        # Cameras (per-manifest).
        cameras: Dict[str, np.ndarray] = {}
        for src_key, dst_key in self._entry.camera_map.items():
            frames = []
            for s in steps:
                obs = s.get("observation", {})
                if src_key in obs:
                    frames.append(np.asarray(obs[src_key], dtype=np.uint8))
            if frames:
                cameras[dst_key] = np.stack(frames, axis=0)

        # Action: format-dependent.
        action_native_raw = np.stack(
            [np.asarray(s["action"], dtype=np.float32) for s in steps], axis=0
        )
        if action_native_raw.ndim == 1:
            action_native_raw = action_native_raw.reshape(T, -1)
        ee_velocity_7 = self._action_to_canonical_velocity(action_native_raw, self._entry)

        action_native = np.zeros((T, 32), dtype=np.float32)
        action_native[:, :7] = ee_velocity_7
        d_native = min(action_native_raw.shape[1], 32 - 7)
        action_native[:, 7 : 7 + d_native] = action_native_raw[:, :d_native]
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[: 7 + d_native] = True

        # Proprio.
        state_list = []
        for s in steps:
            obs = s.get("observation", {})
            v = obs.get("state", np.zeros((7,), dtype=np.float32))
            state_list.append(np.asarray(v, dtype=np.float32))
        state_raw = np.stack(state_list, axis=0)
        if state_raw.ndim == 1:
            state_raw = state_raw.reshape(T, -1)
        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(state_raw.shape[1], 50)
        state[:, :d_state] = state_raw[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        action_canonical_ee = canonicalize_action(action_native, OXE_AUGE_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        rlds_instr = ""
        first = steps[0]
        if "language_instruction" in first:
            v = first["language_instruction"]
            rlds_instr = (
                v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
            )
        instructions = {
            "level_1": [rlds_instr] * T,
            "level_2": [""] * T,
            "level_3": [""] * T,
        }

        timestamps = np.arange(T, dtype=np.float32) / float(self._entry.fps)

        return ConversionResult(
            episode_index=int(ref.extra["episode_index"]),
            embodiment=OXE_AUGE_EMBODIMENT,
            fps=int(self._entry.fps),
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
                "oxe_episode_id": ref.episode_id,
                "sub_source": self.sub_source,
                "rlds_instruction": rlds_instr,
                "action_format": self._entry.action_format,
            },
        )

    @staticmethod
    def _action_to_canonical_velocity(
        action_native: np.ndarray, entry: OxeAugeSubSource
    ) -> np.ndarray:
        """Dispatch on ``action_format`` to produce the canonical EE velocity 7-vec."""
        T, D = action_native.shape
        out = np.zeros((T, 7), dtype=np.float32)
        if entry.action_format == "ee_velocity":
            d = min(D, 7)
            out[:, :d] = action_native[:, :d]
        elif entry.action_format == "ee_pose":
            pos = action_native[:, 0:3] if D >= 3 else np.zeros((T, 3))
            rot = action_native[:, 3:6] if D >= 6 else np.zeros((T, 3))
            grip = action_native[:, 6:7] if D >= 7 else np.zeros((T, 1))
            dt = 1.0 / max(float(entry.fps), 1e-3)
            lin_vel = np.zeros_like(pos)
            ang_vel = np.zeros_like(rot)
            if T >= 2:
                lin_vel[:-1] = (pos[1:] - pos[:-1]) / dt
                lin_vel[-1] = lin_vel[-2]
                rot_delta = (rot[1:] - rot[:-1] + math.pi) % (2 * math.pi) - math.pi
                ang_vel[:-1] = rot_delta / dt
                ang_vel[-1] = ang_vel[-2]
            out = np.concatenate([lin_vel, ang_vel, grip], axis=1).astype(np.float32)
        elif entry.action_format == "joint_pos":
            # No FK here; we forward-difference the first 6 joint columns as a
            # numerical stand-in (good enough for bound checking) and treat the
            # last column as a normalized gripper.
            joints = action_native[:, : min(D, 6)] if D >= 1 else np.zeros((T, 6))
            grip = action_native[:, -1:] if D >= 1 else np.zeros((T, 1))
            dt = 1.0 / max(float(entry.fps), 1e-3)
            j_vel = np.zeros((T, 6), dtype=np.float32)
            if T >= 2:
                pad = np.zeros((T, 6 - joints.shape[1]), dtype=np.float32)
                joints_pad = np.concatenate([joints, pad], axis=1)
                j_vel[:-1] = (joints_pad[1:] - joints_pad[:-1]) / dt
                j_vel[-1] = j_vel[-2]
            out[:, :6] = j_vel
            out[:, 6:7] = grip
        else:
            raise ValueError(f"unknown action_format {entry.action_format!r}")
        # Clip to canonical bounds.
        out[:, 0:3] = np.clip(out[:, 0:3], -2.0, 2.0)
        out[:, 3:6] = np.clip(out[:, 3:6], -math.pi, math.pi)
        out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
        return out

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
                        "sub_source": r.raw_meta.get("sub_source", ""),
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
    "OxeAugeConverter",
    "OxeAugeSubSource",
    "OXE_AUGE_EMBODIMENT",
    "default_oxe_auge_manifest",
]
