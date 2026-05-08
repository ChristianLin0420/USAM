# SPDX-License-Identifier: MIT
"""RH20T (mp4 + json) -> USAM-LeRobot v2.1 converter.

Source: ``rh20t.github.io`` per-task tarballs containing per-camera mp4s, a
``cam_<serial>/``-prefixed metadata json, and a ``transformed/`` action json.
The 7 robot configurations (Franka, UR5, Kuka, …) ship with different camera
serials; the canonical mapping per config lives in
``configs/data/camera_maps/rh20t.yaml``.

Action canonicalization: native action is a 6-DoF EE pose ``[x, y, z, rx, ry,
rz]`` plus a gripper width in mm. We convert to the canonical EE-velocity
frame via ``stage_3_canonical._canon_ee_pose_finite_diff`` (rule
``ee_pose_finite_diff``). Force-torque (6 DoF, F/T sensor) is preserved.

Frame extraction: the action json is sampled at 10 Hz; the per-camera mp4s
are at 30 Hz. We sync by **timestamp interpolation**: for each action frame we
pick the nearest video frame whose timestamp (from the per-frame ``cam_*.json``
metadata sidecar) is closest. The synthetic test path skips the mp4 read
entirely and yields zero-frame camera arrays.
"""

from __future__ import annotations

import json
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

RH20T_EMBODIMENT: str = "rh20t_franka"
RH20T_FPS: int = 10  # canonical action fps
RH20T_VIDEO_FPS: int = 30  # camera frame fps
RH20T_NATIVE_ACTION_DIM: int = 7  # [x, y, z, rx, ry, rz, gripper]


def _select_nearest_indices(
    target_ts: np.ndarray, source_ts: np.ndarray
) -> np.ndarray:
    """Map every timestamp in ``target_ts`` to the nearest index in ``source_ts``.

    Both arrays must be 1-D and monotonically non-decreasing. Used to align the
    10 Hz action stream with the 30 Hz per-camera frames.
    """
    assert target_ts.ndim == 1 and source_ts.ndim == 1, (target_ts.shape, source_ts.shape)
    if source_ts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    # np.searchsorted gives the insertion point; clip to valid range and pick
    # whichever neighbour has smaller |dt|.
    pos = np.clip(np.searchsorted(source_ts, target_ts), 1, source_ts.size - 1)
    left = pos - 1
    right = pos
    left_dt = np.abs(target_ts - source_ts[left])
    right_dt = np.abs(target_ts - source_ts[right])
    out = np.where(left_dt <= right_dt, left, right).astype(np.int64)
    return out


class RH20TConverter(CheckpointedJob):
    """RH20T -> USAM-LeRobot v2.1 converter.

    Parameters
    ----------
    chunk : int
        Chunk id assigned by ``prep.dispatch``.
    output_root : Path
        ``/scratch/usam/rh20t/2a/chunk-XXX/``.
    raw_root : Path
        Local snapshot of one (or all) RH20T configurations. We expect the
        on-disk layout
        ``raw_root/<config>/task_<id>_user_<id>_scene_<id>/{cam_*, transformed}``.
    config : str | None
        e.g. ``"RH20T_cfg1"``. ``None`` means the converter will scan every
        config sub-dir under ``raw_root``.
    camera_map : dict[str, str] | None
        ``cam_<serial> -> canonical_key`` overrides. ``None`` falls back to
        the YAML in ``configs/data/camera_maps/rh20t.yaml`` if loaded.
    version : str
    """

    SOURCE: str = "rh20t"
    STAGE: str = "stage_2a_to_lerobot"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        raw_root: Optional[Path] = None,
        config: Optional[str] = None,
        camera_map: Optional[Dict[str, str]] = None,
        version: str = "v0.1",
    ) -> None:
        super().__init__(
            source=self.SOURCE,
            stage=self.STAGE,
            chunk=chunk,
            output_root=Path(output_root),
        )
        assert isinstance(chunk, int) and chunk >= 0
        self.raw_root = Path(raw_root) if raw_root else None
        self.config = config
        self.camera_map = camera_map or {}
        self.version = version

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        if self.raw_root is None or not self.raw_root.exists():
            _LOG.warning("RH20T raw_root not present at %s; skipping", self.raw_root)
            return iter([])
        configs = (
            [self.config] if self.config else [p.name for p in self.raw_root.glob("RH20T_cfg*")]
        )
        refs: List[EpisodeRef] = []
        for cfg in configs:
            cfg_root = self.raw_root / cfg
            if not cfg_root.exists():
                continue
            for ep_dir in sorted(cfg_root.glob("task_*")):
                ep_id = f"rh20t_{cfg}_{ep_dir.name}"
                # Stable shard assignment by hash of episode_id.
                if int(self.episode_hash(ep_id), 16) % 256 != self.chunk:
                    continue
                refs.append(
                    EpisodeRef(
                        episode_id=ep_id,
                        source=self.SOURCE,
                        raw_path=str(ep_dir),
                        extra={"config": cfg},
                    )
                )
        return iter(refs)

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        ep_dir = Path(ref.raw_path)
        if not ep_dir.exists():
            raise FileNotFoundError(f"RH20T episode dir missing: {ep_dir}")
        action_json = self._find_action_json(ep_dir)
        if action_json is None:
            raise FileNotFoundError(f"no transformed/action.json under {ep_dir}")
        action_data = json.loads(action_json.read_text())
        return self._json_to_result(ref, ep_dir, action_data)

    def _find_action_json(self, ep_dir: Path) -> Optional[Path]:
        """Locate the JSON action stream RH20T provides under ``transformed/``."""
        for candidate in (
            ep_dir / "transformed" / "action.json",
            ep_dir / "transformed.json",
            ep_dir / "action.json",
        ):
            if candidate.exists():
                return candidate
        return None

    def _json_to_result(
        self, ref: EpisodeRef, ep_dir: Path, action_data: Dict
    ) -> ConversionResult:
        """Pure transform: a parsed action json + camera dir -> ConversionResult."""
        # ``frames`` is a list of {"timestamp", "tcp_pose", "gripper", "ft", "joints"}.
        frames = action_data.get("frames", [])
        assert isinstance(frames, list), "expected RH20T frames as a list"
        T = len(frames)
        assert T > 0, f"empty RH20T episode {ref.episode_id}"

        # Per-frame native action: [x, y, z, rx, ry, rz, gripper].
        action_native_raw = np.zeros((T, RH20T_NATIVE_ACTION_DIM), dtype=np.float32)
        timestamps = np.zeros((T,), dtype=np.float32)
        force_torque = np.zeros((T, 6), dtype=np.float32)
        for i, fr in enumerate(frames):
            timestamps[i] = float(fr.get("timestamp", i / RH20T_FPS))
            tcp = fr.get("tcp_pose", [0.0] * 6)
            action_native_raw[i, 0:3] = np.asarray(tcp[0:3], dtype=np.float32)
            action_native_raw[i, 3:6] = np.asarray(tcp[3:6], dtype=np.float32)
            # Gripper width in mm -> 0..1 normalized (RH20T's max ~85 mm).
            grip_mm = float(fr.get("gripper", 0.0))
            action_native_raw[i, 6] = float(np.clip(grip_mm / 85.0, 0.0, 1.0))
            ft = fr.get("ft", [0.0] * 6)
            force_torque[i] = np.asarray(ft, dtype=np.float32)

        action_native = np.zeros((T, 32), dtype=np.float32)
        action_native[:, :RH20T_NATIVE_ACTION_DIM] = action_native_raw
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[:RH20T_NATIVE_ACTION_DIM] = True

        # Proprio: joint positions.
        joints = np.asarray([fr.get("joints", [0.0] * 7) for fr in frames], dtype=np.float32)
        if joints.ndim == 1:
            joints = joints.reshape(T, -1)
        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(joints.shape[1], 50)
        state[:, :d_state] = joints[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        # Cameras: load decoded frames lazily; the dispatcher prefers stages
        # 2b/2c to do the heavy decode, so we leave camera arrays empty here
        # (downstream stages key off raw_meta["camera_video_paths"]).
        cameras: Dict[str, np.ndarray] = {}
        camera_video_paths: Dict[str, str] = {}
        for cam_dir in sorted(ep_dir.glob("cam_*")):
            serial = cam_dir.name
            canonical = self.camera_map.get(serial)
            if canonical is None:
                continue
            mp4s = sorted(cam_dir.glob("*.mp4"))
            if not mp4s:
                continue
            camera_video_paths[canonical] = str(mp4s[0])

        action_canonical_ee = canonicalize_action(action_native, RH20T_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        instructions = {
            "level_1": [str(action_data.get("task_description", ""))] * T,
            "level_2": [""] * T,
            "level_3": [""] * T,
        }

        return ConversionResult(
            episode_index=int(self.episode_hash(ref.episode_id), 16) % (2**31 - 1),
            embodiment=RH20T_EMBODIMENT,
            fps=RH20T_FPS,
            cameras=cameras,
            depth={},
            state=state,
            state_mask=state_mask,
            action_native=action_native,
            action_mask=action_mask,
            action_canonical_ee=action_canonical_ee,
            instructions=instructions,
            force_torque=force_torque,
            timestamps=timestamps,
            raw_meta={
                "rh20t_episode_id": ref.episode_id,
                "config": ref.extra.get("config", ""),
                "camera_video_paths": camera_video_paths,
                "task_description": str(action_data.get("task_description", "")),
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
                ft = r.force_torque[t].tolist() if r.force_torque is not None else [0.0] * 6
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
                        "force_torque": ft,
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
    "RH20TConverter",
    "RH20T_EMBODIMENT",
    "RH20T_FPS",
    "RH20T_VIDEO_FPS",
    "RH20T_NATIVE_ACTION_DIM",
    "_select_nearest_indices",
]
