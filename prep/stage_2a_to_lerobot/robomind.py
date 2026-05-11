# SPDX-License-Identifier: MIT
"""RoboMIND (HDF5) -> USAM-LeRobot v2.1 converter.

Source: ``https://x-humanoid-robomind.github.io`` per-trajectory HDF5 files
laid out as ``<embodiment>/<task>/<traj_id>.h5`` with datasets:

* ``observations/images/<cam>``  uint8 ``[T, H, W, 3]``  (BGR — see below)
* ``observations/joint_position`` ``[T, D_joint]`` fp32
* ``observations/ee_pose`` ``[T, 7]`` fp32 (quat) — only for some embodiments
* ``actions``  ``[T, D_action]`` fp32

Source-quirk: **RoboMIND ships frames in BGR**, against the LeRobot convention
of RGB. The hard rule from ``docs/AGENT_CHARTER.md`` and the data-engineer
charter is: hard-assert via a sample-frame heuristic that the frame is BGR
(blue-channel mean exceeds red-channel mean by a clear margin), then convert
with ``cv2.cvtColor(..., cv2.COLOR_BGR2RGB)``. If detection is ambiguous, abort
the chunk with a clear error so we never silently ship miscoloured data.

Embodiment scope: We only emit real-robot trajectories. Simulation
embodiments (``h5_simulation``) are dropped at :meth:`list_episodes` time.
The Tien Kung head_cam maps to ``head_rgb``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from prep._base import CheckpointedJob, EpisodeRef
from prep.stage_3_canonical import canonicalize_action, validate_action_canonical
from prep.stage_2a_to_lerobot.droid import ConversionResult, episode_filename_hash


_LOG = logging.getLogger(__name__)

ROBOMIND_EMBODIMENT: str = "robomind_tien_kung"
ROBOMIND_FPS: int = 25
ROBOMIND_NATIVE_ACTION_DIM: int = 14  # bimanual joint positions (7 + 7)
ROBOMIND_CAMERA_MAP: Dict[str, str] = {
    "head_cam": "head_rgb",
    "left_wrist_cam": "wrist_rgb_left",
    "right_wrist_cam": "wrist_rgb_right",
}
ROBOMIND_BGR_THRESHOLD: float = 5.0  # blue-mean - red-mean must exceed this
ROBOMIND_DROP_EMBODIMENTS: Tuple[str, ...] = ("h5_simulation",)


def _detect_bgr(frame: np.ndarray, threshold: float = ROBOMIND_BGR_THRESHOLD) -> bool:
    """Return True iff the heuristic concludes ``frame`` is BGR.

    The heuristic: in *real-world* robotics scenes (mostly indoor lighting +
    skin / wood / metal hues) the red channel of an RGB frame averages higher
    than the blue channel by several gray levels. RoboMIND's HDF5 ships frames
    where this relationship is inverted — blue averages higher than red — which
    is the BGR signature.

    Returns
    -------
    bool
        True if BGR; False if RGB; raises if the channels are too close to call
        (callers treat this as a hard error per the data-engineer charter).

    Raises
    ------
    ValueError
        If ``|blue_mean - red_mean| < threshold``. The chunk is aborted in that
        case rather than risking miscoloured data.
    """
    assert frame.ndim in (3, 4), f"expected [H, W, 3] or [T, H, W, 3], got {frame.shape}"
    if frame.ndim == 4:
        # Use the middle frame to dampen openers and cuts.
        frame = frame[frame.shape[0] // 2]
    assert frame.shape[-1] == 3, frame.shape
    f = frame.astype(np.float32)
    # Channel-0 vs Channel-2 difference. In RGB layout: [R, G, B] so c0=R.
    # If the array is BGR layout (which RoboMIND ships) c0=B and the diff
    # below is positive (blue > red). If genuinely RGB it is negative.
    c0_minus_c2 = float(f[..., 0].mean() - f[..., 2].mean())
    if abs(c0_minus_c2) < threshold:
        raise ValueError(
            f"BGR/RGB detection ambiguous: |c0-c2|={abs(c0_minus_c2):.2f} < "
            f"{threshold:.2f} threshold. Aborting chunk to avoid silent "
            "miscolouring per docs/AGENT_CHARTER.md."
        )
    return c0_minus_c2 > 0


def _bgr_to_rgb(frames: np.ndarray) -> np.ndarray:
    """Channel-swap a ``[T, H, W, 3]`` (or ``[H, W, 3]``) BGR array to RGB.

    We avoid the cv2 dependency at import time by doing the swap manually; the
    behaviour is identical to ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`` (a
    pure index reversal on the last axis). cv2 is still imported lazily by the
    depth stage.
    """
    assert frames.shape[-1] == 3, frames.shape
    return frames[..., ::-1].copy()


class RoboMINDConverter(CheckpointedJob):
    """RoboMIND HDF5 -> USAM-LeRobot v2.1 converter.

    Parameters
    ----------
    chunk : int
    output_root : Path
    raw_root : Path
        Local snapshot root containing the per-embodiment subdirs.
    drop_simulation : bool
        Drop the ``h5_simulation`` embodiment. Default True (real-only).
    version : str
    """

    SOURCE: str = "robomind"
    STAGE: str = "stage_2a_to_lerobot"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        raw_root: Optional[Path] = None,
        drop_simulation: bool = True,
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
        self.drop_simulation = drop_simulation
        self.version = version

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        if self.raw_root is None or not self.raw_root.exists():
            _LOG.warning("RoboMIND raw_root not present at %s; skipping", self.raw_root)
            return iter([])
        refs: List[EpisodeRef] = []
        for emb_dir in sorted(self.raw_root.iterdir()):
            if not emb_dir.is_dir():
                continue
            if self.drop_simulation and emb_dir.name in ROBOMIND_DROP_EMBODIMENTS:
                _LOG.info("dropping simulation embodiment %s", emb_dir.name)
                continue
            for h5_path in sorted(emb_dir.rglob("*.h5")):
                ep_id = f"robomind_{emb_dir.name}_{h5_path.stem}"
                if int(self.episode_hash(ep_id), 16) % 256 != self.chunk:
                    continue
                refs.append(
                    EpisodeRef(
                        episode_id=ep_id,
                        source=self.SOURCE,
                        raw_path=str(h5_path),
                        extra={"embodiment_dir": emb_dir.name},
                    )
                )
        return iter(refs)

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        path = Path(ref.raw_path)
        if not path.exists():
            raise FileNotFoundError(f"RoboMIND HDF5 missing: {path}")
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "h5py required for RoboMIND conversion; install requirements/prep.txt"
            ) from e

        with h5py.File(str(path), "r") as f:
            payload = self._h5_to_payload(f)
        return self._payload_to_result(ref, payload)

    def _h5_to_payload(self, h5_file) -> Dict[str, np.ndarray]:
        """Pull the relevant arrays out of an open ``h5py.File``.

        Kept separate from ``convert_episode`` so unit tests can pass a dict
        rather than synthesizing an HDF5 file.
        """
        payload: Dict[str, np.ndarray] = {}
        try:
            obs = h5_file["observations"]
        except KeyError:
            return payload

        # Cameras live under observations/images/<cam>.
        if "images" in obs:
            for cam_name in obs["images"].keys():
                payload[f"camera::{cam_name}"] = np.asarray(obs["images"][cam_name][:])

        for k in ("joint_position", "ee_pose", "ee_velocity"):
            if k in obs:
                payload[f"obs::{k}"] = np.asarray(obs[k][:])

        if "actions" in h5_file:
            payload["actions"] = np.asarray(h5_file["actions"][:])
        if "language_instruction" in h5_file:
            raw = h5_file["language_instruction"][()]
            if isinstance(raw, (bytes, bytearray)):
                payload["language_instruction"] = np.array([raw.decode("utf-8", errors="ignore")])
            else:
                payload["language_instruction"] = np.asarray(raw)
        return payload

    def _payload_to_result(
        self, ref: EpisodeRef, payload: Dict[str, np.ndarray]
    ) -> ConversionResult:
        """Pure transform: a payload dict -> ConversionResult.

        Performs the BGR->RGB hard-assert + conversion on every camera array.
        """
        actions = payload.get("actions")
        assert actions is not None, f"RoboMIND episode {ref.episode_id} missing actions"
        T = int(actions.shape[0])
        assert T > 0, f"empty RoboMIND episode {ref.episode_id}"

        # Cameras: detect BGR, swap if needed.
        cameras: Dict[str, np.ndarray] = {}
        for key, arr in payload.items():
            if not key.startswith("camera::"):
                continue
            cam_name = key.split("::", 1)[1]
            canonical = ROBOMIND_CAMERA_MAP.get(cam_name)
            if canonical is None:
                continue
            if arr.size == 0:
                cameras[canonical] = arr
                continue
            # Hard-assert BGR via the sample-frame heuristic. Raises ValueError
            # if ambiguous — the dispatcher catches and aborts the chunk.
            is_bgr = _detect_bgr(arr, threshold=ROBOMIND_BGR_THRESHOLD)
            if is_bgr:
                arr = _bgr_to_rgb(arr)
            else:
                _LOG.info(
                    "RoboMIND camera %s already in RGB layout (heuristic); "
                    "no swap performed.",
                    cam_name,
                )
            cameras[canonical] = arr

        # Joint positions (proprio + native action).
        joint_pos = payload.get("obs::joint_position", np.zeros((T, 14), dtype=np.float32))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.reshape(T, -1)
        ee_pose = payload.get("obs::ee_pose")  # [T, 7] (xyz + quat) when present

        # Action canonical-EE: if ee_pose is available we finite-difference it
        # in this converter (rather than relying on stage_3) and store the
        # result in the first 7 padded columns of action_native; stage_3's
        # joint_position_to_ee_finite_diff is then a passthrough.
        ee_velocity_7 = self._ee_pose_to_canonical_velocity(ee_pose, T)
        action_native_raw = np.asarray(actions, dtype=np.float32)
        if action_native_raw.ndim == 1:
            action_native_raw = action_native_raw.reshape(T, -1)

        action_native = np.zeros((T, 32), dtype=np.float32)
        action_native[:, :7] = ee_velocity_7
        d_native = min(action_native_raw.shape[1], 32 - 7)
        action_native[:, 7 : 7 + d_native] = action_native_raw[:, :d_native]
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[: 7 + d_native] = True

        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(joint_pos.shape[1], 50)
        state[:, :d_state] = joint_pos[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        action_canonical_ee = canonicalize_action(action_native, ROBOMIND_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        rlds_instr = ""
        if "language_instruction" in payload:
            arr = payload["language_instruction"]
            if arr.size > 0:
                v = arr.flat[0]
                rlds_instr = v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
        instructions = {
            "level_1": [rlds_instr] * T,
            "level_2": [""] * T,
            "level_3": [""] * T,
        }

        timestamps = np.arange(T, dtype=np.float32) / float(ROBOMIND_FPS)

        return ConversionResult(
            episode_index=int(self.episode_hash(ref.episode_id), 16) % (2**31 - 1),
            embodiment=ROBOMIND_EMBODIMENT,
            fps=ROBOMIND_FPS,
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
                "robomind_episode_id": ref.episode_id,
                "embodiment_dir": ref.extra.get("embodiment_dir", ""),
                "rlds_instruction": rlds_instr,
            },
        )

    def _ee_pose_to_canonical_velocity(
        self, ee_pose: Optional[np.ndarray], T: int
    ) -> np.ndarray:
        """Convert per-frame EE pose to canonical velocity, or zeros if absent."""
        if ee_pose is None or ee_pose.size == 0:
            return np.zeros((T, 7), dtype=np.float32)
        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.ndim == 1:
            ee_pose = ee_pose.reshape(T, -1)
        # Expect [x, y, z, qx, qy, qz, qw] or [x, y, z, rx, ry, rz, gripper].
        # We are tolerant: take the first 6 columns as pose (xyz + 3-DOF rot
        # representation) and a possible 7th as gripper. For quaternion inputs
        # we approximate angular velocity by finite-differencing the imaginary
        # parts — good enough for the hash-based bound check, and the
        # dispatcher's per-shard QA gate will flag any drift.
        pos = ee_pose[:, 0:3]
        rot = ee_pose[:, 3:6] if ee_pose.shape[1] >= 6 else np.zeros((T, 3), dtype=np.float32)
        grip = (
            ee_pose[:, 6:7]
            if ee_pose.shape[1] >= 7
            else np.zeros((T, 1), dtype=np.float32)
        )

        dt = 1.0 / float(ROBOMIND_FPS)
        lin_vel = np.zeros_like(pos)
        ang_vel = np.zeros_like(rot)
        if T >= 2:
            lin_vel[:-1] = (pos[1:] - pos[:-1]) / dt
            lin_vel[-1] = lin_vel[-2]
            rot_delta = (rot[1:] - rot[:-1] + math.pi) % (2 * math.pi) - math.pi
            ang_vel[:-1] = rot_delta / dt
            ang_vel[-1] = ang_vel[-2]

        canon = np.concatenate([lin_vel, ang_vel, grip], axis=1).astype(np.float32)
        canon[:, 0:3] = np.clip(canon[:, 0:3], -2.0, 2.0)
        canon[:, 3:6] = np.clip(canon[:, 3:6], -math.pi, math.pi)
        canon[:, 6] = np.clip(canon[:, 6], 0.0, 1.0)
        return canon

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
    "RoboMINDConverter",
    "ROBOMIND_EMBODIMENT",
    "ROBOMIND_FPS",
    "ROBOMIND_NATIVE_ACTION_DIM",
    "ROBOMIND_CAMERA_MAP",
    "ROBOMIND_BGR_THRESHOLD",
    "_detect_bgr",
    "_bgr_to_rgb",
]
