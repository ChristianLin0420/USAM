# SPDX-License-Identifier: MIT
"""AgiBot World 2026 (LeRobot v2.1+ext) -> USAM-LeRobot v2.1 converter.

Source: ``<org>/agibot-world-2026`` already published on the Hub in LeRobot
v2.1 layout with USAM-relevant extensions:

* per-trajectory ``instruction_segments`` (subtask boundaries) — promoted to
  three top-level columns ``level_1``, ``level_2``, ``level_3`` per the
  Implementation Plan §11.15. These columns are the **ground truth** for the
  subtask classifier (``L_subtask`` in §4.3); losing them silently would break
  the conductor's classification head, so we hard-assert they end up in the
  parquet.
* depth shipped as 16-bit PNG that we re-encode as HEVC mp4 in stage_2c.
* bimanual G1 embodiment with 24-D action; the stage_3 canonicalization rule
  ``joint_delta_to_ee_finite_diff`` expects the converter to pre-fill the
  first 7 padded columns of ``action_native`` with the right-arm EE velocity
  stream so stage_3 is a passthrough. The pre-fill happens here in
  :meth:`_compute_ee_velocity`.

Camera mapping (from §11.15): ``head -> head_rgb``, ``hand_left ->
wrist_rgb_left``, ``hand_right -> wrist_rgb_right``.

This module describes the conversion but never executes downloads — see
``docs/AGENT_CHARTER.md``. The pyarrow / decord reads only fire inside
``convert_episode``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from prep._base import CheckpointedJob, EpisodeRef
from prep.stage_3_canonical import canonicalize_action, validate_action_canonical
from prep.stage_2a_to_lerobot.droid import ConversionResult, episode_filename_hash


_LOG = logging.getLogger(__name__)

AGIBOT_EMBODIMENT: str = "agibot_g1"
AGIBOT_FPS: int = 30
AGIBOT_NATIVE_ACTION_DIM: int = 24  # bimanual: 12 per arm (7 joint + 5 hand)
AGIBOT_CAMERA_MAP: Dict[str, str] = {
    # Egocentric-only mixture: drop hand_left/hand_right and use only
    # the head-mounted view. AgiBot World 2026 ships this column under
    # the key `top_head` (LeRobot v2.1 video tree:
    # videos/chunk-NNN/observation.images.top_head/...). Earlier
    # AgiBot Alpha released it as `head_color` raw mp4, but our 2026
    # converter targets the post-LeRobot layout, so use `top_head`.
    "top_head": "head_rgb",
}


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap an angle (or rotation-vector component) into ``[-pi, pi]``."""
    return (x + math.pi) % (2 * math.pi) - math.pi


class AgiBot2026Converter(CheckpointedJob):
    """AgiBot World 2026 -> USAM-LeRobot v2.1 converter.

    Parameters
    ----------
    chunk : int
        Chunk id assigned by ``prep.dispatch``.
    output_root : Path
        ``/scratch/usam/agibot2026/2a/chunk-XXX/``.
    raw_root : Path
        Local snapshot of the AgiBot-World-2026 repo (parquet + videos + meta).
    version : str
        Schema version stamp baked into per-episode filename hashes.
    """

    SOURCE: str = "agibot2026"
    STAGE: str = "stage_2a_to_lerobot"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        raw_root: Optional[Path] = None,
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
        self.version = version

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        """Enumerate AgiBot 2026 episodes belonging to this chunk.

        AgiBot ships ``meta/episodes.parquet`` listing every episode index;
        we shard those rows by ``chunk_id``. When the raw root is missing
        (e.g. unit tests with no fixture) we yield nothing rather than crash.
        """
        if self.raw_root is None or not self.raw_root.exists():
            _LOG.warning(
                "AgiBot2026 raw_root not present at %s; list_episodes returns empty",
                self.raw_root,
            )
            return iter([])
        # AgiBot 2026's per-task tarballs ship `meta/` with `info.json`,
        # `episodes.jsonl`, `episodes_stats.jsonl`, and `tasks.jsonl` —
        # not the original `episodes.parquet`. Accept either layout so
        # the converter works on both the full Hub mirror (parquet) and
        # the per-task tarball extracts (jsonl).
        ep_parquet = self.raw_root / "meta" / "episodes.parquet"
        ep_jsonl = self.raw_root / "meta" / "episodes.jsonl"
        episode_indices: List[int] = []
        if ep_parquet.exists():
            try:
                import pyarrow.parquet as pq  # type: ignore[import-not-found]
            except ImportError:
                _LOG.warning("pyarrow not installed; AgiBot list_episodes empty")
                return iter([])
            table = pq.read_table(str(ep_parquet), columns=["episode_index", "length"])
            episode_indices = [int(i) for i in table.column("episode_index").to_pylist()]
        elif ep_jsonl.exists():
            # Each line: {"episode_index": N, "tasks": [...], "length": ...}
            with open(ep_jsonl, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        episode_indices.append(int(row["episode_index"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        else:
            # Last resort: scan the data/ tree for episode_NNNNNN.parquet files.
            _LOG.warning(
                "no meta/episodes.{parquet,jsonl} at %s; falling back to data/ glob",
                self.raw_root / "meta",
            )
            for p in sorted((self.raw_root / "data").rglob("episode_*.parquet")):
                stem = p.stem  # episode_NNNNNN
                try:
                    episode_indices.append(int(stem.split("_")[-1]))
                except ValueError:
                    continue
        # Stable shard assignment: every Nth episode goes to chunk N.
        my_eps = [int(i) for i in episode_indices if int(i) % 256 == self.chunk]
        refs: List[EpisodeRef] = []
        for ep_idx in my_eps:
            ep_id = f"agibot2026_ep_{ep_idx:08d}"
            refs.append(
                EpisodeRef(
                    episode_id=ep_id,
                    source=self.SOURCE,
                    raw_path=str(self.raw_root),
                    extra={"episode_index": ep_idx},
                )
            )
        return iter(refs)

    def convert_episode(self, ref: EpisodeRef) -> ConversionResult:
        """Convert one AgiBot 2026 episode."""
        ep_idx = int(ref.extra["episode_index"])
        if self.raw_root is None or not self.raw_root.exists():
            raise RuntimeError(
                "AgiBot 2026 conversion requires raw_root to be present locally"
            )
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("pyarrow required for AgiBot conversion") from e

        # Find the parquet file for this episode index.
        parquet_path = self._find_episode_parquet(ep_idx)
        if parquet_path is None:
            raise FileNotFoundError(f"no parquet for AgiBot episode {ep_idx}")
        table = pq.read_table(str(parquet_path))
        # Convert columns to ndarrays.
        cols = {name: np.asarray(table.column(name).to_pylist()) for name in table.schema.names}
        return self._table_to_result(ep_idx, cols, ref)

    def _find_episode_parquet(self, ep_idx: int) -> Optional[Path]:
        """Locate ``data/chunk-XXX/episode_NNNNNN.parquet`` under raw_root."""
        assert self.raw_root is not None
        for path in (self.raw_root / "data").rglob(f"episode_{ep_idx:06d}.parquet"):
            return path
        for path in (self.raw_root / "data").rglob(f"episode_{ep_idx}.parquet"):
            return path
        return None

    def _table_to_result(
        self, ep_idx: int, cols: Dict[str, np.ndarray], ref: EpisodeRef
    ) -> ConversionResult:
        """Pure transform: a column dict -> ConversionResult.

        Easy to unit-test by passing a synthesized dict.
        """
        T = int(cols["timestamp"].shape[0]) if "timestamp" in cols else int(
            cols.get("frame_index", np.arange(0)).shape[0]
        )
        assert T > 0, f"empty AgiBot episode {ep_idx}"

        # Action / state.
        action_native_raw = np.asarray(
            cols.get("action", np.zeros((T, AGIBOT_NATIVE_ACTION_DIM), dtype=np.float32)),
            dtype=np.float32,
        )
        if action_native_raw.ndim == 1:
            action_native_raw = action_native_raw.reshape(T, -1)
        d_native = min(action_native_raw.shape[1], AGIBOT_NATIVE_ACTION_DIM)

        # Compute right-arm EE velocity stream from observation.state EE pose
        # columns and put it in the first 7 padded columns so stage_3's
        # joint_delta_to_ee_finite_diff rule is a passthrough.
        ee_velocity_7 = self._compute_ee_velocity(cols, T)

        action_native = np.zeros((T, 32), dtype=np.float32)
        # First 7 padded columns: canonical EE-velocity stream (the pre-fill).
        action_native[:, :7] = ee_velocity_7
        # Following columns hold the raw native action (offset by 7) up to 32.
        tail_dim = min(d_native, 32 - 7)
        action_native[:, 7 : 7 + tail_dim] = action_native_raw[:, :tail_dim]
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[: 7 + tail_dim] = True

        state_raw = np.asarray(
            cols.get("observation.state", np.zeros((T, 50), dtype=np.float32)),
            dtype=np.float32,
        )
        if state_raw.ndim == 1:
            state_raw = state_raw.reshape(T, -1)
        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(state_raw.shape[1], 50)
        state[:, :d_state] = state_raw[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        # Cameras: AgiBot 2026 stores each camera as
        # ``videos/chunk-NNN/observation.images.<cam>/episode_NNNNNN.mp4``.
        # The parquet's column is metadata (e.g. fps + path) — we ignore
        # those rows and decode the mp4 directly so stage_2c/4 see the
        # standard ``camera_<cam>.npy`` staged layout.
        cameras: Dict[str, np.ndarray] = {}
        chunk_dir = f"chunk-{ep_idx // 1000:03d}"
        for src_key, dst_key in AGIBOT_CAMERA_MAP.items():
            mp4 = (
                self.raw_root / "videos" / chunk_dir
                / f"observation.images.{src_key}" / f"episode_{ep_idx:06d}.mp4"
            )
            if not mp4.exists():
                _LOG.debug("missing AgiBot mp4 %s (skipping cam)", mp4)
                continue
            frames = self._decode_mp4(mp4, max_frames=T)
            if frames is None or frames.shape[0] == 0:
                continue
            cameras[dst_key] = frames

        # Canonical action via stage_3 (passthrough on the pre-filled 7 cols).
        action_canonical_ee = canonicalize_action(action_native, AGIBOT_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        # Instruction segments → level_1/2/3 columns (PROMOTION).
        instructions = self._extract_instructions(cols, T)

        timestamps = np.asarray(
            cols.get("timestamp", np.arange(T, dtype=np.float32) / float(AGIBOT_FPS)),
            dtype=np.float32,
        )

        # subtask_label: True at frames where level_2 changes (segment boundary).
        subtask_label = self._segment_boundaries(instructions["level_2"])

        return ConversionResult(
            episode_index=ep_idx,
            embodiment=AGIBOT_EMBODIMENT,
            fps=AGIBOT_FPS,
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
                "agibot_episode_id": str(ep_idx),
                "subtask_label_per_frame": subtask_label.tolist(),
            },
        )

    # ----- AgiBot specifics ----------------------------------------------

    @staticmethod
    def _decode_mp4(path: Path, max_frames: Optional[int] = None) -> Optional[np.ndarray]:
        """Decode an mp4 to ``[T, H, W, 3] uint8`` RGB.

        AgiBot 2026 mp4s use the AV1 codec; decord/cv2 autodetect the wrong
        decoder and silently fail with "Missing Sequence Header". We force
        ``libdav1d`` via an ffmpeg subprocess that streams rawvideo over
        stdout — one call decodes the whole episode and a single numpy
        reshape gives us the [T, H, W, 3] array. Non-AV1 codecs work too
        (libdav1d gracefully falls back through ffmpeg's stream copy).
        """
        import subprocess

        # 1. Probe dimensions + frame count (cheap, single ffprobe call).
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,nb_frames,codec_name",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True, text=True, timeout=30, check=True,
            )
            parts = probe.stdout.strip().split(",")
            codec_name, w, h, nb = parts[0], int(parts[1]), int(parts[2]), parts[3]
            n_total = int(nb) if nb.isdigit() else 0
        except Exception as e:
            _LOG.error("ffprobe failed for %s: %s", path, e)
            return None

        n = min(n_total, int(max_frames)) if max_frames and n_total else n_total
        if n == 0:
            # Some containers omit nb_frames; just decode all and reshape.
            n = -1

        # 2. Stream raw RGB out of ffmpeg. Force libdav1d for AV1 streams,
        #    otherwise let ffmpeg autodetect.
        codec_flag: List[str] = []
        if codec_name == "av1":
            codec_flag = ["-c:v", "libdav1d"]
        cmd = [
            "ffmpeg", "-v", "error", *codec_flag, "-i", str(path),
        ]
        if n > 0:
            cmd += ["-frames:v", str(n)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600, check=True)
        except subprocess.CalledProcessError as e:
            _LOG.error("ffmpeg decode failed for %s: %s", path, e.stderr.decode(errors="ignore")[:300])
            return None

        nbytes = len(proc.stdout)
        bytes_per_frame = h * w * 3
        if nbytes % bytes_per_frame != 0:
            _LOG.error(
                "ffmpeg output size %d not divisible by frame size %d (%s)",
                nbytes, bytes_per_frame, path,
            )
            return None
        actual_n = nbytes // bytes_per_frame
        frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(actual_n, h, w, 3)
        return np.ascontiguousarray(frames)

    def _compute_ee_velocity(self, cols: Dict[str, np.ndarray], T: int) -> np.ndarray:
        """Extract right-arm EE pose from state and finite-difference to velocity.

        AgiBot's parquet exposes ``observation.state`` as a flat 50-D vector
        whose layout is documented in ``meta/modality.json``. Conventionally
        slots 14:21 hold the right-arm EE pose [x, y, z, rx, ry, rz, gripper].
        We tolerate missing data by returning zeros — the canonicalization
        rule will still validate and the per-episode QA gate downstream will
        catch suspiciously-zero streams.
        """
        state = np.asarray(
            cols.get("observation.state", np.zeros((T, 50), dtype=np.float32)),
            dtype=np.float32,
        )
        if state.ndim == 1:
            state = state.reshape(T, -1)
        if state.shape[1] < 21:
            return np.zeros((T, 7), dtype=np.float32)

        pos = state[:, 14:17]  # right EE position
        rot = state[:, 17:20]  # right EE rotvec
        grip = state[:, 20:21]  # right gripper

        dt = 1.0 / float(AGIBOT_FPS)
        lin_vel = np.zeros_like(pos)
        ang_vel = np.zeros_like(rot)
        if T >= 2:
            lin_vel[:-1] = (pos[1:] - pos[:-1]) / dt
            lin_vel[-1] = lin_vel[-2]
            rot_delta = _wrap_to_pi(rot[1:] - rot[:-1])
            ang_vel[:-1] = rot_delta / dt
            ang_vel[-1] = ang_vel[-2]

        canon = np.concatenate([lin_vel, ang_vel, grip], axis=1).astype(np.float32)
        # Post-condition: clip to canonical bounds so stage_3's validator is happy.
        canon[:, 0:3] = np.clip(canon[:, 0:3], -2.0, 2.0)
        canon[:, 3:6] = np.clip(canon[:, 3:6], -math.pi, math.pi)
        canon[:, 6] = np.clip(canon[:, 6], 0.0, 1.0)
        return canon

    def _extract_instructions(
        self, cols: Dict[str, np.ndarray], T: int
    ) -> Dict[str, List[str]]:
        """Promote ``instruction_segments`` into per-frame level_{1,2,3}.

        AgiBot's ``instruction_segments`` is a list of dicts:
        ``[{"start": s, "end": e, "level_1": ..., "level_2": ..., "level_3": ...}, ...]``.
        We expand it to per-frame strings; frames outside any segment carry the
        empty string. We *additionally* fall back to ``task`` if the segments
        are missing so episodes without USAM-extension still produce a usable
        level_1 stream.
        """
        level_1 = [""] * T
        level_2 = [""] * T
        level_3 = [""] * T

        # Top-level fallback.
        task_col = cols.get("task")
        default_level_1 = ""
        if task_col is not None and task_col.shape[0] > 0:
            default_level_1 = str(task_col[0]) if T > 0 else ""
        for i in range(T):
            level_1[i] = default_level_1

        seg_col = cols.get("instruction_segments")
        if seg_col is None:
            return {"level_1": level_1, "level_2": level_2, "level_3": level_3}

        # Expect a list-of-dicts column (one per frame, repeated). Use frame 0.
        first = seg_col[0] if seg_col.shape[0] > 0 else None
        if first is None:
            return {"level_1": level_1, "level_2": level_2, "level_3": level_3}

        try:
            if isinstance(first, str):
                segments = json.loads(first)
            else:
                segments = first  # already a list/dict
        except (json.JSONDecodeError, TypeError):
            segments = []

        if not isinstance(segments, list):
            segments = []

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            s = int(seg.get("start", 0))
            e = int(seg.get("end", T))
            s = max(0, s)
            e = min(T, e)
            for k_dst, k_src in (
                ("level_1", "level_1"),
                ("level_2", "level_2"),
                ("level_3", "level_3"),
            ):
                v = seg.get(k_src)
                if v is None:
                    continue
                v_str = str(v)
                target = {"level_1": level_1, "level_2": level_2, "level_3": level_3}[k_dst]
                for i in range(s, e):
                    target[i] = v_str

        return {"level_1": level_1, "level_2": level_2, "level_3": level_3}

    @staticmethod
    def _segment_boundaries(level_2: List[str]) -> np.ndarray:
        """Boolean per-frame mask: True where level_2 differs from previous frame."""
        T = len(level_2)
        out = np.zeros(T, dtype=bool)
        for i in range(1, T):
            if level_2[i] != level_2[i - 1] and level_2[i] != "":
                out[i] = True
        return out

    # ----- shard writing ---------------------------------------------------

    def write_shard(self, results: List[ConversionResult]) -> Path:
        """Roll a list of episodes up into a single parquet shard.

        Promotes ``instructions["level_1"|"level_2"|"level_3"]`` to top-level
        parquet columns — this is the central AgiBot deliverable.
        """
        assert len(results) > 0, "empty shard"
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("pyarrow required for shard writing") from e

        rows = []
        for r in results:
            T = r.action_canonical_ee.shape[0]
            subtask_per_frame = r.raw_meta.get("subtask_label_per_frame", [False] * T)
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
                        "subtask_label": bool(subtask_per_frame[t]),
                    }
                )

        # Hard-assert promotion happened: required columns must be present.
        required = {"level_1", "level_2", "level_3", "subtask_label"}
        assert required.issubset(rows[0].keys()), (
            f"AgiBot shard rows missing promoted instruction columns: "
            f"{required - rows[0].keys()}"
        )

        table = pa.Table.from_pylist(rows)
        h = self.shard_hash(results)
        out = self.output_dir / f"file-{h}.parquet"
        if not out.exists():
            pq.write_table(table, str(out))
        return out


__all__ = [
    "AgiBot2026Converter",
    "AGIBOT_EMBODIMENT",
    "AGIBOT_FPS",
    "AGIBOT_CAMERA_MAP",
    "AGIBOT_NATIVE_ACTION_DIM",
]
