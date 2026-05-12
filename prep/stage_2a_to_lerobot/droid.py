# SPDX-License-Identifier: MIT
"""DROID RLDS -> USAM-LeRobot v2.1 converter.

Source: ``gs://gresearch/robotics/droid`` via ``tfds.load("droid", ...)``.
Reads a cleaner language-instruction set from the ``KarlP/droid`` HF dataset
when present and falls back to the RLDS ``language_instruction`` field.

This module *describes* downloads but never executes them — see hard rule in
``docs/AGENT_CHARTER.md``. The TFDS / HF reads happen only inside
``convert_episode`` when the dispatcher actually runs the converter.

Phase 1 scope: produce a single chunk's worth of USAM-LeRobot v2.1 shards
(parquet metadata + .mp4 stubs are ok as placeholders) for the DROID source.
The actual MP4 encoding step is shared with stages 2b/2c and lives in
``prep/_video.py`` (pipeline-engineer's file, imported here once it exists).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from prep._base import CheckpointedJob, EpisodeRef  # type: ignore[import-not-found]
from prep.stage_3_canonical import canonicalize_action, validate_action_canonical


_LOG = logging.getLogger(__name__)

# The 7-D DROID action layout: cartesian linear velocity (3) + angular
# velocity (3) + gripper (1). See droid_dataset_builder.py upstream.
DROID_ACTION_DIM: int = 7
DROID_EMBODIMENT: str = "droid_franka"
DROID_FPS: int = 15
# DROID's RLDS exposes two ``exterior_image_*`` and ``wrist_image_*`` keys.
# We map to USAM's canonical camera names. USAM follows the LDA-1B
# convention of using only the EGOCENTRIC head view for world-model
# training; wrist-cam adds redundant information that hurts the world
# model's spatial coherence (the wrist view moves with the gripper and
# breaks the world-frame stationarity DINOv3 expects). To re-enable
# wrist for downstream tasks that benefit (e.g., contact-rich
# manipulation policies), add ``"wrist_image_left": "wrist_rgb"`` here
# and bump ``num_views`` to 2 in the model config.
DROID_CAMERA_MAP = {
    "exterior_image_1_left": "head_rgb",
}


@dataclass
class ConversionResult:
    """The unified internal record produced by every Stage-2a converter.

    See ``docs/IMPLEMENTATION_PLAN.md §5.2``.
    """

    episode_index: int
    embodiment: str
    fps: int
    cameras: dict  # canonical_key -> np.ndarray [T, H, W, 3] uint8
    depth: dict  # canonical_key -> np.ndarray [T, H, W] uint16
    state: np.ndarray  # [T, 50]
    state_mask: np.ndarray  # [50]
    action_native: np.ndarray  # [T, 32]
    action_mask: np.ndarray  # [32]
    action_canonical_ee: np.ndarray  # [T, 7]
    instructions: dict  # level_1 / level_2 / level_3
    force_torque: Optional[np.ndarray]
    timestamps: np.ndarray
    raw_meta: dict = field(default_factory=dict)


def episode_filename_hash(episode_index: int, source: str, version: str) -> str:
    """Stable per-episode hash for idempotent filenames.

    Used by ``write_shard`` so re-runs never duplicate work and so we can detect
    bit-rotted shards by content rather than mtime.
    """
    h = hashlib.sha1(f"{source}:{version}:{episode_index}".encode()).hexdigest()
    return h[:12]


class DroidConverter(CheckpointedJob):
    """RLDS DROID -> USAM-LeRobot v2.1 converter.

    Parameters
    ----------
    chunk : int
        Chunk id assigned by ``prep.dispatch``. Each chunk holds ~256 episodes.
    output_root : Path
        ``/scratch/usam/droid/2a/chunk-XXX/``.
    rlds_data_dir : str
        TFDS data dir (e.g. ``"gs://gresearch/robotics"`` or a local mirror).
    karlp_droid_root : Path | None
        Optional snapshot dir of ``KarlP/droid`` for cleaner language. If
        ``None`` we silently fall back to the RLDS ``language_instruction``.
    version : str
        Schema version stamp baked into per-episode filename hashes.
    """

    SOURCE: str = "droid"
    STAGE: str = "2a"

    def __init__(
        self,
        chunk: int,
        output_root: Path,
        rlds_data_dir: str = "gs://gresearch/robotics",
        karlp_droid_root: Optional[Path] = None,
        version: str = "v0.1",
    ) -> None:
        super().__init__(
            source=self.SOURCE,
            stage=self.STAGE,
            chunk=chunk,
            output_root=Path(output_root),
        )
        # Parent already validated chunk + set self.output_root / self.output_dir.
        self.rlds_data_dir = rlds_data_dir
        self.karlp_droid_root = Path(karlp_droid_root) if karlp_droid_root else None
        self.version = version

        self._karlp_lookup: Optional[dict[str, str]] = None  # episode_id -> instruction

    # ----- KarlP/droid override --------------------------------------------

    def _load_karlp_lookup(self) -> dict[str, str]:
        """Lazy-load the KarlP/droid cleaner language map."""
        if self._karlp_lookup is not None:
            return self._karlp_lookup
        if self.karlp_droid_root is None or not self.karlp_droid_root.exists():
            self._karlp_lookup = {}
            return self._karlp_lookup
        cleaned: dict[str, str] = {}
        for jsonl in self.karlp_droid_root.rglob("*.jsonl"):
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep_id = str(row.get("episode_id") or row.get("id") or "")
                txt = str(row.get("language_instruction") or row.get("instruction") or "")
                if ep_id and txt:
                    cleaned[ep_id] = txt
        self._karlp_lookup = cleaned
        return cleaned

    def _resolve_instruction(self, episode_id: str, rlds_instruction: str) -> str:
        """Prefer KarlP/droid cleaned annotations; fall back to RLDS."""
        cleaned = self._load_karlp_lookup().get(episode_id, "")
        return cleaned or rlds_instruction

    # ----- CheckpointedJob hooks ------------------------------------------

    def list_episodes(self) -> Iterator[EpisodeRef]:
        """Yield one ``EpisodeRef`` per DROID episode assigned to this chunk.

        We delegate the RLDS sharding to ``prep._base.shard_assignment`` (which
        pipeline-engineer owns). When invoked from a unit test the iterator may
        be empty if the RLDS data is not present locally.
        """
        try:
            import tensorflow_datasets as tfds  # type: ignore
        except ImportError:
            _LOG.warning("tensorflow_datasets not installed; DROID enumeration skipped")
            return iter([])

        builder = tfds.builder("droid", data_dir=self.rlds_data_dir)
        # ``builder.info.splits["train"].num_examples`` partitioned across chunks
        n_total = int(builder.info.splits["train"].num_examples)
        from prep._base import shard_assignment  # type: ignore[import-not-found]

        my_indices = shard_assignment(n_total=n_total, chunk=self.chunk)
        for ep_idx in my_indices:
            # Canonical EpisodeRef schema (matches bridge / rh20t / robomind):
            # episode_id is a string; the integer index is recovered via int().
            yield EpisodeRef(
                episode_id=str(int(ep_idx)),
                source=self.SOURCE,
                raw_path=self.rlds_data_dir,
            )

    def _ep_shard_hash(self, ep: EpisodeRef) -> str:
        """Return the deterministic 12-char shard hash for one episode."""
        return episode_filename_hash(int(ep.episode_id), self.SOURCE, self.version)

    def is_done(self, ep: EpisodeRef) -> bool:
        marker = self.output_root / f"ep_{self._ep_shard_hash(ep)}.done"
        return marker.exists()

    def process(self, ep: EpisodeRef) -> None:
        result = self.convert_episode(ep)
        if result is None:
            return
        self._stage_episode(result)
        marker = self.output_root / f"ep_{self._ep_shard_hash(ep)}.done"
        marker.write_text(json.dumps({"episode_index": int(ep.episode_id)}))

    # ----- conversion ------------------------------------------------------

    def convert_episode(self, ep: EpisodeRef) -> Optional[ConversionResult]:
        """Convert one DROID episode to a ``ConversionResult``.

        Returns
        -------
        ConversionResult | None
            ``None`` if the episode is malformed (logged + skipped).
        """
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_datasets as tfds  # type: ignore
        except ImportError:
            raise RuntimeError(
                "DROID conversion requires tensorflow + tensorflow_datasets at runtime"
            )

        ep_index = int(ep.episode_id)
        builder = tfds.builder("droid", data_dir=self.rlds_data_dir)
        ds = builder.as_dataset(split=f"train[{ep_index}:{ep_index + 1}]")
        for tf_episode in ds.take(1):
            return self._tf_episode_to_result(tf_episode, ep)
        return None

    def _tf_episode_to_result(self, tf_episode, ep: EpisodeRef) -> ConversionResult:
        """Pure data-shaping. No I/O. Easy to unit-test if a dummy is passed in."""
        ep_index = int(ep.episode_id)
        steps = list(tf_episode["steps"].as_numpy_iterator())
        T = len(steps)
        assert T > 0, f"empty DROID episode {ep_index}"

        # Cameras
        cameras: dict[str, np.ndarray] = {}
        for src_key, dst_key in DROID_CAMERA_MAP.items():
            frames = []
            for s in steps:
                obs = s["observation"]
                if src_key in obs:
                    frames.append(np.asarray(obs[src_key], dtype=np.uint8))
            if frames:
                cameras[dst_key] = np.stack(frames, axis=0)

        # Action / state — DROID is already 7-D EE-velocity + grip in [0, 1].
        action_native_raw = np.stack(
            [np.asarray(s["action"], dtype=np.float32) for s in steps], axis=0
        )
        # Ensure exactly DROID_ACTION_DIM columns
        if action_native_raw.shape[1] > DROID_ACTION_DIM:
            action_native_raw = action_native_raw[:, :DROID_ACTION_DIM]
        action_native = np.zeros((T, 32), dtype=np.float32)
        action_native[:, :DROID_ACTION_DIM] = action_native_raw
        action_mask = np.zeros((32,), dtype=bool)
        action_mask[:DROID_ACTION_DIM] = True

        # DROID 1.0.1 RLDS exposes proprio as three separate keys; we
        # concatenate into a single robot_state vector of length 14:
        #   [joint_position(7), gripper_position(1), cartesian_position(6)]
        # Earlier in development the converter assumed a flat "robot_state"
        # key, which doesn't exist in the actual schema.
        joint_pos = np.stack(
            [np.asarray(s["observation"]["joint_position"], dtype=np.float32) for s in steps],
            axis=0,
        )  # (T, 7)
        gripper_pos = np.stack(
            [np.asarray(s["observation"]["gripper_position"], dtype=np.float32) for s in steps],
            axis=0,
        )  # (T, 1)
        cart_pos = np.stack(
            [np.asarray(s["observation"]["cartesian_position"], dtype=np.float32) for s in steps],
            axis=0,
        )  # (T, 6)
        state_raw = np.concatenate([joint_pos, gripper_pos, cart_pos], axis=1)  # (T, 14)
        state = np.zeros((T, 50), dtype=np.float32)
        d_state = min(state_raw.shape[1], 50)
        state[:, :d_state] = state_raw[:, :d_state]
        state_mask = np.zeros((50,), dtype=bool)
        state_mask[:d_state] = True

        # Canonical action via the shared canonicalization stage.
        action_canonical_ee = canonicalize_action(action_native_raw, DROID_EMBODIMENT)
        validate_action_canonical(action_canonical_ee)

        # Language: prefer KarlP cleaner annotation
        rlds_instr = ""
        if "language_instruction" in steps[0]:
            rlds_instr = bytes(steps[0]["language_instruction"]).decode("utf-8", errors="ignore")
        cleaned_instr = self._resolve_instruction(ep.episode_id, rlds_instr)
        instructions = {
            "level_1": [cleaned_instr] * T,  # high-level goal
            "level_2": [""] * T,  # not provided by DROID
            "level_3": [""] * T,
        }

        timestamps = np.arange(T, dtype=np.float32) / float(DROID_FPS)

        return ConversionResult(
            episode_index=ep_index,
            embodiment=DROID_EMBODIMENT,
            fps=DROID_FPS,
            cameras=cameras,
            depth={},  # depth is computed in stage_2c
            state=state,
            state_mask=state_mask,
            action_native=action_native,
            action_mask=action_mask,
            action_canonical_ee=action_canonical_ee,
            instructions=instructions,
            force_torque=None,
            timestamps=timestamps,
            raw_meta={"droid_episode_id": ep.episode_id, "rlds_instruction": rlds_instr},
        )

    # ----- shard writing ---------------------------------------------------

    def _stage_episode(self, result: ConversionResult) -> None:
        """Write per-episode JSON + npy files into the chunk staging dir.

        Stage-2a writes raw-ish staging artifacts; the final parquet+mp4 shard
        roll-up happens in :meth:`write_shard`. We separate the two so a partial
        chunk can be resumed without re-reading RLDS.
        """
        ep_dir = self.output_root / f"ep_{episode_filename_hash(result.episode_index, self.SOURCE, self.version)}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "action_native.npy", result.action_native)
        np.save(ep_dir / "action_canonical_ee.npy", result.action_canonical_ee)
        np.save(ep_dir / "state.npy", result.state)
        np.save(ep_dir / "timestamps.npy", result.timestamps)
        # Cameras are large — write them as raw .npy here; the encoder stage
        # consumes these files directly to produce mp4s.
        for cam, arr in result.cameras.items():
            np.save(ep_dir / f"camera_{cam}.npy", arr)
        meta = {
            "episode_index": result.episode_index,
            "embodiment": result.embodiment,
            "fps": result.fps,
            "instructions": result.instructions,
            "action_mask": result.action_mask.tolist(),
            "state_mask": result.state_mask.tolist(),
            "raw_meta": result.raw_meta,
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta))

    def write_shard(self, results: List[ConversionResult]) -> Path:
        """Roll a list of episodes up into a single parquet shard.

        Returns the path of the written ``data/chunk-XXX/file-YYY.parquet``.
        Idempotent: if the destination exists with the same hash list, it is
        left untouched.
        """
        assert len(results) > 0, "empty shard"
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:  # pragma: no cover
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
                        "subtask_label": False,  # DROID has no segment ground truth
                    }
                )

        table = pa.Table.from_pylist(rows)
        chunk_dir = self.output_root / "data" / f"chunk-{self.chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        # Use a content-derived file id so re-runs are bit-identical.
        h = hashlib.sha1(json.dumps(sorted(r.episode_index for r in results)).encode()).hexdigest()[:6]
        out = chunk_dir / f"file-{h}.parquet"
        if not out.exists():
            pq.write_table(table, str(out))
        return out


__all__ = [
    "ConversionResult",
    "DroidConverter",
    "DROID_ACTION_DIM",
    "DROID_CAMERA_MAP",
    "DROID_EMBODIMENT",
    "DROID_FPS",
    "episode_filename_hash",
]
