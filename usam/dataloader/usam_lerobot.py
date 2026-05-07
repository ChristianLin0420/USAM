# SPDX-License-Identifier: MIT
"""Runtime dataloader for USAM-LeRobot v2.1 datasets.

USAM extends the LeRobot v2.1 layout with a per-modality fp16 DINO feature
cache (``features/{rgb,depth,flow}/chunk-XXX/file-YYY.safetensors``). At
training time we always read from the cache and never decode MP4s — see
``docs/IMPLEMENTATION_PLAN.md §11.10``.

The streaming-decode branch (``use_cached_features=False``) exists only for
smoke tests and developer sanity checks; it never runs in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from usam.dataloader.feature_cache import FeatureCache


# ----- on-disk schema constants ------------------------------------------------
TASK_ID_BY_NAME: Dict[str, int] = {
    "Policy": 0,
    "FDM": 1,
    "IDM": 2,
    "VisFcst": 3,
}
DEFAULT_MODALITIES: Tuple[str, ...] = ("rgb", "depth", "flow")
DEFAULT_CAMERAS: Tuple[str, ...] = ("head", "wrist")


@dataclass
class _EpisodeMeta:
    """Internal cache of one episode's metadata.

    Attributes
    ----------
    episode_index : int
    length : int
        Number of native-fps action frames.
    chunk : int
    file : int
        Parquet shard id.
    embodiment : str
    """

    episode_index: int
    length: int
    chunk: int
    file: int
    embodiment: str


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing meta/info.json in {root}")
    return json.loads(info_path.read_text())


def _load_episode_index(root: Path) -> List[_EpisodeMeta]:
    """Read ``meta/episodes.parquet`` (or .jsonl fallback).

    Returns a list of episode metadata in episode-index order.
    """
    eps_parquet = root / "meta" / "episodes.parquet"
    eps_jsonl = root / "meta" / "episodes.jsonl"
    rows: List[dict]
    if eps_parquet.exists():
        try:
            import pyarrow.parquet as pq  # type: ignore

            tbl = pq.read_table(str(eps_parquet))
            rows = tbl.to_pylist()
        except Exception:  # pragma: no cover - parquet is preferred but optional
            import pandas as pd  # type: ignore

            rows = pd.read_parquet(eps_parquet).to_dict(orient="records")
    elif eps_jsonl.exists():
        rows = [json.loads(line) for line in eps_jsonl.read_text().splitlines() if line.strip()]
    else:
        raise FileNotFoundError(f"missing meta/episodes.{{parquet,jsonl}} in {root}")

    out: List[_EpisodeMeta] = []
    for r in rows:
        out.append(
            _EpisodeMeta(
                episode_index=int(r["episode_index"]),
                length=int(r["length"]),
                chunk=int(r.get("chunk", 0)),
                file=int(r.get("file", 0)),
                embodiment=str(r.get("embodiment", "unknown")),
            )
        )
    out.sort(key=lambda m: m.episode_index)
    return out


def _load_episode_frames(root: Path, ep: _EpisodeMeta) -> Dict[str, np.ndarray]:
    """Read the parquet shard for ``ep`` and return a per-column dict of arrays.

    The on-disk schema mirrors the unified record from §5.2:
        - ``proprio`` ``[T, D_state_padded=50]`` fp32
        - ``state_mask`` ``[50]`` bool (replicated per row)
        - ``action_native`` ``[T, 32]`` fp32
        - ``action_mask`` ``[32]`` bool
        - ``action_canonical_ee`` ``[T, 7]`` fp32
        - ``timestamps`` ``[T]`` fp32
        - ``level_1`` / ``level_2`` / ``level_3`` ``[T]`` str (subtask labels)
        - ``subtask_label`` ``[T]`` bool
    """
    shard = root / "data" / f"chunk-{ep.chunk:03d}" / f"file-{ep.file:03d}.parquet"
    if not shard.exists():
        raise FileNotFoundError(f"missing parquet shard {shard}")

    try:
        import pyarrow.parquet as pq  # type: ignore

        tbl = pq.read_table(str(shard))
        df_dict = {col: tbl.column(col).to_pylist() for col in tbl.column_names}
    except Exception:  # pragma: no cover
        import pandas as pd  # type: ignore

        df = pd.read_parquet(shard)
        df_dict = {col: df[col].tolist() for col in df.columns}

    # Each row may contain multiple episodes; filter by episode_index.
    if "episode_index" in df_dict:
        keep = [i for i, e in enumerate(df_dict["episode_index"]) if int(e) == ep.episode_index]
        df_dict = {col: [vals[i] for i in keep] for col, vals in df_dict.items()}
    return _vectorize_columns(df_dict)


def _vectorize_columns(df_dict: Dict[str, list]) -> Dict[str, np.ndarray]:
    """Convert per-row python lists into stacked numpy arrays where possible."""
    out: Dict[str, np.ndarray] = {}
    for k, vals in df_dict.items():
        if len(vals) == 0:
            out[k] = np.asarray(vals)
            continue
        first = vals[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            try:
                out[k] = np.asarray(vals, dtype=np.float32)
            except (TypeError, ValueError):
                out[k] = np.asarray(vals, dtype=object)
        elif isinstance(first, (bool, np.bool_)):
            out[k] = np.asarray(vals, dtype=bool)
        elif isinstance(first, (int, np.integer)):
            out[k] = np.asarray(vals, dtype=np.int64)
        elif isinstance(first, (float, np.floating)):
            out[k] = np.asarray(vals, dtype=np.float32)
        else:
            out[k] = np.asarray(vals, dtype=object)
    return out


# ----- public dataset ---------------------------------------------------------

class USAMLeRobotDataset(Dataset):
    """USAM-LeRobot v2.1 dataset with cached fp16 DINO features by default.

    Parameters
    ----------
    repo_id_or_path : str | Path
        Either an HF ``"<org>/usam-<source>"`` id or a local directory laid
        out per ``docs/IMPLEMENTATION_PLAN.md §5.1``. We never download here:
        if the path is non-local, the caller must have materialized it via
        ``huggingface_hub.snapshot_download`` upstream.
    split : str
        ``"train"`` or ``"val"``. Currently a no-op selector: episode splits
        are encoded directly in ``meta/episodes.parquet`` via the optional
        ``split`` column. Phase 1 keeps everything as ``"train"``.
    use_cached_features : bool
        If True (default), read fp16 DINO features from ``features/<modality>/``.
        If False, raise NotImplementedError unless a streaming hook is set.
        Streaming-decode is a fallback only.
    modalities : list[str]
        Which modalities to read per camera. Subset of ``{"rgb","depth","flow"}``.
    cameras : list[str]
        Which canonical camera keys to read. Subset of the on-disk feature dirs.
    history_frames : int
        Number of past feature-fps frames to include (T in §4.2).
    future_frames : int
        Number of future feature-fps frames for visual forecasting.
    action_chunk : int
        Number of native-fps action steps per sample (e.g. 16 for 30 Hz).
    fps_features : int
        Native cache fps (default 5).
    fps_action : int
        Native action fps (default 30).

    Returns
    -------
    Each ``__getitem__`` produces a dict matching §11.10's interface.
    """

    def __init__(
        self,
        repo_id_or_path: str | Path,
        split: str = "train",
        use_cached_features: bool = True,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        history_frames: int = 4,
        future_frames: int = 8,
        action_chunk: int = 16,
        fps_features: int = 5,
        fps_action: int = 30,
        streaming_encoder: Optional[object] = None,
    ) -> None:
        assert split in {"train", "val"}, split
        assert history_frames >= 1
        assert future_frames >= 0
        assert action_chunk >= 1
        assert fps_features > 0 and fps_action > 0
        assert all(m in {"rgb", "depth", "flow"} for m in modalities)
        assert len(cameras) >= 1

        self.root = Path(repo_id_or_path)
        if not self.root.exists():
            raise FileNotFoundError(
                f"USAM-LeRobot path does not exist: {self.root}. "
                "If this is an HF repo id, snapshot_download it first."
            )
        self.split = split
        self.use_cached_features = bool(use_cached_features)
        self.modalities = list(modalities)
        self.cameras = list(cameras)
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        self.action_chunk = int(action_chunk)
        self.fps_features = int(fps_features)
        self.fps_action = int(fps_action)
        self._streaming_encoder = streaming_encoder

        self.info = _load_info(self.root)
        self.episodes: List[_EpisodeMeta] = _load_episode_index(self.root)
        assert len(self.episodes) > 0, f"no episodes in {self.root}"

        # Open one feature cache per (camera, modality) combination.
        self._feature_caches: Dict[Tuple[str, str], FeatureCache] = {}
        if self.use_cached_features:
            for cam in self.cameras:
                for mod in self.modalities:
                    cache_root = self.root / "features" / cam
                    if not cache_root.exists():
                        # Allow datasets that only have a single un-prefixed
                        # cache (e.g. tiny test fixtures).
                        cache_root = self.root / "features"
                    self._feature_caches[(cam, mod)] = FeatureCache(cache_root, mod)

        # Build a flat (episode_idx, t_native) index so __len__ is the number
        # of *samples*, not episodes. Window width covers history+future on the
        # feature axis plus the action chunk on the action axis.
        self._sample_index: List[Tuple[int, int]] = []
        stride_action = max(self.fps_action // self.fps_features, 1)
        feat_window = self.history_frames + self.future_frames
        for ep in self.episodes:
            min_native = (self.history_frames - 1) * stride_action
            max_native = ep.length - max(self.action_chunk, self.future_frames * stride_action)
            for t in range(min_native, max(min_native + 1, max_native), stride_action):
                self._sample_index.append((ep.episode_index, t))
        if len(self._sample_index) == 0:
            # Episodes too short for the configured window — fall back to one
            # sample per episode at t=0 (used by tiny test fixtures).
            for ep in self.episodes:
                self._sample_index.append((ep.episode_index, 0))

        # Cache parquet frames lazily; tiny fixtures fit in RAM.
        self._frame_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._ep_by_idx: Dict[int, _EpisodeMeta] = {ep.episode_index: ep for ep in self.episodes}

    # ----- helpers ---------------------------------------------------------

    def _frames_for(self, episode_index: int) -> Dict[str, np.ndarray]:
        if episode_index not in self._frame_cache:
            ep = self._ep_by_idx[episode_index]
            self._frame_cache[episode_index] = _load_episode_frames(self.root, ep)
        return self._frame_cache[episode_index]

    def _feature_window(
        self, ep_idx: int, native_t: int, ep_length: int
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Return ``(camera, modality) -> [T_total, N, D]`` for the window."""
        stride = max(self.fps_action // self.fps_features, 1)
        start = native_t - (self.history_frames - 1) * stride
        feat_idx_native = list(
            range(start, start + (self.history_frames + self.future_frames) * stride, stride)
        )

        out: Dict[Tuple[str, str], torch.Tensor] = {}
        for cam in self.cameras:
            for mod in self.modalities:
                cache = self._feature_caches.get((cam, mod))
                if cache is None or not cache.has(ep_idx):
                    continue
                # Clamp to the cache's actual frame count for safety. We
                # assume the cache stores a frame every `stride` native frames
                # starting from frame 0.
                cache_t = cache.num_frames(ep_idx)
                feat_indices = torch.tensor(
                    [max(0, min(i // stride, cache_t - 1)) for i in feat_idx_native],
                    dtype=torch.long,
                )
                out[(cam, mod)] = cache.get(ep_idx, feat_indices)
        return out

    def _streaming_window(
        self, ep_idx: int, native_t: int, ep_length: int
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Fallback: decode MP4 + run the encoder. Smoke-test only."""
        if self._streaming_encoder is None:
            raise NotImplementedError(
                "use_cached_features=False requires passing streaming_encoder=TriDinoTower(...)"
            )
        # We don't import decord at module load: keep it lazy so tests on
        # tiny fixtures don't need it installed.
        raise NotImplementedError(
            "Streaming decode is intentionally a smoke-test fallback and is not "
            "wired in Phase 1. Use use_cached_features=True."
        )

    # ----- dataset interface ----------------------------------------------

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int | float | bool]:
        assert 0 <= idx < len(self._sample_index), idx
        ep_idx, native_t = self._sample_index[idx]
        ep = self._ep_by_idx[ep_idx]
        frames = self._frames_for(ep_idx)

        # Action chunk over the next `action_chunk` native frames.
        a_start = native_t
        a_end = min(native_t + self.action_chunk, ep.length)
        action_canonical = np.zeros((self.action_chunk, 7), dtype=np.float32)
        action_native = np.zeros((self.action_chunk, 32), dtype=np.float32)
        if "action_canonical_ee" in frames and len(frames["action_canonical_ee"]) > 0:
            real = np.asarray(frames["action_canonical_ee"][a_start:a_end], dtype=np.float32)
            action_canonical[: real.shape[0]] = real
        if "action_native" in frames and len(frames["action_native"]) > 0:
            real_n = np.asarray(frames["action_native"][a_start:a_end], dtype=np.float32)
            # tolerate per-source action dim < 32: zero-pad on the right
            d = min(real_n.shape[1], 32) if real_n.ndim == 2 else 0
            if d > 0:
                action_native[: real_n.shape[0], :d] = real_n[:, :d]

        proprio = np.zeros((50,), dtype=np.float32)
        if "proprio" in frames and len(frames["proprio"]) > 0:
            row = np.asarray(frames["proprio"][native_t], dtype=np.float32)
            d = min(row.shape[0], 50)
            proprio[:d] = row[:d]

        action_mask = np.zeros((32,), dtype=bool)
        if "action_mask" in frames and len(frames["action_mask"]) > 0:
            am = np.asarray(frames["action_mask"][0], dtype=bool)
            action_mask[: am.shape[0]] = am[:32]
        state_mask = np.zeros((50,), dtype=bool)
        if "state_mask" in frames and len(frames["state_mask"]) > 0:
            sm = np.asarray(frames["state_mask"][0], dtype=bool)
            state_mask[: sm.shape[0]] = sm[:50]

        instruction = ""
        if "level_1" in frames and len(frames["level_1"]) > native_t:
            instruction = str(frames["level_1"][native_t])
        elif "instruction" in frames and len(frames["instruction"]) > native_t:
            instruction = str(frames["instruction"][native_t])

        subtask_label = False
        if "subtask_label" in frames and len(frames["subtask_label"]) > native_t:
            subtask_label = bool(frames["subtask_label"][native_t])

        # Feature window
        if self.use_cached_features:
            feats = self._feature_window(ep_idx, native_t, ep.length)
        else:
            feats = self._streaming_window(ep_idx, native_t, ep.length)

        sample: Dict[str, torch.Tensor | str | int | float | bool] = {
            "proprio": torch.from_numpy(proprio),
            "state_mask": torch.from_numpy(state_mask),
            "action_chunk": torch.from_numpy(action_canonical),
            "action_native": torch.from_numpy(action_native),
            "action_mask": torch.from_numpy(action_mask),
            "instruction": instruction,
            "task_id": int(TASK_ID_BY_NAME["Policy"]),
            "noise_level": 0.0,
            "subtask_label": bool(subtask_label),
            "episode_index": int(ep_idx),
            "embodiment": ep.embodiment,
        }

        # Concatenate per-camera modality features into the names §4.2 expects.
        for mod in self.modalities:
            stacked: List[torch.Tensor] = []
            for cam in self.cameras:
                t = feats.get((cam, mod))
                if t is not None:
                    stacked.append(t)
            if len(stacked) == 0:
                continue
            seq = stacked[0] if len(stacked) == 1 else torch.cat(stacked, dim=1)
            sample[f"{mod}_dino_seq"] = seq

        # head_keyframe_rgb_dino = first head-camera RGB frame in the window
        head_rgb = feats.get((self.cameras[0], "rgb")) if self.use_cached_features else None
        if head_rgb is not None and head_rgb.numel() > 0:
            sample["head_keyframe_rgb_dino"] = head_rgb[0]
        return sample
