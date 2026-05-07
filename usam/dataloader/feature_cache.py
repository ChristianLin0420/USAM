# SPDX-License-Identifier: MIT
"""Memory-mapped fp16 DINO feature cache reader.

Each shard on disk is a `safetensors` file containing one tensor per episode
under the key ``"ep_{episode_index:08d}"`` with shape ``[T_features, N_tokens, D]``
and dtype fp16. We open the shard with ``safetensors.safe_open`` in the
``"pt"`` framework on the ``"cpu"`` device, which gives true mmap semantics:
the kernel page cache is shared across worker processes, so two DataLoader
workers reading the same shard never duplicate it in RAM.

The on-disk index is a tiny JSON next to the shard:

    file-000.safetensors            ← N tensors, one per episode
    file-000.safetensors.index.json ← {"ep_00000001": {"chunk": 0, "file": 0}}

We don't need to deserialize the index here — `safe_open(...).keys()` already
exposes the per-shard episode list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open


def _episode_key(episode_index: int) -> str:
    """Return the canonical safetensors tensor key for an episode."""
    assert isinstance(episode_index, int) and episode_index >= 0
    return f"ep_{episode_index:08d}"


def write_feature_shard(
    shard_path: Path,
    episode_features: Dict[int, torch.Tensor],
) -> None:
    """Write a feature shard in the layout expected by ``FeatureCache``.

    Parameters
    ----------
    shard_path : Path
        Output ``.safetensors`` file. Parent directory is created if missing.
    episode_features : dict[int, torch.Tensor]
        Mapping ``episode_index -> tensor[T, N, D]``, dtype fp16.

    Notes
    -----
    Used by ``prep/stage_4_dino_cache.py`` and the test fixture builder. We
    keep this helper here so writers and readers agree on the key scheme.
    """
    assert isinstance(shard_path, Path)
    assert isinstance(episode_features, dict) and len(episode_features) > 0

    from safetensors.torch import save_file

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, torch.Tensor] = {}
    index: Dict[str, Dict[str, int]] = {}
    for ep_idx, tensor in episode_features.items():
        assert tensor.dtype == torch.float16, "feature shards must be fp16"
        assert tensor.ndim == 3, "expected [T, N, D]"
        key = _episode_key(int(ep_idx))
        payload[key] = tensor.contiguous()
        index[key] = {
            "T": int(tensor.shape[0]),
            "N": int(tensor.shape[1]),
            "D": int(tensor.shape[2]),
        }
    save_file(payload, str(shard_path))
    shard_path.with_suffix(shard_path.suffix + ".index.json").write_text(
        json.dumps(index)
    )


class FeatureCache:
    """Memory-mapped reader for fp16 DINO feature shards.

    Parameters
    ----------
    root : str | Path
        Directory laid out as ``<root>/<modality>/chunk-XXX/file-YYY.safetensors``.
    modality : {"rgb", "depth", "flow"}
        Which sub-directory under ``root`` to read.
    dtype : torch.dtype
        Materialization dtype. The on-disk dtype is fp16; we cast on demand.

    Notes
    -----
    We hold one ``safe_open`` handle per shard and keep them in a dict keyed by
    ``(chunk_id, shard_id)``. The handles are opened with ``device="cpu"`` so
    that ``get_tensor(...).to_dense()`` actually touches disk only when sliced.
    Multiple worker processes can hold their own ``FeatureCache`` instances on
    the same files: the underlying mmap pages are shared by the kernel.
    """

    def __init__(
        self,
        root: str | Path,
        modality: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        assert modality in {"rgb", "depth", "flow"}, modality
        self.root = Path(root)
        self.modality = modality
        self.dtype = dtype

        # episode_index -> (chunk_id, shard_path)
        self._index: Dict[int, Tuple[int, Path]] = {}

        modality_dir = self.root / modality
        if modality_dir.exists():
            for chunk_dir in sorted(modality_dir.glob("chunk-*")):
                try:
                    chunk_id = int(chunk_dir.name.split("-")[-1])
                except ValueError:
                    continue
                for shard in sorted(chunk_dir.glob("file-*.safetensors")):
                    self._register_shard(chunk_id, shard)

    # ----- internal helpers ------------------------------------------------

    def _register_shard(self, chunk_id: int, shard_path: Path) -> None:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith("ep_"):
                    continue
                try:
                    ep_idx = int(key.split("_")[-1])
                except ValueError:
                    continue
                self._index[ep_idx] = (chunk_id, shard_path)

    # ----- public API ------------------------------------------------------

    def episodes(self) -> Iterable[int]:
        """Iterate over all episode indices known to this cache."""
        return iter(self._index.keys())

    def has(self, episode_index: int) -> bool:
        """True if ``episode_index`` is present in the cache."""
        return int(episode_index) in self._index

    def get(
        self,
        episode_index: int,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[T_select, N, D]`` features for an episode.

        Parameters
        ----------
        episode_index : int
            Episode key.
        frame_indices : torch.Tensor | None
            Long tensor of frame positions to gather. ``None`` returns the full
            episode. Negative indices are not supported.

        Returns
        -------
        torch.Tensor
            ``[T_select, N, D]`` on CPU in ``self.dtype``.
        """
        assert isinstance(episode_index, int) and episode_index >= 0
        if episode_index not in self._index:
            raise KeyError(
                f"episode {episode_index} not in {self.modality} cache rooted at {self.root}"
            )
        _chunk_id, shard_path = self._index[episode_index]
        key = _episode_key(episode_index)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f_cpu:
            if frame_indices is None:
                tensor = f_cpu.get_tensor(key)
            else:
                assert frame_indices.dtype in (torch.long, torch.int64), (
                    f"frame_indices must be long, got {frame_indices.dtype}"
                )
                assert frame_indices.ndim == 1
                slc = f_cpu.get_slice(key)
                t_max = int(slc.get_shape()[0])
                idx_list = frame_indices.tolist()
                for i in idx_list:
                    if i < 0 or i >= t_max:
                        raise IndexError(
                            f"frame index {i} out of bounds for episode "
                            f"{episode_index} (T={t_max})"
                        )
                lo, hi = min(idx_list), max(idx_list) + 1
                window = slc[lo:hi]
                if not isinstance(window, torch.Tensor):
                    window = torch.as_tensor(window)
                gather = torch.tensor([i - lo for i in idx_list], dtype=torch.long)
                tensor = window.index_select(0, gather)

        if tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)
        return tensor

    def num_frames(self, episode_index: int) -> int:
        """Number of cached frames for ``episode_index``."""
        assert isinstance(episode_index, int) and episode_index >= 0
        if episode_index not in self._index:
            raise KeyError(episode_index)
        _chunk_id, shard_path = self._index[episode_index]
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            return int(f.get_slice(_episode_key(episode_index)).get_shape()[0])

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return (
            f"FeatureCache(root={self.root}, modality={self.modality}, "
            f"episodes={len(self._index)}, shards={len(set(p for _, p in self._index.values()))})"
        )
