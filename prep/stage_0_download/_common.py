# SPDX-License-Identifier: MIT
"""Helpers shared by every Stage-0 downloader.

The downloaders are deliberately thin: each one chooses between two
backends (``huggingface_hub.snapshot_download`` for HF-hosted sources or
``tensorflow_datasets.builder().download_and_prepare()`` for RLDS sources)
and writes a tiny ``download_manifest.json`` next to the cache so
``stage_1_index`` knows which directory to scan.

The helpers here are import-light and never touch the network until a
non-dry-run ``download`` call is made.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "DownloadResult",
    "load_yaml_config",
    "snapshot_hf_repo",
    "prepare_tfds",
    "write_download_manifest",
    "make_cli",
]

_LOG = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Outcome of a Stage-0 download.

    Parameters
    ----------
    source : str
        e.g. ``"droid"``.
    cache_path : Path
        Local directory containing the raw data.
    backend : str
        ``"hf_snapshot"``, ``"tfds"``, or ``"noop"`` (dry run).
    files : int
        Approximate file count after the download.
    extra : dict
        Backend-specific metadata. JSON-serializable.
    """

    source: str
    cache_path: Path
    backend: str
    files: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def load_yaml_config(path: Path) -> Mapping[str, Any]:
    """Load a per-source YAML config.

    Lazy-imports ``yaml`` so callers without it can still ``import`` the
    module (they just cannot actually run a download).
    """
    assert isinstance(path, Path), f"path must be a Path, got {type(path).__name__}"
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - tested in real CI only
        raise RuntimeError("PyYAML is required to load source configs") from e
    return yaml.safe_load(path.read_text())


def snapshot_hf_repo(
    repo_id: str,
    cache_path: Path,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> int:
    """Wrap ``huggingface_hub.snapshot_download`` and return file count.

    Lazy import keeps this module test-friendly. Returns the number of
    files inside ``cache_path`` after the snapshot completes.
    """
    assert isinstance(repo_id, str) and "/" in repo_id, f"repo_id must be 'org/name', got {repo_id!r}"
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download  # local import

    _LOG.info("snapshot_download(%s) -> %s", repo_id, cache_path)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(cache_path),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        revision=revision,
        token=token,
    )
    return sum(1 for _ in cache_path.rglob("*") if _.is_file())


def prepare_tfds(
    dataset_name: str,
    data_dir: str,
    cache_path: Path,
    download: bool = True,
) -> int:
    """Wrap ``tensorflow_datasets.builder().download_and_prepare()``.

    The ``data_dir`` argument may point at a GCS bucket
    (``gs://gresearch/robotics``) or a local mirror. We always pass
    ``data_dir`` directly to TFDS so its on-disk layout determines the
    cache location; ``cache_path`` is recorded only for the manifest.
    """
    assert isinstance(dataset_name, str) and dataset_name, "dataset_name required"
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow_datasets as tfds  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - real-runtime path
        raise RuntimeError(
            "tensorflow_datasets is required for RLDS downloads "
            "(pip install -r requirements/prep.txt)"
        ) from e

    _LOG.info("tfds.builder(%s, data_dir=%s).download_and_prepare()", dataset_name, data_dir)
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    if download:
        builder.download_and_prepare()
    n_examples = 0
    try:
        n_examples = int(builder.info.splits["train"].num_examples)
    except Exception:  # pragma: no cover - tfds builders sometimes lack splits
        n_examples = 0
    return n_examples


def write_download_manifest(cache_path: Path, result: DownloadResult) -> Path:
    """Drop a ``download_manifest.json`` next to the data.

    Stage-1 indexers consume this file to discover where the raw data lives
    and which backend produced it. The manifest is also a useful idempotency
    fence: if a re-run finds it, the downloader can skip work cheaply.
    """
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": result.source,
        "cache_path": str(cache_path.resolve()),
        "backend": result.backend,
        "files": int(result.files),
        "extra": result.extra,
        "written_at": int(time.time()),
    }
    out = cache_path / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _LOG.info("wrote download manifest at %s", out)
    return out


def make_cli(
    source: str,
    download_fn: Any,
    description: str,
) -> argparse.ArgumentParser:
    """Build the standard ``--config / --cache / --dry-run`` argument parser.

    Each per-source ``__main__`` block calls
    ``parser = make_cli(source="droid", download_fn=download, description="...")``
    and then runs ``download_fn`` with the parsed args.
    """
    p = argparse.ArgumentParser(
        prog=f"prep.stage_0_download.{source}",
        description=description,
    )
    p.add_argument("--config", type=Path, required=True, help="Per-source YAML config path")
    p.add_argument("--cache", type=Path, required=True, help="Local cache root")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan the download but do not contact the network or read data")
    p.add_argument("--allow-patterns", nargs="*", default=None,
                   help="Optional glob patterns restricting HF snapshot scope")
    return p
