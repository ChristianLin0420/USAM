# SPDX-License-Identifier: MIT
"""Stage-0 downloader: BridgeData V2 (RLDS).

Hosted at ``gs://gresearch/robotics/bridge``. We use TFDS's
``download_and_prepare`` to materialize the dataset into the local cache.

CLI
---
::

    python -m prep.stage_0_download.bridge \
        --config configs/data/bridge.yaml \
        --cache /scratch/usam/bridge/raw \
        --dry-run
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from prep.stage_0_download._common import (
    DownloadResult,
    load_yaml_config,
    make_cli,
    prepare_tfds,
    write_download_manifest,
)

__all__ = ["download"]

_LOG = logging.getLogger(__name__)


def download(config_path: Path, cache: Path, dry_run: bool = False, allow_patterns: list[str] | None = None) -> DownloadResult:
    """Prepare the BridgeData V2 RLDS dataset into ``cache``."""
    assert isinstance(cache, Path)
    cfg: Any = load_yaml_config(Path(config_path))
    rlds_data_dir = str(cfg["download"]["rlds_data_dir"])
    dataset_name = str(cfg["download"].get("rlds_dataset_name", "bridge"))

    cache.mkdir(parents=True, exist_ok=True)
    if dry_run:
        _LOG.info("DRY RUN: would prepare TFDS %s in %s", dataset_name, rlds_data_dir)
        result = DownloadResult(
            source="bridge",
            cache_path=cache,
            backend="noop",
            files=0,
            extra={"rlds_data_dir": rlds_data_dir, "dataset_name": dataset_name, "dry_run": True},
        )
        write_download_manifest(cache, result)
        return result

    n_episodes = prepare_tfds(
        dataset_name=dataset_name,
        data_dir=rlds_data_dir,
        cache_path=cache,
        download=True,
    )
    result = DownloadResult(
        source="bridge",
        cache_path=cache,
        backend="tfds",
        files=int(n_episodes),
        extra={"rlds_data_dir": rlds_data_dir, "dataset_name": dataset_name},
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = make_cli("bridge", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
