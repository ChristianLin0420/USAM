# SPDX-License-Identifier: MIT
"""Stage-0 downloader: RoboMIND per-trajectory HDF5 archives.

RoboMIND (x-humanoid-robomind.github.io) is mirrored on HF Hub as
``x-humanoid/RoboMIND``; we snapshot it locally. If the config sets
``download.drop_simulation = true`` we drop the ``h5_simulation`` subtree
post-hoc by deleting it from the cache before stage_1 indexes it.

CLI
---
::

    python -m prep.stage_0_download.robomind \
        --config configs/data/robomind.yaml \
        --cache /scratch/usam/robomind/raw \
        --dry-run
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from prep.stage_0_download._common import (
    DownloadResult,
    load_yaml_config,
    make_cli,
    snapshot_hf_repo,
    write_download_manifest,
)

__all__ = ["download"]

_LOG = logging.getLogger(__name__)


def download(config_path: Path, cache: Path, dry_run: bool = False, allow_patterns: list[str] | None = None) -> DownloadResult:
    """Snapshot RoboMIND HDF5s, optionally pruning ``h5_simulation``."""
    assert isinstance(cache, Path)
    cfg: Any = load_yaml_config(Path(config_path))
    raw_root = Path(str(cfg["download"].get("raw_root", cache)))
    drop_sim = bool(cfg["download"].get("drop_simulation", True))
    repo_id = str(cfg["download"].get("hub_repo_id", "x-humanoid/RoboMIND"))

    cache.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _LOG.info("DRY RUN: would snapshot %s into %s, drop_simulation=%s", repo_id, cache, drop_sim)
        result = DownloadResult(
            source="robomind",
            cache_path=cache,
            backend="noop",
            files=0,
            extra={"repo_id": repo_id, "drop_simulation": drop_sim, "dry_run": True, "raw_root": str(raw_root)},
        )
        write_download_manifest(cache, result)
        return result

    n_files = snapshot_hf_repo(
        repo_id=repo_id,
        cache_path=cache,
        repo_type="dataset",
        allow_patterns=allow_patterns,
    )
    if drop_sim:
        sim_dir = cache / "h5_simulation"
        if sim_dir.exists():
            _LOG.info("dropping simulation subtree at %s", sim_dir)
            shutil.rmtree(sim_dir, ignore_errors=True)

    result = DownloadResult(
        source="robomind",
        cache_path=cache,
        backend="hf_snapshot",
        files=n_files,
        extra={"repo_id": repo_id, "drop_simulation": drop_sim, "raw_root": str(raw_root)},
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = make_cli("robomind", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
