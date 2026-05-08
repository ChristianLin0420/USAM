# SPDX-License-Identifier: MIT
"""Stage-0 downloader: RH20T per-config tarballs.

RH20T is shipped as 7 per-configuration tarballs from rh20t.github.io.
There is **no** Hub mirror today, so this downloader assumes the operator
mirrors the tarballs to ``cfg.download.raw_root`` ahead of time and merely
records that the directory exists. ``--dry-run`` always succeeds.

CLI
---
::

    python -m prep.stage_0_download.rh20t \
        --config configs/data/rh20t.yaml \
        --cache /scratch/usam/rh20t/raw \
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
    write_download_manifest,
)

__all__ = ["download"]

_LOG = logging.getLogger(__name__)


def download(config_path: Path, cache: Path, dry_run: bool = False, allow_patterns: list[str] | None = None) -> DownloadResult:
    """Verify the RH20T raw-root layout, drop a manifest, return result.

    Real downloads happen out-of-band; this function only ensures we know
    where the data lives. We therefore never make a network call.
    """
    assert isinstance(cache, Path)
    cfg: Any = load_yaml_config(Path(config_path))
    raw_root = Path(str(cfg["download"]["raw_root"]))
    configs = list(cfg["download"].get("configs", []))

    cache.mkdir(parents=True, exist_ok=True)
    n_files = 0
    if not dry_run and raw_root.exists():
        n_files = sum(1 for _ in raw_root.rglob("*") if _.is_file())

    if dry_run:
        _LOG.info("DRY RUN: would verify RH20T raw_root=%s configs=%s", raw_root, configs)

    result = DownloadResult(
        source="rh20t",
        cache_path=cache,
        backend="noop" if dry_run else "verify_local",
        files=n_files,
        extra={
            "raw_root": str(raw_root),
            "configs": configs,
            "raw_root_exists": raw_root.exists(),
            "dry_run": dry_run,
        },
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = make_cli("rh20t", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
