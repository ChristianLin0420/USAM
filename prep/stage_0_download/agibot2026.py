# SPDX-License-Identifier: MIT
"""Stage-0 downloader: AgiBot World 2026.

The dataset is hosted on HF Hub at ``agibot-world/AgiBot-World-2026`` and
already follows the LeRobot v2.1 layout, so we just snapshot it locally.

CLI
---
::

    python -m prep.stage_0_download.agibot2026 \
        --config configs/data/agibot2026.yaml \
        --cache /scratch/usam/agibot2026/raw \
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
    snapshot_hf_repo,
    write_download_manifest,
)

__all__ = ["download"]

_LOG = logging.getLogger(__name__)


def download(config_path: Path, cache: Path, dry_run: bool = False, allow_patterns: list[str] | None = None) -> DownloadResult:
    """Snapshot the AgiBot-World-2026 HF dataset into ``cache``.

    Parameters
    ----------
    config_path : Path
        Path to ``configs/data/agibot2026.yaml``.
    cache : Path
        Local cache root.
    dry_run : bool
        If True, no network IO; only a placeholder manifest is written.
    allow_patterns : list[str] | None
        Optional patterns forwarded to ``snapshot_download``.
    """
    assert isinstance(cache, Path)
    cfg: Any = load_yaml_config(Path(config_path))
    repo_id = str(cfg["download"]["hub_repo_id"])
    repo_type = str(cfg["download"].get("hub_repo_type", "dataset"))
    cache.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _LOG.info("DRY RUN: would snapshot %s -> %s", repo_id, cache)
        result = DownloadResult(
            source="agibot2026",
            cache_path=cache,
            backend="noop",
            files=0,
            extra={"repo_id": repo_id, "repo_type": repo_type, "dry_run": True},
        )
        write_download_manifest(cache, result)
        return result

    n_files = snapshot_hf_repo(
        repo_id=repo_id,
        cache_path=cache,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
    )
    result = DownloadResult(
        source="agibot2026",
        cache_path=cache,
        backend="hf_snapshot",
        files=n_files,
        extra={"repo_id": repo_id, "repo_type": repo_type},
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = make_cli("agibot2026", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
