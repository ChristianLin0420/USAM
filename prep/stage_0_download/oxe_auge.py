# SPDX-License-Identifier: MIT
"""Stage-0 downloader: OXE-AugE meta-collection (RLDS).

OXE-AugE is a meta-dataset of ~50 RLDS sub-sources rooted at
``gs://gresearch/robotics``. We iterate every entry from the per-source
manifest (``configs/data/oxe_auge.yaml::manifest``) and call
``tfds.builder(...).download_and_prepare()`` on each one. Sub-sources
without an ego camera are filtered out before any download begins.

CLI
---
::

    python -m prep.stage_0_download.oxe_auge \
        --config configs/data/oxe_auge.yaml \
        --cache /scratch/usam/oxe_auge/raw \
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
    """Prepare every ego-camera-bearing OXE-AugE sub-source into ``cache``."""
    assert isinstance(cache, Path)
    cfg: Any = load_yaml_config(Path(config_path))
    rlds_data_dir = str(cfg["download"]["rlds_data_dir"])
    manifest = dict(cfg.get("manifest", {}) or {})

    eligible = {
        name: entry for name, entry in manifest.items()
        if bool(entry.get("has_ego_camera", False))
    }
    cache.mkdir(parents=True, exist_ok=True)

    per_source_examples: dict[str, int] = {}
    if dry_run:
        _LOG.info(
            "DRY RUN: would prepare %d OXE-AugE sub-sources from %s",
            len(eligible), rlds_data_dir,
        )
        for name in eligible:
            per_source_examples[name] = 0
        result = DownloadResult(
            source="oxe_auge",
            cache_path=cache,
            backend="noop",
            files=0,
            extra={
                "rlds_data_dir": rlds_data_dir,
                "sub_sources": list(eligible.keys()),
                "per_source_examples": per_source_examples,
                "dry_run": True,
            },
        )
        write_download_manifest(cache, result)
        return result

    for name in eligible:
        sub_cache = cache / name
        try:
            per_source_examples[name] = prepare_tfds(
                dataset_name=name,
                data_dir=rlds_data_dir,
                cache_path=sub_cache,
                download=True,
            )
        except Exception as exc:  # pragma: no cover - real-runtime path
            _LOG.warning("OXE-AugE sub-source %s failed: %s", name, exc)
            per_source_examples[name] = -1

    files = sum(1 for _ in cache.rglob("*") if _.is_file())
    result = DownloadResult(
        source="oxe_auge",
        cache_path=cache,
        backend="tfds",
        files=files,
        extra={
            "rlds_data_dir": rlds_data_dir,
            "sub_sources": list(eligible.keys()),
            "per_source_examples": per_source_examples,
        },
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = make_cli("oxe_auge", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
