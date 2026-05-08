# SPDX-License-Identifier: MIT
"""Stage-0 downloader: DROID + KarlP/droid cleaner annotations.

DROID itself ships in TFDS at ``gs://gresearch/robotics/droid``. The
language-instruction overlay from the ``KarlP/droid`` HF repo is fetched
separately so the converter can prefer it over the (noisier) RLDS
``language_instruction`` field.

CLI
---
::

    python -m prep.stage_0_download.droid \
        --config configs/data/droid.yaml \
        --cache /scratch/usam/droid/raw \
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
    snapshot_hf_repo,
    write_download_manifest,
)

__all__ = ["download"]

_LOG = logging.getLogger(__name__)


def download(config_path: Path, cache: Path, dry_run: bool = False, allow_patterns: list[str] | None = None) -> DownloadResult:
    """Materialize DROID raw data + KarlP language overlay into ``cache``.

    Parameters
    ----------
    config_path : Path
        Path to ``configs/data/droid.yaml``.
    cache : Path
        Local cache root. TFDS data lands at ``cache/tfds`` and the
        KarlP overlay at ``cache/karlp_droid``.
    dry_run : bool
        If True, returns a planned ``DownloadResult`` and writes a manifest
        marker but performs no network IO.
    allow_patterns : list[str] | None
        Forwarded to ``snapshot_download`` for the KarlP overlay only.
    """
    assert isinstance(cache, Path), f"cache must be a Path, got {type(cache).__name__}"
    cfg: Any = load_yaml_config(Path(config_path))
    rlds_data_dir = str(cfg["download"]["rlds_data_dir"])
    karlp_repo = str(cfg["download"].get("karlp_droid_repo", "KarlP/droid"))

    cache.mkdir(parents=True, exist_ok=True)
    tfds_dir = cache / "tfds"
    karlp_dir = cache / "karlp_droid"

    if dry_run:
        _LOG.info("DRY RUN: would prepare TFDS droid at %s and snapshot %s", tfds_dir, karlp_repo)
        result = DownloadResult(
            source="droid",
            cache_path=cache,
            backend="noop",
            files=0,
            extra={"rlds_data_dir": rlds_data_dir, "karlp_repo": karlp_repo, "dry_run": True},
        )
        write_download_manifest(cache, result)
        return result

    n_episodes = prepare_tfds(
        dataset_name="droid",
        data_dir=rlds_data_dir,
        cache_path=tfds_dir,
        download=True,
    )
    n_files = snapshot_hf_repo(
        repo_id=karlp_repo,
        cache_path=karlp_dir,
        repo_type="dataset",
        allow_patterns=allow_patterns,
    )
    result = DownloadResult(
        source="droid",
        cache_path=cache,
        backend="tfds+hf_snapshot",
        files=n_files,
        extra={
            "rlds_data_dir": rlds_data_dir,
            "tfds_episodes": int(n_episodes),
            "karlp_repo": karlp_repo,
            "karlp_files": int(n_files),
        },
    )
    write_download_manifest(cache, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 on usage error."""
    parser = make_cli("droid", download, __doc__ or "")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    download(args.config, args.cache, dry_run=args.dry_run, allow_patterns=args.allow_patterns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
