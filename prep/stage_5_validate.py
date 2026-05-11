# SPDX-License-Identifier: MIT
"""Stage 5: final per-shard validation gate.

For a given source's output dir we walk every shard produced by stages
2a/2b/2c/3/4 and run the appropriate ``prep._validation`` validator on it:

* ``data/chunk-XXX/file-YYY.parquet`` -> :func:`prep._validation.validate_parquet`
  with the canonical USAM-LeRobot v2.1 column set.
* ``videos/<key>/chunk-XXX/file-YYY.mp4`` -> :func:`prep._validation.validate_mp4`
  with per-modality resolution gates.
* ``features/<modality>/chunk-XXX/file-YYY.safetensors`` ->
  :func:`prep._validation.validate_safetensors` with ``F16``.

We collect every error per shard rather than failing fast and produce a
JSON report at ``<output_root>/<source>/validation_report.json``. Exit
code is 0 if every shard passes, non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prep._validation import (
    ValidationResult,
    validate_mp4,
    validate_parquet,
    validate_safetensors,
)

__all__ = [
    "ShardReport",
    "ValidationReport",
    "validate_source_outputs",
    "main",
]

_LOG = logging.getLogger(__name__)

# Canonical parquet column set produced by stage_2a converters.
_PARQUET_REQUIRED_COLUMNS: list[str] = [
    "episode_index",
    "frame_index",
    "timestamp",
    "embodiment",
    "proprio",
    "action_native",
    "action_canonical_ee",
    "action_mask",
    "state_mask",
    "level_1",
    "level_2",
    "level_3",
    "subtask_label",
]

# Per-camera-key MP4 gates. Resolution is `(width, height)`; ``None`` means
# the validator runs but does not assert resolution.
_MP4_RESOLUTIONS: dict[str, tuple[int, int] | None] = {
    "head_rgb": (378, 378),
    "wrist_rgb": (378, 378),
    "wrist_rgb_left": (378, 378),
    "wrist_rgb_right": (378, 378),
    "head_depth": (192, 192),
    "wrist_depth": (192, 192),
}

_SAFETENSORS_DTYPE: str = "F16"


@dataclass
class ShardReport:
    """Per-shard validation result.

    Parameters
    ----------
    path : str
        Absolute path of the shard.
    kind : str
        ``"parquet"``, ``"mp4"``, or ``"safetensors"``.
    ok : bool
    errors : list[str]
    info : dict
        Validator-specific extra info (resolution, row count, ...).
    """

    path: str
    kind: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated per-source report.

    Parameters
    ----------
    source : str
    output_root : str
    shards : list[ShardReport]
    summary : dict
        Counts: ``parquet_ok``, ``parquet_fail``, ``mp4_ok``, ``mp4_fail``,
        ``safetensors_ok``, ``safetensors_fail``.
    """

    source: str
    output_root: str
    shards: list[ShardReport] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _to_shard_report(path: Path, kind: str, vr: ValidationResult) -> ShardReport:
    return ShardReport(
        path=str(path),
        kind=kind,
        ok=vr.ok,
        errors=list(vr.errors),
        info=dict(vr.info),
    )


def _iter_parquets(source_root: Path) -> Iterable[Path]:
    return sorted((source_root / "data").rglob("*.parquet")) if (source_root / "data").exists() else []


def _iter_mp4s(source_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(camera_key, mp4_path)`` pairs for every shipped video."""
    videos = source_root / "videos"
    if not videos.exists():
        return
    for cam_dir in sorted(p for p in videos.iterdir() if p.is_dir()):
        # Strip the LeRobot 'observation.images.' prefix when present so we
        # can look up resolution gates by the short canonical key.
        key = cam_dir.name.replace("observation.images.", "")
        for mp4 in sorted(cam_dir.rglob("*.mp4")):
            yield key, mp4


def _iter_safetensors(source_root: Path) -> Iterable[Path]:
    return sorted((source_root / "features").rglob("*.safetensors")) if (source_root / "features").exists() else []


def validate_source_outputs(
    output_root: Path,
    source: str,
    parquet_columns: list[str] | None = None,
    mp4_resolutions: dict[str, tuple[int, int] | None] | None = None,
    safetensors_dtype: str = _SAFETENSORS_DTYPE,
) -> ValidationReport:
    """Validate every shard under ``<output_root>/<source>`` and return a report.

    Parameters
    ----------
    output_root : Path
    source : str
    parquet_columns : list[str] | None
        Columns that *must* be present in every parquet. ``None`` means use
        the canonical USAM-LeRobot v2.1 set.
    mp4_resolutions : dict | None
        Per-camera-key gates. Unrecognised keys validate without a
        resolution assertion.
    safetensors_dtype : str
        Expected safetensors dtype string. Default ``F16``.

    Returns
    -------
    ValidationReport
    """
    assert isinstance(output_root, Path), f"output_root must be a Path, got {type(output_root).__name__}"
    assert isinstance(source, str) and source, "source must be a non-empty str"
    parquet_columns = parquet_columns or list(_PARQUET_REQUIRED_COLUMNS)
    mp4_resolutions = dict(_MP4_RESOLUTIONS if mp4_resolutions is None else mp4_resolutions)

    source_root = output_root / source
    report = ValidationReport(source=source, output_root=str(source_root.resolve()))

    n = {"parquet_ok": 0, "parquet_fail": 0, "mp4_ok": 0, "mp4_fail": 0,
         "safetensors_ok": 0, "safetensors_fail": 0}

    for parquet in _iter_parquets(source_root):
        sr = _to_shard_report(parquet, "parquet", validate_parquet(parquet, parquet_columns))
        report.shards.append(sr)
        n["parquet_ok" if sr.ok else "parquet_fail"] += 1

    for cam_key, mp4 in _iter_mp4s(source_root):
        expected = mp4_resolutions.get(cam_key, None)
        sr = _to_shard_report(mp4, "mp4", validate_mp4(mp4, expected_resolution=expected))
        sr.info["camera_key"] = cam_key
        report.shards.append(sr)
        n["mp4_ok" if sr.ok else "mp4_fail"] += 1

    for st in _iter_safetensors(source_root):
        sr = _to_shard_report(st, "safetensors", validate_safetensors(st, safetensors_dtype))
        report.shards.append(sr)
        n["safetensors_ok" if sr.ok else "safetensors_fail"] += 1

    report.summary = n
    return report


def write_report(report: ValidationReport, dest: Path) -> Path:
    """Persist ``report`` as ``validation_report.json`` and return the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": report.source,
        "output_root": report.output_root,
        "summary": report.summary,
        "shards": [asdict(s) for s in report.shards],
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI: validate all shards for a source and exit 0 only on full pass."""
    parser = argparse.ArgumentParser(prog="prep.stage_5_validate")
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset",
                    help="Source name (one A100 node per dataset, per Wave F).")
    ds.add_argument("--source", dest="dataset",
                    help="(deprecated) use --dataset")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Root containing <dataset>/{data,videos,features}/")
    parser.add_argument("--report", type=Path, default=None,
                        help="Where to write validation_report.json "
                             "(default: <output-root>/<dataset>/validation_report.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Accepted for symmetry with other stages; validation is stateless")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    report = validate_source_outputs(args.output_root, args.dataset)
    dest = args.report or (args.output_root / args.dataset / "validation_report.json")
    write_report(report, dest)

    fails = sum(v for k, v in report.summary.items() if k.endswith("_fail"))
    if fails:
        msg = ", ".join(f"{k}={v}" for k, v in sorted(report.summary.items()))
        print(f"VALIDATION FAILED: {msg}; report at {dest}", file=sys.stderr)
        return 1
    print(f"VALIDATION OK: report at {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
