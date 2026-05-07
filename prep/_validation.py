# SPDX-License-Identifier: MIT
"""Common validation helpers for USAM Phase A pipeline outputs.

Three shape-and-dtype gates:

* :func:`validate_mp4` — runs ``ffprobe`` and asserts container readable,
  resolution matches expected, frame count > 0.
* :func:`validate_parquet` — opens the file with ``pyarrow``, asserts the
  expected columns are present (subset, not equality, so subclasses can add
  extras), and that row count > 0.
* :func:`validate_safetensors` — opens with ``safetensors.safe_open`` and
  asserts every tensor's dtype matches ``expected_dtype``.

Each returns a ``ValidationResult(ok: bool, errors: list[str])`` instead of
raising — callers (``stage_5_validate``, the reviewer agent's smoke checks)
want to surface every problem in a single pass.

# Optional dependencies

``pyarrow``, ``safetensors``, and ``ffprobe`` are imported/invoked lazily.
On a node missing one of them, the corresponding validator returns
``ok=False`` with an error message rather than blowing up at import time.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ValidationResult",
    "validate_mp4",
    "validate_parquet",
    "validate_safetensors",
]

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a single validation gate.

    Parameters
    ----------
    ok : bool
        True iff every check passed.
    errors : list[str]
        One human-readable string per failure. Empty when ``ok=True``.
    info : dict
        Extra metadata extracted during validation (resolution, row count,
        etc.); informational only.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _make_failure(msg: str, **info: object) -> ValidationResult:
    return ValidationResult(ok=False, errors=[msg], info=dict(info))


def validate_mp4(path: Path, expected_resolution: tuple[int, int] | None = None) -> ValidationResult:
    """Validate that ``path`` is a readable MP4 of the expected resolution.

    Uses ``ffprobe`` (must be on ``$PATH``).

    Parameters
    ----------
    path : Path
        MP4 file path.
    expected_resolution : tuple[int, int] | None
        ``(width, height)`` the validator should match. ``None`` skips the
        resolution check.

    Returns
    -------
    ValidationResult
        ``info`` may contain keys: ``width``, ``height``, ``nb_frames``,
        ``codec_name``.
    """
    assert isinstance(path, Path), f"path must be a Path, got {type(path).__name__}"
    if not path.exists():
        return _make_failure(f"file does not exist: {path}")
    if path.stat().st_size == 0:
        return _make_failure(f"file is empty: {path}")
    if shutil.which("ffprobe") is None:
        return _make_failure("ffprobe not found on PATH; cannot validate mp4")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,nb_read_packets,codec_name",
        "-of", "default=noprint_wrappers=1",
        "-count_packets",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except subprocess.TimeoutExpired:
        return _make_failure(f"ffprobe timed out on {path}")
    except subprocess.CalledProcessError as exc:
        return _make_failure(f"ffprobe exit {exc.returncode}: {exc.stderr.strip()}")

    info: dict = {}
    for line in out.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()

    errors: list[str] = []
    try:
        width = int(info.get("width", "0"))
        height = int(info.get("height", "0"))
    except ValueError:
        errors.append(f"could not parse width/height from ffprobe: {info}")
        width = height = 0

    # ffprobe reports nb_frames as 'N/A' for some encodings; fall back to
    # nb_read_packets which is always populated when ``-count_packets`` is set.
    raw_frames = info.get("nb_frames", "N/A")
    if raw_frames in ("N/A", ""):
        raw_frames = info.get("nb_read_packets", "0")
    try:
        nb_frames = int(raw_frames)
    except ValueError:
        nb_frames = 0
    info["nb_frames"] = nb_frames
    info["width"] = width
    info["height"] = height

    if nb_frames <= 0:
        errors.append(f"video has zero frames: {path}")
    if expected_resolution is not None:
        ew, eh = expected_resolution
        if (width, height) != (ew, eh):
            errors.append(f"resolution mismatch: got {width}x{height}, expected {ew}x{eh}")

    return ValidationResult(ok=not errors, errors=errors, info=info)


def validate_parquet(path: Path, expected_columns: list[str]) -> ValidationResult:
    """Validate ``path`` is a non-empty parquet file with the expected columns.

    The check is a **subset** check: every name in ``expected_columns`` must
    appear in the schema, but the file may carry additional columns.

    Parameters
    ----------
    path : Path
        Parquet file path.
    expected_columns : list[str]
        Column names that must be present.

    Returns
    -------
    ValidationResult
        ``info`` contains ``num_rows`` and ``columns``.
    """
    assert isinstance(path, Path), f"path must be a Path, got {type(path).__name__}"
    assert isinstance(expected_columns, list), "expected_columns must be a list"
    if not path.exists():
        return _make_failure(f"file does not exist: {path}")
    if path.stat().st_size == 0:
        return _make_failure(f"file is empty: {path}")

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return _make_failure("pyarrow is not installed; cannot validate parquet")

    try:
        pf = pq.ParquetFile(str(path))
    except Exception as exc:
        return _make_failure(f"failed to open parquet {path}: {exc}")

    schema_names = list(pf.schema_arrow.names)
    num_rows = pf.metadata.num_rows if pf.metadata is not None else 0
    info = {"num_rows": num_rows, "columns": schema_names}

    errors: list[str] = []
    missing = [c for c in expected_columns if c not in schema_names]
    if missing:
        errors.append(f"parquet {path} missing columns: {missing}")
    if num_rows <= 0:
        errors.append(f"parquet {path} has zero rows")

    return ValidationResult(ok=not errors, errors=errors, info=info)


def validate_safetensors(path: Path, expected_dtype: str) -> ValidationResult:
    """Validate every tensor in ``path`` has the expected dtype.

    Parameters
    ----------
    path : Path
        Safetensors file path.
    expected_dtype : str
        e.g. ``"F16"``, ``"BF16"``, ``"F32"``. Must match the
        safetensors-format dtype string exactly.

    Returns
    -------
    ValidationResult
        ``info`` contains ``num_tensors`` and ``tensor_names``.
    """
    assert isinstance(path, Path), f"path must be a Path, got {type(path).__name__}"
    assert isinstance(expected_dtype, str) and expected_dtype, "expected_dtype must be a non-empty string"
    if not path.exists():
        return _make_failure(f"file does not exist: {path}")
    if path.stat().st_size == 0:
        return _make_failure(f"file is empty: {path}")

    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return _make_failure("safetensors is not installed; cannot validate safetensors")

    errors: list[str] = []
    tensor_names: list[str] = []
    bad_dtypes: dict[str, str] = {}

    try:
        with safe_open(str(path), framework="pt") as f:  # type: ignore[no-untyped-call]
            for name in f.keys():
                tensor_names.append(name)
                meta = f.get_slice(name)
                # Safetensors exposes dtype as a torch dtype on PT framework;
                # we re-stringify and normalize to the safetensors short form.
                dt = _normalize_dtype(getattr(meta, "get_dtype", lambda: None)())
                if dt is None:
                    # Older safetensors: read the header directly.
                    import json as _json
                    with open(path, "rb") as raw:  # noqa: PTH123 - need binary handle
                        header_size = int.from_bytes(raw.read(8), "little")
                        header = _json.loads(raw.read(header_size))
                    dt = header.get(name, {}).get("dtype")
                if dt and dt != expected_dtype:
                    bad_dtypes[name] = dt
    except Exception as exc:
        return _make_failure(f"failed to read safetensors {path}: {exc}")

    if bad_dtypes:
        errors.append(
            f"safetensors {path} has tensors with wrong dtype "
            f"(expected {expected_dtype}): {bad_dtypes}"
        )
    if not tensor_names:
        errors.append(f"safetensors {path} contains zero tensors")

    return ValidationResult(
        ok=not errors,
        errors=errors,
        info={"num_tensors": len(tensor_names), "tensor_names": tensor_names},
    )


def _normalize_dtype(dt: object) -> str | None:
    """Map a torch dtype (or string) to a safetensors short dtype name."""
    if dt is None:
        return None
    s = str(dt)
    # torch.float16 -> 'torch.float16'
    table = {
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float64": "F64",
        "torch.int8": "I8",
        "torch.int16": "I16",
        "torch.int32": "I32",
        "torch.int64": "I64",
        "torch.uint8": "U8",
    }
    return table.get(s, s.upper())
