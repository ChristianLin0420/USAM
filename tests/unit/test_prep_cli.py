# SPDX-License-Identifier: MIT
"""Wave-F CLI surface test: every prep stage exposes ``--dataset``.

Per Wave F, each dataset runs on a separate A100 Slurm node, so the CLI
flag for the dataset name is ``--dataset`` (with ``--source`` kept as a
deprecated alias for one release).

We exercise the ``--help`` output for each stage rather than running the
stage proper so the test is fast and dependency-free.
"""

from __future__ import annotations

import subprocess
import sys

PY = sys.executable
REPO_ROOT_ENV = {"PYTHONUNBUFFERED": "1"}

_STAGES = [
    "prep.stage_1_index",
    "prep.stage_2c_compute_depth",
    "prep.stage_3_canonical",
    "prep.stage_4_dino_cache",
    "prep.stage_5_validate",
    "prep.stage_6_upload",
]


def _run_help(module: str) -> str:
    """Return stdout of ``python -m <module> --help``."""
    out = subprocess.run(
        [PY, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # ``argparse``'s --help exits 0; any other code means the module failed to import.
    assert out.returncode == 0, (
        f"`python -m {module} --help` exited {out.returncode}\n"
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    )
    return out.stdout


def test_stage_1_index_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_1_index")


def test_stage_2c_compute_depth_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_2c_compute_depth")


def test_stage_3_canonical_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_3_canonical")


def test_stage_4_dino_cache_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_4_dino_cache")


def test_stage_5_validate_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_5_validate")


def test_stage_6_upload_help_has_dataset() -> None:
    assert "--dataset" in _run_help("prep.stage_6_upload")


def test_source_is_deprecated_alias_for_stage_2c() -> None:
    """``--source`` must still accept input and route to ``args.dataset``."""
    help_text = _run_help("prep.stage_2c_compute_depth")
    assert "--source" in help_text, "deprecated --source alias missing"
    assert "deprecated" in help_text.lower()


def test_stage_2c_compute_depth_default_ckpt_is_da3() -> None:
    """The default for ``--dav3-ckpt`` must be DA3MONO-LARGE."""
    help_text = _run_help("prep.stage_2c_compute_depth")
    assert "depth-anything/DA3MONO-LARGE" in help_text, (
        "stage_2c's --dav3-ckpt should default to depth-anything/DA3MONO-LARGE\n"
        f"got help:\n{help_text}"
    )
