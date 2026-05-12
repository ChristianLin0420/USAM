# SPDX-License-Identifier: MIT
"""Generate USAM golden test fixtures.

Two operating modes:

1. ``--use-mocks`` (default; offline). Calls each per-source
   ``_synthesize_<source>.py`` helper and produces a deterministic, on-disk
   fixture in the USAM-LeRobot v2.1 layout. No network access required;
   total footprint is well under 5 MB.

2. **Real-download path** (commented out below; manual). Pulls a tiny slice
   of each Tier-1 source via ``huggingface-cli`` and runs the Phase A
   pipeline (``prep.stage_2a_to_lerobot.<source>`` + ``prep.stage_4_dino_cache``)
   over those raw episodes. The team lead (not the agent) executes this
   when LFS is provisioned.

Both modes are **idempotent**: re-running over an existing ``--out``
directory overwrites cleanly without duplicating shards. The script asserts
the total fixture footprint is ≤ ``--max-size-mb`` after generation;
exceeding the budget is a hard error.

Usage
-----
.. code-block:: bash

   # Mock mode (CI / agent runs):
   python tests/golden_data/build_fixtures.py --use-mocks

   # Mock mode against a custom destination + tighter budget:
   python tests/golden_data/build_fixtures.py \\
       --use-mocks --out /tmp/usam_fixtures --max-size-mb 50

Real-download path (run manually, not from CI):

.. code-block:: bash

   # 1. Authenticate first.
   huggingface-cli login

   # 2. DROID — first 5 episodes.
   huggingface-cli download \\
       cadene/droid_100 \\
       --include "data/chunk-000/episode_00000{0..4}.*" \\
       --local-dir tests/golden_data/raw/droid

   # 3. AgiBot 2026 — first 5 episodes.
   huggingface-cli download \\
       agibot-world/AgiBotWorld-Beta \\
       --include "data/chunk-0/file-0000{0..4}.*" \\
       --local-dir tests/golden_data/raw/agibot2026

   # 4. RoboMIND 2.0 — first 3 episodes (Tien Kung subset).
   huggingface-cli download \\
       AgiBotTech/RoboMIND_v1_0_2 \\
       --include "Tien_Kung/h5_real/*0000.h5" \\
       --local-dir tests/golden_data/raw/robomind

   # 5. Bridge V2 — first 3 episodes from the RLDS dump on GCS.
   gsutil -m cp -r 'gs://gresearch/robotics/bridge/0.1.0/bridge_dataset-train.tfrecord-0000{0..2}-of-*' \\
       tests/golden_data/raw/bridge

   # 6. RH20T — first 3 episodes (RH20T_cfg1 subset).
   huggingface-cli download \\
       robofm/rh20t \\
       --include "RH20T_cfg1/scene_00*/folder_00{0..2}/*" \\
       --local-dir tests/golden_data/raw/rh20t

   # 7. Run the converters + Phase A.4 DINO cache against the raw slices.
   #    See scripts/prep_run_local.sh for the exact invocation used in CI rehearsal.

Notes
-----
* The mock path is what unit + integration tests use today. The real path
  exists so the same fixture *layout* can be re-generated at any time from
  authentic upstream data when LFS is available.
* The 200 MB budget is the LFS hard cap from the agent charter; the mock
  path uses < 5 MB so we have ample headroom for adding sources.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict


# Each entry: (source_name -> callable(out_root: Path) -> Path)
# We import lazily inside ``_synthesizers()`` so the module imports cleanly
# even if a single source's deps are missing.
def _synthesizers() -> Dict[str, Callable[[Path], Path]]:
    from tests.golden_data._synthesize_agibot import synthesize_tiny_agibot
    from tests.golden_data._synthesize_bridge import synthesize_tiny_bridge
    from tests.golden_data._synthesize_rh20t import synthesize_tiny_rh20t
    from tests.golden_data._synthesize_robomind import synthesize_tiny_robomind
    from tests.golden_data._synthesize_tiny_droid import synthesize_tiny_droid

    return {
        "tiny_droid": synthesize_tiny_droid,
        "tiny_agibot": synthesize_tiny_agibot,
        "tiny_robomind": synthesize_tiny_robomind,
        "tiny_bridge": synthesize_tiny_bridge,
        "tiny_rh20t": synthesize_tiny_rh20t,
    }


def _dir_size_bytes(path: Path) -> int:
    """Recursive total byte count under ``path`` (zero if missing)."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    if n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    return f"{n_bytes / 1024 ** 3:.1f} GB"


def build_fixtures_mock(out_root: Path, sources: list[str] | None = None) -> Dict[str, Path]:
    """Run every per-source synthesizer under ``out_root``.

    Parameters
    ----------
    out_root : Path
        Root directory under which each source becomes a subdirectory
        (``out_root/tiny_droid``, ``out_root/tiny_agibot``, ...).
    sources : list[str] | None
        Subset of source names to materialize. ``None`` means "all".

    Returns
    -------
    dict[str, Path]
        Mapping ``source_name -> directory path``.
    """
    assert isinstance(out_root, Path), f"out_root must be Path, got {type(out_root)}"
    out_root.mkdir(parents=True, exist_ok=True)

    synthesizers = _synthesizers()
    selected = sources or list(synthesizers.keys())
    unknown = [s for s in selected if s not in synthesizers]
    assert not unknown, f"unknown sources: {unknown}; valid: {sorted(synthesizers)}"

    written: Dict[str, Path] = {}
    for name in selected:
        src_root = out_root / name
        logging.info("synthesizing %s -> %s", name, src_root)
        path = synthesizers[name](src_root)
        written[name] = path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent,
        help="Output root (default: this directory).",
    )
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        default=True,
        help="Generate fixtures from the in-process synthesizers (default; offline).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Restrict to a subset of source names; omit to build all.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=200,
        help="Hard ceiling on total fixture size (LFS budget). Default 200 MB.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.use_mocks:
        # The real-download path is intentionally unimplemented in-process.
        # See the module docstring for the manual ``huggingface-cli`` recipe.
        raise SystemExit(
            "Real-download path is not driven by this script. See the module "
            "docstring for the huggingface-cli commands the team lead runs "
            "manually when LFS is provisioned."
        )

    written = build_fixtures_mock(args.out, sources=args.sources)
    for name, path in written.items():
        size = _dir_size_bytes(path)
        logging.info("%-20s %s (%s)", name, path, _format_size(size))

    total = sum(_dir_size_bytes(p) for p in written.values())
    max_bytes = int(args.max_size_mb) * 1024 ** 2
    logging.info("total: %s (budget %d MB)", _format_size(total), args.max_size_mb)
    assert total <= max_bytes, (
        f"fixture total {_format_size(total)} exceeds --max-size-mb={args.max_size_mb} "
        f"({_format_size(max_bytes)}); shrink the per-source synthesizer or raise the budget"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
