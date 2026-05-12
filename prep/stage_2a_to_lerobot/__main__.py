# SPDX-License-Identifier: MIT
"""CLI entry for Stage 2a — per-source RLDS/HDF5 -> USAM-LeRobot v2.1 conversion.

Driven once per (source, chunk) pair. Slurm's ``job.sbatch`` and the local
``scripts/prep_run_local.sh`` both invoke this as:

    python -m prep.stage_2a_to_lerobot --dataset <source> --chunk <N> --resume

The CLI loads ``configs/data/<source>.yaml`` to discover the RLDS/source root
and any per-source extras (KarlP overlay for DROID etc.), instantiates the
right converter class, and drives :meth:`Converter.process` per episode so
the staged ``ep_<hash>/camera_*.npy`` + ``action_native.npy`` + ``meta.json``
layout that downstream stages (2c depth, 3 canonical, 4 dino cache) consume
gets written.

The buffered parquet roll-up via ``write_shard`` is **not** invoked here —
it requires the full episode buffer in memory and is meant for the
``CheckpointedJob.run()`` driver. Per-episode staging is the correct unit
for the Phase A pipeline since the downstream stages all operate
episode-by-episode.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml  # type: ignore[import-not-found]

_LOG = logging.getLogger("prep.stage_2a_to_lerobot")


_SOURCES = ("droid", "agibot2026", "bridge", "rh20t", "robomind")


def _build_converter(args: argparse.Namespace, cfg: dict):
    """Instantiate the per-source converter from CLI args + YAML config.

    Only the source class import is per-dataset; the constructor shape is
    largely the same (chunk, output_root, source-specific extras).
    """
    download = cfg.get("download", {}) or {}
    output_root = args.staged_root / args.dataset / f"chunk-{args.chunk:03d}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "droid":
        from prep.stage_2a_to_lerobot.droid import DroidConverter
        rlds = args.rlds_data_dir or download.get("rlds_data_dir") or "gs://gresearch/robotics"
        karlp = args.karlp_root
        if karlp is None and "karlp_droid_repo" in download:
            karlp = Path(args.staged_root).parent / "raw" / "karlp_droid"
        return DroidConverter(
            chunk=args.chunk,
            output_root=output_root,
            rlds_data_dir=str(rlds),
            karlp_droid_root=Path(karlp) if karlp else None,
        )

    if args.dataset == "bridge":
        from prep.stage_2a_to_lerobot.bridge import BridgeConverter
        rlds = args.rlds_data_dir or download.get("rlds_data_dir") or "gs://gresearch/robotics"
        return BridgeConverter(
            chunk=args.chunk,
            output_root=output_root,
            rlds_data_dir=str(rlds),
        )

    if args.dataset == "agibot2026":
        from prep.stage_2a_to_lerobot.agibot2026 import AgibotConverter
        raw_root = args.raw_root
        if raw_root is None and "raw_root" in download:
            raw_root = Path(download["raw_root"])
        return AgibotConverter(
            chunk=args.chunk,
            output_root=output_root,
            raw_root=Path(raw_root) if raw_root else None,
        )

    if args.dataset == "rh20t":
        from prep.stage_2a_to_lerobot.rh20t import Rh20TConverter
        raw_root = args.raw_root or download.get("raw_root")
        if not raw_root:
            raise SystemExit(
                "RH20T requires --raw-root or download.raw_root in the YAML. "
                "RH20T tarballs must be mirrored locally before stage_2a — see configs/data/rh20t.yaml."
            )
        return Rh20TConverter(
            chunk=args.chunk,
            output_root=output_root,
            raw_root=Path(raw_root),
            config=args.rh20t_config or (download.get("configs") or [None])[args.chunk % len(download.get("configs") or [None])],
        )

    if args.dataset == "robomind":
        from prep.stage_2a_to_lerobot.robomind import RobomindConverter
        raw_root = args.raw_root or download.get("raw_root")
        if not raw_root:
            raise SystemExit(
                "RoboMIND requires --raw-root or download.raw_root in the YAML. "
                "HDF5 trajectories must be present locally before stage_2a — see configs/data/robomind.yaml."
            )
        return RobomindConverter(
            chunk=args.chunk,
            output_root=output_root,
            raw_root=Path(raw_root),
            drop_simulation=bool(download.get("drop_simulation", True)),
        )

    raise NotImplementedError(
        f"Stage 2a CLI not wired for dataset={args.dataset!r}. "
        f"Implement the converter import in prep.stage_2a_to_lerobot.__main__."
    )


def _process_one(converter, ref) -> None:
    """Run conversion + per-episode staging for any Tier-1 source.

    DROID's converter ships its own ``process`` (it pre-dates the base-class
    workflow). Every other converter only implements ``convert_episode`` from
    :class:`prep._base.CheckpointedJob`. To keep the downstream stages
    (stage_2c, stage_4, assemble) cross-dataset, we stage every episode in
    the same ``ep_<hash>/{camera_*.npy, action_*.npy, meta.json}`` layout
    that DroidConverter wrote — regardless of which converter produced the
    :class:`ConversionResult`.
    """
    if hasattr(converter, "process"):
        # DROID and any future converter that overrode `process` directly.
        return converter.process(ref)

    import json as _json
    import numpy as _np
    from prep.stage_2a_to_lerobot.droid import episode_filename_hash

    result = converter.convert_episode(ref)
    if result is None:
        return
    ep_dir = converter.output_root / (
        f"ep_{episode_filename_hash(int(result.episode_index), converter.SOURCE, converter.version)}"
    )
    ep_dir.mkdir(parents=True, exist_ok=True)
    _np.save(ep_dir / "action_native.npy", result.action_native)
    _np.save(ep_dir / "action_canonical_ee.npy", result.action_canonical_ee)
    _np.save(ep_dir / "state.npy", result.state)
    _np.save(ep_dir / "timestamps.npy", result.timestamps)
    for cam, arr in result.cameras.items():
        _np.save(ep_dir / f"camera_{cam}.npy", arr)
    meta = {
        "episode_index": int(result.episode_index),
        "embodiment": result.embodiment,
        "fps": float(result.fps),
        "instructions": result.instructions,
        "action_mask": result.action_mask.tolist(),
        "state_mask": result.state_mask.tolist(),
        "raw_meta": result.raw_meta,
    }
    (ep_dir / "meta.json").write_text(_json.dumps(meta))
    # The base class's per-episode marker (used by `is_done`) lives in
    # `<output_root>/done/<hash>.ok`. DROID writes a sibling `ep_<hash>.done`
    # next to the ep_ dir — keep both so resume-from-either works.
    marker = converter.output_root / f"ep_{episode_filename_hash(int(result.episode_index), converter.SOURCE, converter.version)}.done"
    marker.write_text(_json.dumps({"episode_index": int(result.episode_index)}))
    try:
        converter.mark_done(ref)
    except Exception:  # base-class mark_done is best-effort
        pass


def _worker_main(
    rank: int,
    world_size: int,
    args_dict: dict,
    cfg: dict,
) -> tuple[int, int]:
    """One worker process: build its own converter, claim episodes ``i % world_size == rank``.

    Stage 2a is CPU-only (TFDS decoding + numpy + disk writes), so plain
    multiprocessing is safe (no CUDA fork hazards). Each worker has its own
    TFDS builder, KarlP lookup, and writes ``ep_*/`` directories that the
    other workers don't touch.
    """
    import argparse as _argparse
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"[w{rank}/{world_size}] %(asctime)s %(message)s",
    )
    log = _logging.getLogger(__name__)
    args = _argparse.Namespace(**args_dict)
    converter = _build_converter(args, cfg)
    all_refs = list(converter.list_episodes())
    my_refs = [r for i, r in enumerate(all_refs) if i % world_size == rank]

    n_processed = 0
    n_skipped = 0
    for ref in my_refs:
        if args.resume and converter.is_done(ref):
            n_skipped += 1
            continue
        _process_one(converter, ref)
        n_processed += 1
        if n_processed % 5 == 0:
            log.info("staged %d episodes (skipped %d)", n_processed, n_skipped)
    log.info("worker done: processed=%d skipped=%d", n_processed, n_skipped)
    return n_processed, n_skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prep.stage_2a_to_lerobot")
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset", choices=sorted(_SOURCES))
    ds.add_argument(
        "--source",
        dest="dataset",
        choices=sorted(_SOURCES),
        help="(deprecated) use --dataset",
    )
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument(
        "--staged-root",
        type=Path,
        default=Path("/workspace/output/staged"),
        help="Root directory where ``<dataset>/chunk-NNN/ep_*/`` are written.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configs/data/<dataset>.yaml (default: configs/data/<dataset>.yaml).",
    )
    parser.add_argument(
        "--rlds-data-dir",
        type=str,
        default=None,
        help="Override the TFDS data_dir (else read from the YAML config).",
    )
    parser.add_argument(
        "--karlp-root",
        type=Path,
        default=None,
        help="Optional local snapshot of KarlP/droid (DROID only).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Local root for raw data (AgiBot HF snapshot / RH20T tarballs / "
             "RoboMIND HDF5 trees). Required when the source's download mode "
             "is 'local mirror'.",
    )
    parser.add_argument(
        "--rh20t-config",
        type=str,
        default=None,
        help="Pin a single RH20T per-rig config (e.g. 'RH20T_cfg3'). When "
             "omitted, the chunk index round-robins over download.configs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose ``ep_<hash>.done`` marker exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel CPU worker processes (default 1). >1 fans out via "
             "multiprocessing.Pool — TFDS decoding + numpy + disk I/O scale "
             "near-linearly across cores. Use ~num_cpu_cores or 16 for a "
             "modern node.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cfg_path = args.config or Path("configs/data") / f"{args.dataset}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    _LOG.info(
        "stage_2a: dataset=%s chunk=%d staged_root=%s num_workers=%d",
        args.dataset, args.chunk, args.staged_root, args.num_workers,
    )

    if args.num_workers <= 1:
        # Single-process path (bit-identical to the pre-parallel impl).
        converter = _build_converter(args, cfg)
        n_processed = 0
        n_skipped = 0
        for ref in converter.list_episodes():
            if args.resume and converter.is_done(ref):
                n_skipped += 1
                continue
            _process_one(converter, ref)
            n_processed += 1
            if n_processed % 5 == 0:
                _LOG.info("  staged %d episodes (skipped %d)", n_processed, n_skipped)
    else:
        # Multi-process: each worker builds its own converter and claims
        # episodes by `i % world_size == rank`. We use 'spawn' to avoid any
        # fork-after-import hazards from TFDS / PyTorch globals.
        import multiprocessing as mp
        # argparse.Namespace is not pickle-safe across spawn for Path args;
        # convert to a plain dict the worker reconstitutes via Namespace(**).
        args_dict = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        }
        # Restore Path types on the worker side too.
        args_dict["staged_root"] = Path(args_dict["staged_root"])
        if args_dict.get("karlp_root"):
            args_dict["karlp_root"] = Path(args_dict["karlp_root"])
        if args_dict.get("config"):
            args_dict["config"] = Path(args_dict["config"])
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            results = pool.starmap(
                _worker_main,
                [(r, args.num_workers, args_dict, cfg) for r in range(args.num_workers)],
            )
        n_processed = sum(p for p, _ in results)
        n_skipped = sum(s for _, s in results)

    _LOG.info(
        "stage_2a done: dataset=%s chunk=%d processed=%d skipped=%d",
        args.dataset, args.chunk, n_processed, n_skipped,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
