# SPDX-License-Identifier: MIT
"""Unified per-dataset preparation pipeline.

One Python entry that walks chunks for a single dataset and runs every Phase A
stage (2a -> 2c -> 3 -> 4 -> 5) in-process, then writes an atomic
``.pipeline_complete`` marker. The orchestrator is preemption-safe: between
chunks it polls for SIGUSR1 and, when received, exits with
:data:`prep._base.PREEMPT_EXIT_CODE` (124) so the wrapping Slurm script can
``scontrol requeue`` and resume on the next allocation.

# Resume semantics

Resume is **local-only**: the cursor is the set of
``<output_root>/<dataset>/chunk-NNN/.pipeline_complete`` markers on disk. On
restart, the orchestrator scans markers and skips any chunk whose marker
exists. There is no HF Hub round-trip and no login-node daemon coordination.

# Walltime sizing

The wrapping ``slurm/pipeline_<dataset>.sbatch`` runs with ``--time=04:00:00``
and ``--signal=B:USR1@600``. The orchestrator installs a single SIGUSR1
handler in :meth:`PipelineOrchestrator.run`; the handler flips a flag that
``run`` polls between chunks. The in-flight chunk is allowed to finish
(stage_2a's own per-episode markers guard against mid-chunk loss). Once the
chunk finishes, the loop checks the flag and exits 124 if set.

# Chunk discovery

We don't pre-compute the chunk count. Each iteration calls
:func:`prep.stage_2a_to_lerobot.__main__.run_chunk`; if it produces zero
episodes the orchestrator infers we've reached the end and exits 0. This
avoids loading every per-source enumerator at startup just to learn the
total episode count.

# Stage flow per chunk

1. ``stage_2a_to_lerobot.run_chunk`` writes ``ep_<hash>/`` staging dirs.
2. ``stage_2c_compute_depth.compute_depth_multigpu`` writes ``depth_<cam>.npy``.
3. ``stage_3_canonical.run_chunk`` re-canonicalizes ``action_native.npy`` to
   ``action_canonical_ee.npy`` (idempotent: 2a already wrote it; 3 confirms
   the schema and embodiment match the registry).
4. ``stage_4_dino_cache.encode_chunk_multigpu`` writes Tri-DINO fp16
   safetensors under ``<dino_root>/<dataset>/...``.
5. ``stage_5_validate.validate_source_outputs`` walks the chunk dir; on full
   pass we then run cleanup and write the marker.
6. ``_cleanup_chunk`` removes intermediate files that downstream training does
   not need (``depth_*.npy``, ``depth_*.json``, ``done/``, ``_scratch/``).
   We keep: ``camera_*.npy`` (RGB), ``action_native.npy``,
   ``action_canonical_ee.npy``, ``state.npy``, ``timestamps.npy``,
   ``meta.json`` (instructions), and the DINO safetensors. Disable with
   ``--no-cleanup`` for debugging.

The ``.pipeline_complete`` marker is the ONLY file that promotes a chunk
from "in-progress" to "done"; it is written via the
``tmp + os.replace`` atomic pattern so a crash mid-write never leaves a
half-written marker. The marker payload records ``cleanup_bytes_freed``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import yaml  # type: ignore[import-not-found]

from prep._base import PREEMPT_EXIT_CODE

__all__ = [
    "PipelineOrchestrator",
    "DATASETS",
    "DATASET_TO_EMBODIMENT",
    "main",
]

_LOG = logging.getLogger("prep.run_pipeline")

# The 4 Tier-1 datasets the pipeline knows how to process. Each one has a
# matching ``configs/data/<dataset>.yaml`` and a per-source converter under
# ``prep/stage_2a_to_lerobot/``.
DATASETS: tuple[str, ...] = ("droid", "agibot2026", "robomind", "bridge")

DATASET_TO_EMBODIMENT: dict[str, str] = {
    "droid": "droid_franka",
    "bridge": "bridge_widowx",
    "agibot2026": "agibot_g1",
    "robomind": "robomind_tien_kung",
}


_PIPELINE_COMPLETE_MARKER = ".pipeline_complete"


@dataclass
class PipelineOrchestrator:
    """Per-dataset, per-chunk preparation driver.

    Parameters
    ----------
    dataset : str
        One of :data:`DATASETS`.
    output_root : Path
        Root for staged outputs. Each chunk lands at
        ``<output_root>/<dataset>/chunk-NNN/``.
    cfg : dict
        Parsed ``configs/data/<dataset>.yaml``. Used to look up depth /
        DINO / RLDS / raw-root settings.
    cfg_path : Path | None
        Path the cfg was loaded from; passed to stage_2a so per-source
        defaults that depend on the YAML's location still resolve.
    num_workers_2a : int
        CPU worker count for stage_2a. Default 8.
    num_gpus : int | None
        GPU count for stages 2c and 4. ``None`` auto-detects.
    workers_per_gpu : int
        Multiplier for GPU oversubscription on stages 2c / 4. Default 1.
    raw_root : Path | None
        Override for the per-source raw data root (AgiBot/RoboMIND need this).
    rlds_data_dir : str | None
        Override for the TFDS data dir (DROID/Bridge).
    karlp_root : Path | None
        Optional KarlP/droid local snapshot (DROID-only).
    dinov3_ckpt : str | None
        HF Hub model id or local path for stage_4. ``None`` means "use the
        YAML default; if absent, write zero-tensor placeholders".
    da3_ckpt : str | None
        HF Hub model id or local path for stage_2c.
    start_chunk : int
        First chunk id to consider (default 0).
    max_chunks : int | None
        Stop after this many chunks have been *attempted* in this invocation
        (default unlimited). Resume continues from the next un-done chunk on
        the next invocation.
    cleanup : bool
        Delete intermediate files after a chunk's stages succeed and before
        the ``.pipeline_complete`` marker is written. Default True. Set to
        False for debugging when you want to inspect the staged depth npy or
        the per-episode ``done/<hash>.ok`` markers.
    """

    dataset: str
    output_root: Path
    cfg: dict
    cfg_path: Path | None = None
    num_workers_2a: int = 8
    num_gpus: int | None = None
    workers_per_gpu: int = 1
    raw_root: Path | None = None
    rlds_data_dir: str | None = None
    karlp_root: Path | None = None
    dinov3_ckpt: str | None = None
    da3_ckpt: str | None = "depth-anything/DA3MONO-LARGE"
    start_chunk: int = 0
    max_chunks: int | None = None
    cleanup: bool = True
    _stop_requested: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        assert self.dataset in DATASETS, f"unknown dataset {self.dataset!r}; have {DATASETS}"
        self.output_root = Path(self.output_root)
        (self.output_root / self.dataset).mkdir(parents=True, exist_ok=True)

    @property
    def dataset_root(self) -> Path:
        return self.output_root / self.dataset

    def chunk_dir(self, chunk: int) -> Path:
        return self.dataset_root / f"chunk-{chunk:03d}"

    def marker_path(self, chunk: int) -> Path:
        return self.chunk_dir(chunk) / _PIPELINE_COMPLETE_MARKER

    def is_complete(self, chunk: int) -> bool:
        return self.marker_path(chunk).exists()

    def write_marker(self, chunk: int, payload: dict) -> None:
        """Atomically write the per-chunk completion marker."""
        marker = self.marker_path(chunk)
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, marker)

    # ------------------------------------------------------------------
    # SIGUSR1 handling
    # ------------------------------------------------------------------

    def _on_sigusr1(self, signum: int, frame: FrameType | None) -> None:
        # Keep the handler trivial: no IO, no exceptions. ``run`` polls
        # ``self._stop_requested`` between chunks.
        self._stop_requested = True
        _LOG.warning("SIGUSR1 received; will exit %d after current chunk", PREEMPT_EXIT_CODE)

    # ------------------------------------------------------------------
    # Per-stage callouts
    # ------------------------------------------------------------------

    def _run_stage_2a(self, chunk: int) -> tuple[int, int]:
        from prep.stage_2a_to_lerobot.__main__ import run_chunk as run_2a

        return run_2a(
            dataset=self.dataset,
            chunk=chunk,
            staged_root=self.output_root,
            cfg=self.cfg,
            raw_root=self.raw_root,
            rlds_data_dir=self.rlds_data_dir,
            karlp_root=self.karlp_root,
            config=self.cfg_path,
            resume=True,
            num_workers=self.num_workers_2a,
        )

    def _run_stage_2c(self, chunk: int) -> None:
        from prep.stage_2c_compute_depth import DepthConfig, compute_depth_multigpu

        depth_cfg_yaml = (self.cfg.get("depth") or {})
        depth_cfg = DepthConfig(
            target_hw=tuple(depth_cfg_yaml.get("target_hw", (192, 192))),  # type: ignore[arg-type]
            max_range_mm=int(depth_cfg_yaml.get("max_range_mm", 5000)),
            fp16=bool(depth_cfg_yaml.get("fp16", True)),
        )
        cameras = self._cameras()
        depth_out = self.dataset_root / f"chunk-{chunk:03d}"
        compute_depth_multigpu(
            staged_chunk_dir=self.chunk_dir(chunk),
            output_dir=depth_out,
            cameras=cameras,
            dav3_ckpt=self.da3_ckpt,
            config=depth_cfg,
            world_size=int(self.num_gpus or 0),
            workers_per_gpu=int(self.workers_per_gpu),
        )

    def _run_stage_3(self, chunk: int) -> int:
        from prep.stage_3_canonical import run_chunk as run_3

        embodiment = DATASET_TO_EMBODIMENT.get(self.dataset, self.dataset)
        return run_3(
            dataset=self.dataset,
            chunk=chunk,
            staged_root=self.output_root,
            embodiment=embodiment,
        )

    def _run_stage_4(self, chunk: int) -> None:
        from prep.stage_4_dino_cache import DinoCacheConfig, encode_chunk_multigpu

        dino_cfg_yaml = (self.cfg.get("dino_cache") or {})
        target_hw = tuple(dino_cfg_yaml.get("target_hw", (378, 378)))
        cfg = DinoCacheConfig(
            target_hw=target_hw,  # type: ignore[arg-type]
            n_keep_tokens=int(dino_cfg_yaml.get("n_keep_tokens", 64)),
            cache_fps=int(dino_cfg_yaml.get("cache_fps", 5)),
            fp16=bool(dino_cfg_yaml.get("fp16", True)),
        )
        cameras = self._cameras()
        ckpt = Path(self.dinov3_ckpt) if self.dinov3_ckpt else None
        encode_chunk_multigpu(
            staged_chunk_dir=self.chunk_dir(chunk),
            output_root=self.dataset_root,
            modalities=("rgb", "depth"),
            cameras=cameras,
            dinov3_ckpt=ckpt,
            source_fps=int(self.cfg.get("fps_native", 30)),
            world_size=int(self.num_gpus or 0),
            config=cfg,
            workers_per_gpu=int(self.workers_per_gpu),
        )

    def _run_stage_5(self, chunk: int) -> dict:
        """Lightweight per-chunk validation: required staging files present.

        Full ``validate_source_outputs`` expects the final LeRobot v2.1 layout
        (data/, videos/, features/) which the orchestrator does not produce on
        its own — that's the job of ``prep.stage_2a_to_lerobot._assemble``,
        which the e2e scripts run as a separate step. Until assembly is wired
        into the orchestrator, this lighter check just asserts every staged
        episode has the files downstream consumers expect.

        Returns ``{"summary": {...}, "fails": int}``.
        """
        chunk_dir = self.chunk_dir(chunk)
        ep_dirs = sorted(p for p in chunk_dir.glob("ep_*") if p.is_dir())
        required_files = [
            "meta.json",
            "action_native.npy",
            "action_canonical_ee.npy",
            "state.npy",
            "timestamps.npy",
        ]
        n_eps = 0
        n_missing = 0
        n_no_camera = 0
        for ep in ep_dirs:
            for fname in required_files:
                if not (ep / fname).exists():
                    n_missing += 1
            # Require AT LEAST ONE camera_*.npy per episode, not a specific
            # name. The config's cameras list is an aspirational set; each
            # converter writes whatever subset is actually present in the
            # source (e.g. DroidConverter only writes head_rgb because the
            # wrist view is disabled for egocentric-only training).
            if not list(ep.glob("camera_*.npy")):
                n_no_camera += 1
            n_eps += 1
        summary = {
            "n_episodes": int(n_eps),
            "n_missing_required": int(n_missing),
            "n_eps_without_camera": int(n_no_camera),
        }
        return {
            "summary": summary,
            "fails": int(n_missing > 0 or n_no_camera > 0 or n_eps == 0),
        }

    def _cameras(self) -> list[str]:
        cams = (self.cfg.get("convert") or {}).get("cameras") or ["head_rgb"]
        return list(cams)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_chunk(self, chunk: int) -> dict:
        """Delete intermediate files; keep only what training needs.

        After all 5 stages succeed, the chunk directory holds a mix of
        finished artifacts and disposable intermediates. Training consumes
        per-episode RGB ``camera_*.npy``, the action / state / timestamps /
        instructions, and the chunk-scoped DINO safetensors (which live at
        ``<dataset_root>/<cam>/<mod>/chunk-XXX/file-YYY.safetensors``). The
        rest is intermediate:

        * ``depth_<cam>.npy`` — raw depth used only to compute the depth
          modality of the DINO cache; the DINO safetensors already encode it.
        * ``depth_<cam>.json`` — sidecar flag (low_quality, source); not used
          by training.
        * ``done/<hash>.ok`` — per-episode CheckpointedJob markers; once
          ``.pipeline_complete`` exists the chunk is the unit of resume and
          per-episode markers are redundant.
        * ``_scratch/`` — CheckpointedJob's working area.

        Returns a dict with ``files_removed``, ``bytes_freed``, and lists of
        per-category sizes for the marker payload.

        Best-effort: missing files are skipped silently; an IOError on a
        deletion raises (so a bad permissions setup surfaces visibly rather
        than silently leaving space-leak debris on every chunk).
        """
        chunk_dir = self.chunk_dir(chunk)
        if not chunk_dir.exists():
            return {"files_removed": 0, "bytes_freed": 0}

        files_removed = 0
        bytes_freed = 0
        per_category = {"depth_npy": 0, "depth_json": 0, "done_markers": 0, "scratch": 0}

        # 1. Per-episode depth intermediates.
        for ep_dir in chunk_dir.glob("ep_*"):
            for path in list(ep_dir.glob("depth_*.npy")):
                bytes_freed += path.stat().st_size
                per_category["depth_npy"] += path.stat().st_size
                path.unlink()
                files_removed += 1
            for path in list(ep_dir.glob("depth_*.json")):
                bytes_freed += path.stat().st_size
                per_category["depth_json"] += path.stat().st_size
                path.unlink()
                files_removed += 1

        # 2. The chunk's done/<hash>.ok marker directory.
        done_dir = chunk_dir / "done"
        if done_dir.exists():
            for path in done_dir.rglob("*"):
                if path.is_file():
                    bytes_freed += path.stat().st_size
                    per_category["done_markers"] += path.stat().st_size
                    files_removed += 1
            shutil.rmtree(done_dir)

        # 3. CheckpointedJob scratch_dir for this chunk (under SCRATCH or
        #    output_root/_scratch/<dataset>/stage_2a/chunk-NNN/).
        scratch_candidates = [
            self.output_root / "_scratch" / self.dataset / "stage_2a" / f"chunk-{chunk:03d}",
            chunk_dir / "_scratch",
        ]
        env_scratch = os.environ.get("SCRATCH")
        if env_scratch:
            scratch_candidates.append(
                Path(env_scratch) / self.dataset / "stage_2a" / f"chunk-{chunk:03d}"
            )
        for cand in scratch_candidates:
            if cand.exists() and cand.is_dir():
                for path in cand.rglob("*"):
                    if path.is_file():
                        bytes_freed += path.stat().st_size
                        per_category["scratch"] += path.stat().st_size
                        files_removed += 1
                shutil.rmtree(cand, ignore_errors=True)

        return {
            "files_removed": int(files_removed),
            "bytes_freed": int(bytes_freed),
            "per_category": per_category,
        }

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_chunk(self, chunk: int) -> bool:
        """Run all 5 stages for one chunk. Returns True iff the chunk had work.

        A return value of ``False`` means stage_2a produced zero episodes,
        which the caller treats as "no more chunks" and exits cleanly.
        """
        chunk_dir = self.chunk_dir(chunk)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _LOG.info(
            "[%s] chunk %03d: starting stage_2a", self.dataset, chunk,
        )
        n_proc, n_skip = self._run_stage_2a(chunk)
        if n_proc == 0 and n_skip == 0:
            _LOG.info(
                "[%s] chunk %03d: stage_2a produced 0 episodes; treating as end-of-dataset",
                self.dataset, chunk,
            )
            return False

        _LOG.info("[%s] chunk %03d: stage_2c (depth)", self.dataset, chunk)
        self._run_stage_2c(chunk)
        _LOG.info("[%s] chunk %03d: stage_3 (canonical)", self.dataset, chunk)
        self._run_stage_3(chunk)
        _LOG.info("[%s] chunk %03d: stage_4 (dino_cache)", self.dataset, chunk)
        self._run_stage_4(chunk)
        _LOG.info("[%s] chunk %03d: stage_5 (validate)", self.dataset, chunk)
        report = self._run_stage_5(chunk)
        if report["fails"] > 0:
            raise RuntimeError(
                f"[{self.dataset}] chunk {chunk:03d} validation reported {report['fails']} fail(s); "
                f"summary={report['summary']}"
            )

        cleanup_info: dict = {"files_removed": 0, "bytes_freed": 0}
        if self.cleanup:
            _LOG.info("[%s] chunk %03d: cleanup intermediates", self.dataset, chunk)
            cleanup_info = self._cleanup_chunk(chunk)
            _LOG.info(
                "[%s] chunk %03d: freed %.2f MiB across %d intermediate files",
                self.dataset, chunk,
                cleanup_info["bytes_freed"] / (1024 ** 2),
                cleanup_info["files_removed"],
            )

        elapsed = time.time() - t0
        self.write_marker(chunk, {
            "dataset": self.dataset,
            "chunk": chunk,
            "n_processed": int(n_proc),
            "n_skipped": int(n_skip),
            "elapsed_seconds": float(elapsed),
            "validation_summary": report["summary"],
            "cleanup_files_removed": int(cleanup_info.get("files_removed", 0)),
            "cleanup_bytes_freed": int(cleanup_info.get("bytes_freed", 0)),
            "finished_at": int(time.time()),
        })
        _LOG.info(
            "[%s] chunk %03d: COMPLETE (%.1f s, %d episodes)",
            self.dataset, chunk, elapsed, n_proc,
        )
        return True

    def run(self) -> int:
        """Main loop. Returns 0 (all chunks done) or 124 (preempted)."""
        previous = signal.signal(signal.SIGUSR1, self._on_sigusr1)
        try:
            chunk = int(self.start_chunk)
            attempted = 0
            while True:
                if self._stop_requested:
                    _LOG.warning(
                        "[%s] SIGUSR1 before chunk %03d; exiting %d",
                        self.dataset, chunk, PREEMPT_EXIT_CODE,
                    )
                    return PREEMPT_EXIT_CODE
                if self.max_chunks is not None and attempted >= self.max_chunks:
                    _LOG.info(
                        "[%s] reached --max-chunks=%d; stopping at chunk %03d",
                        self.dataset, self.max_chunks, chunk,
                    )
                    return 0
                if self.is_complete(chunk):
                    _LOG.info(
                        "[%s] chunk %03d already complete; skipping",
                        self.dataset, chunk,
                    )
                    chunk += 1
                    continue
                attempted += 1
                has_work = self.run_chunk(chunk)
                if not has_work:
                    _LOG.info("[%s] all chunks processed; exiting 0", self.dataset)
                    return 0
                chunk += 1
                if self._stop_requested:
                    _LOG.warning(
                        "[%s] SIGUSR1 after chunk %03d; exiting %d",
                        self.dataset, chunk - 1, PREEMPT_EXIT_CODE,
                    )
                    return PREEMPT_EXIT_CODE
        finally:
            try:
                signal.signal(signal.SIGUSR1, previous)
            except (ValueError, TypeError):
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cfg(dataset: str, override: Path | None) -> tuple[dict, Path]:
    path = override if override is not None else Path("configs/data") / f"{dataset}.yaml"
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    return cfg, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prep.run_pipeline", description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="One of droid, agibot2026, robomind, bridge.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root for staged outputs. Each chunk lands at "
             "<output_root>/<dataset>/chunk-NNN/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override configs/data/<dataset>.yaml.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override per-source raw root (AgiBot HF snapshot / RoboMIND HDF5 tree).",
    )
    parser.add_argument(
        "--rlds-data-dir",
        type=str,
        default=None,
        help="Override the TFDS data_dir (DROID/Bridge).",
    )
    parser.add_argument(
        "--karlp-root",
        type=Path,
        default=None,
        help="Optional KarlP/droid local snapshot (DROID only).",
    )
    parser.add_argument(
        "--dinov3-ckpt",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HF Hub id or local path for DINOv3. Pass empty string to write "
             "zero-tensor placeholders (smoke mode).",
    )
    parser.add_argument(
        "--da3-ckpt",
        type=str,
        default="depth-anything/DA3MONO-LARGE",
        help="HF Hub id or local path for Depth-Anything-V3. Pass empty string "
             "to skip the model load (placeholder mode).",
    )
    parser.add_argument(
        "--num-workers-2a",
        type=int,
        default=8,
        help="CPU worker processes for stage_2a (default 8).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="GPU count for stages 2c and 4. 0 = auto-detect.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Workers per physical GPU on stages 2c/4 (default 1).",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="First chunk id to consider (default 0). Smaller-numbered "
             "chunks are skipped without checking their state.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after attempting this many chunks (default unlimited).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for parity with stage CLIs; the orchestrator is "
             "always resumable via .pipeline_complete markers.",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help="Disable post-chunk intermediate-file cleanup (depth npy/json, "
             "done/ markers, _scratch/). Default: cleanup enabled. Use this "
             "flag when debugging stage outputs.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg, cfg_path = _load_cfg(args.dataset, args.config)

    dinov3 = args.dinov3_ckpt if args.dinov3_ckpt else None
    da3 = args.da3_ckpt if args.da3_ckpt else None

    orch = PipelineOrchestrator(
        dataset=args.dataset,
        output_root=args.output_root,
        cfg=cfg,
        cfg_path=cfg_path,
        num_workers_2a=int(args.num_workers_2a),
        num_gpus=int(args.num_gpus) or None,
        workers_per_gpu=int(args.workers_per_gpu),
        raw_root=args.raw_root,
        rlds_data_dir=args.rlds_data_dir,
        karlp_root=args.karlp_root,
        dinov3_ckpt=dinov3,
        da3_ckpt=da3,
        start_chunk=int(args.start_chunk),
        max_chunks=args.max_chunks,
        cleanup=bool(args.cleanup),
    )
    rc = orch.run()
    return int(rc)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
