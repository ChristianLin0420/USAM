# USAM staging disk-space estimate

This document is the budget you should plan ``$USAM_SCRATCH`` against when
you submit the 4 per-dataset preparation pipelines (``slurm/pipeline_<ds>.sbatch``).
Numbers reflect what survives ``prep.run_pipeline``'s automatic cleanup at the
end of each chunk — i.e. RGB + DINO features + state + action + instructions.
Intermediate depth npy files, per-episode ``done/`` markers, and CheckpointedJob
scratch directories are removed before ``.pipeline_complete`` is written.

The numbers below are *post-cleanup* steady-state. During a chunk's execution
the on-disk footprint briefly grows by an additional ~10-20% for the depth
npy intermediates; the cleanup step shrinks it back before moving on.

## TL;DR

| Dataset    | Episodes (≈) | Chunks (256/chunk) | Per-chunk peak (≈) | Per-dataset total (≈) |
| ---------- | ------------ | ------------------ | ------------------ | --------------------- |
| DROID      |    76 000    |     297            |     22 GiB         |     6.4 TiB           |
| AgiBot2026 | 1 000 000    |   3 907            |     65 GiB         |    >100 TiB           |
| RoboMIND   |    30 000    |     118            |     35 GiB         |     4.0 TiB           |
| Bridge V2  |    60 000    |     235            |      7 GiB         |     1.6 TiB           |
| **Total**  |              |                    |                    | **>110 TiB**          |

**Reality check on the AgiBot total.** AgiBot World 2026 ships ~1M episodes
across ~80 embodiments and 3 cameras at 30 Hz; the storage cost is roughly
proportional to (cameras × frames × resolution). The naive uncompressed
staging tree for AgiBot at full size will not fit on most Slurm scratch
filesystems. Two practical knobs:

1. Subset selection. The default config processes everything; in practice
   you start with a single embodiment (e.g. ``g1_dual_arm``) which is
   ~50 000 episodes ≈ 5 TiB.
2. Compression. The staged ``camera_*.npy`` files are uncompressed
   ``uint8``. Encoding to H.264 at CRF 23 (per the LeRobot v2.1 spec) shrinks
   them by ~15×. The assembly step
   (``prep.stage_2a_to_lerobot._assemble``) is what produces the final
   compressed layout; until it runs, expect the uncompressed staging cost.

## Per-modality cost per episode

Numbers assume the canonical preprocessing defaults: RGB resized to 378×378,
depth at 192×192 ``uint16``, DINOv3 ViT-L fp16 cache at 5 Hz with
65 tokens (64 patches + [CLS]).

Symbols: ``T`` = native episode length (frames at native fps).

| File                              | Dtype  | Per-frame bytes | Notes                              |
| --------------------------------- | ------ | --------------- | ---------------------------------- |
| ``camera_<cam>.npy``              | uint8  | 378·378·3 ≈ 429 KB | one per camera per episode      |
| ``depth_<cam>.npy``               | uint16 | 192·192·2 ≈ 72 KB | one per camera per episode       |
| ``state.npy``                     | fp32   | 200 B           | 50-D padded proprio                |
| ``action_native.npy``             | fp32   | 128 B           | 32-D padded                        |
| ``action_canonical_ee.npy``       | fp32   | 28 B            | 7-D                                |
| ``timestamps.npy``                | fp32   | 4 B             |                                    |
| ``meta.json``                     | text   | ~2 KB           | (per episode, not per frame)       |
| ``features/<cam>/<mod>/*.sft``    | fp16   | T·(5/fps)·65·1024·2 ≈ 133 KB·(5/fps)·T | per camera × {rgb, depth} |

## Per-chunk peak (256 episodes)

A chunk's directory hits its maximum size when stage_4 has just finished
(all stage outputs are present, no cleanup has run). Per-camera cost
dominates; per-episode totals scale linearly with the average episode
length and the number of cameras.

### DROID
- 1 camera (``head_rgb`` only; wrist disabled by default in ``DROID_CAMERA_MAP``)
- 15 Hz, average ~150 frames per episode (10 s)
- Per episode: 150·429 KB (RGB) + 150·72 KB (depth) + ~7 MB (DINO) ≈ 82 MB
- **Per chunk: 256·82 MB ≈ 21 GiB**

### AgiBot 2026
- 3 cameras (``head_rgb``, ``wrist_rgb_left``, ``wrist_rgb_right``)
- 30 Hz, average ~300 frames per episode (10 s)
- Per episode: 300·429·3 + 300·72·3 + 3·~7 MB (DINO) ≈ 251 MB
- **Per chunk: 256·251 MB ≈ 63 GiB**

### RoboMIND
- 3 cameras (``head_rgb``, ``wrist_rgb_left``, ``wrist_rgb_right``)
- 25 Hz, average ~250 frames per episode
- Per episode: 250·429·3 + 250·72·3 + 3·~7 MB ≈ 175 MB
- **Per chunk: 256·175 MB ≈ 44 GiB**

### Bridge V2
- 2 cameras (``head_rgb``, optional ``wrist_rgb``; assume both)
- 5 Hz, average ~50 frames per episode
- Per episode: 50·429·2 + 50·72·2 + 2·~7 MB ≈ 30 MB
- **Per chunk: 256·30 MB ≈ 7.5 GiB**

## Automatic cleanup (default behavior)

The orchestrator deletes intermediates as its second-to-last step in
``run_chunk`` (after stage_5 succeeds, before writing
``.pipeline_complete``). Removed:

* ``ep_*/depth_<cam>.npy`` — raw depth (the DINO depth modality already
  encodes it; training does not read the raw npy).
* ``ep_*/depth_<cam>.json`` — sidecar (low_quality / source flag).
* ``done/<hash>.ok`` — per-episode CheckpointedJob markers; redundant once
  ``.pipeline_complete`` exists.
* ``_scratch/`` — CheckpointedJob's working area.

Kept:

* ``ep_*/camera_<cam>.npy`` — RGB (uint8 staging frames).
* ``ep_*/{action_native,action_canonical_ee,state,timestamps}.npy``.
* ``ep_*/meta.json`` — instructions, masks, raw_meta.
* ``<dataset_root>/<cam>/<mod>/chunk-*/file-*.safetensors`` — Tri-DINO fp16
  features for both rgb and depth modalities.

The marker payload records ``cleanup_files_removed`` and ``cleanup_bytes_freed``
so you can audit how much was reclaimed per chunk:

```bash
jq '{chunk, cleanup_bytes_freed, cleanup_files_removed}' \
    $USAM_SCRATCH/staged/droid/chunk-*/.pipeline_complete
```

Pass ``--no-cleanup`` to ``prep.run_pipeline`` to skip cleanup for debugging.

## Further reclaim

If you still need more space:

* Cheap: ``rm -rf <output_root>/<dataset>/chunk-NNN/ep_*/camera_*.npy`` once
  you've assembled the LeRobot v2.1 layout with mp4-compressed RGB. The
  saved npy is the largest single contributor.
* Aggressive: ``rm -rf <output_root>/<dataset>/chunk-NNN/`` after a chunk's
  outputs have been mirrored elsewhere (HF Hub, archive volume). This wipes
  the ``.pipeline_complete`` marker too, so the orchestrator would re-run
  the chunk on next launch unless you preserve the marker.

## Minimum scratch budget per active job

The 4 sbatch jobs run in parallel and each holds an exclusive node. Their
in-flight peak (the chunk being processed plus any complete chunks still on
disk for that dataset) is what matters for ``df -h``:

* DROID: ~21 GiB per chunk in-flight; cumulative as new chunks accumulate.
* AgiBot: ~63 GiB per chunk in-flight.
* RoboMIND: ~44 GiB per chunk in-flight.
* Bridge: ~7.5 GiB per chunk in-flight.

If ``$USAM_SCRATCH`` is the same volume for all 4 datasets, the cumulative
"all chunks for all 4 datasets" is the upper-bound budget — see the TL;DR
table. If your scratch is smaller than that, partition per-dataset to
separate volumes via symlinks:

```bash
mkdir -p /lustre/scratch/$USER/usam_droid /lustre/scratch/$USER/usam_agibot
ln -s /lustre/scratch/$USER/usam_droid  $USAM_SCRATCH/staged/droid
ln -s /lustre/scratch/$USER/usam_agibot $USAM_SCRATCH/staged/agibot2026
```

## Sanity checks

* The script ``scripts/prep_run_local.sh --dataset droid --output-root /tmp/usam --max-chunks 1`` writes ~21 GiB for DROID's first chunk — use this to
  confirm your scratch path has the right permissions before launching the
  real job.
* On a successfully finished chunk, ``<chunk-NNN>/.pipeline_complete`` records
  ``n_processed`` and ``elapsed_seconds``; multiply by the projected chunk
  count to extrapolate total runtime.

## Footnotes

* DINO cache per-chunk peak above assumes the chunk-id bug in
  ``encode_chunk`` is unfixed (it writes every chunk's safetensors into
  ``<output_root>/<cam>/<mod>/chunk-000/``). Pre-existing; not in scope for
  the unified pipeline change.
* "Per chunk" excludes the tiny ``meta.json`` and ``.pipeline_complete``
  files (KB-scale). ``done/<hash>.ok`` markers are also negligible.
* All numbers are *uncompressed staging*. Final compressed parquet+H.264+HEVC
  layout (post-``_assemble``) is ~10-15× smaller for RGB, 3-5× smaller for
  depth, and unchanged for DINO safetensors.
