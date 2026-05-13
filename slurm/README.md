# USAM Slurm operational guide

Phase A pipeline runs on the T1 tier: 8×A100, preemptible, 4 h walltime.
This directory holds the universal job template (`job.sbatch`), the
unified per-dataset pipeline template (`pipeline.sbatch.tmpl`) and its 4
rendered sbatch files, the env bootstrap (`env.sh`), and this runbook.

## TL;DR — recommended path: 4 parallel per-dataset sbatch files

```bash
# One-time per cluster setup
export USAM_REPO=/home/$USER/USAM
export USAM_SIF=/home/$USER/usam_prep.sif
export USAM_SCRATCH=/lustre/.../$USER/usam   # plenty of space for staged outputs

# (Re-)generate the 4 rendered sbatch files from the template.
bash scripts/prep_render_sbatch.sh

# Submit one job per dataset in parallel. Each one runs the full
# stage-2a → 2c → 3 → 4 → 5 pipeline for that dataset, requeuing itself
# every 4 h until all chunks land on disk.
for d in droid agibot2026 robomind bridge; do
    sbatch slurm/pipeline_${d}.sbatch
done

# Watch progress.
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
ls $USAM_SCRATCH/staged/droid/chunk-*/.pipeline_complete | wc -l
```

The output of each dataset lands at `$USAM_SCRATCH/staged/<dataset>/chunk-NNN/`
with the LeRobot v2.1 layout (`data/`, `videos/`, `features/`) plus
per-episode `ep_<hash>/` staging dirs and a final
`.pipeline_complete` JSON marker. Nothing is uploaded to HF Hub by this
pipeline — see the legacy section below for the optional Hub path.

## Resume semantics (local-only)

Resume is driven entirely by on-disk markers:

* `chunk-NNN/.pipeline_complete` — written atomically as the last step of
  `PipelineOrchestrator.run_chunk` after stage_5 validation passes. The
  orchestrator skips any chunk whose marker exists.
* `chunk-NNN/done/<hash>.ok` (inherited from `CheckpointedJob`) — per-episode
  marker inside stage_2a. Guards against partial-chunk loss if SIGUSR1 lands
  mid-stage-2a.

On Slurm requeue (after a 4 h timeout or pre-walltime SIGUSR1), the orchestrator
re-scans markers and resumes from the first un-complete chunk. There is no HF
round-trip, no `.upload_state.json` to check, no login-node daemon to keep
alive.

## How preemption + requeue works

```
   sbatch slurm/pipeline_<ds>.sbatch ─▶ allocates 8×A100 for 4h
              │
              ▼
   bash pipeline_<ds>.sbatch
              │   #SBATCH --signal=B:USR1@600
              │   #SBATCH --requeue
              │   #SBATCH --dependency=singleton
              │
              ├─▶ timeout 3.95h srun python -m prep.run_pipeline --dataset <ds> &
              │     │
              │     ▼
              │   PipelineOrchestrator.run()
              │     ├── installs SIGUSR1 handler
              │     ├── per-chunk loop
              │     │     ├── stage_2a -> 2c -> 3 -> 4 -> 5
              │     │     └── write .pipeline_complete
              │     │
              │     ▼
              │   if _stop_requested ─▶ return PREEMPT_EXIT_CODE (124)
              │
              ▼
              wait $PYPID
              │
   ┌──────────┴──────────┐
   │                     │
 EXIT in {124, 137}    EXIT == 0          EXIT not in {0,124,137}
   │                     │                  │
   ▼                     ▼                  ▼
 scontrol requeue      exit 0             exit EXIT
   │                     │                  │
   ▼                     ▼                  ▼
 Slurm resubmits        all chunks done    Slurm marks FAILED
 the same job id        (job done)         (alert humans)
```

Key invariants:

* Slurm sends `SIGUSR1` to the **batch shell** 600 s before walltime, because
  of the `B:` prefix in `--signal=B:USR1@600`. Without `B:`, only `srun`'s
  immediate children get signalled.
* `PipelineOrchestrator._on_sigusr1` does not raise; it flips
  `self._stop_requested` and `run` polls it between chunks. The in-flight
  chunk gets to finish, write its `.pipeline_complete`, and only then the
  process exits 124.
* `--dependency=singleton` keeps at most one job per dataset in the queue at
  a time, so requeue can't race the current job.
* Exit code 124 means "graceful preempt; please requeue". Exit code 137
  means SIGKILL (Slurm hit the hard walltime); we requeue on that too as a
  safety net. Any other non-zero exit is a real failure.

## Submitting jobs (legacy per-stage path)

The original `slurm/job.sbatch` template is preserved for ad-hoc per-stage
runs and for the integration tests that target a single stage. Signature:

```bash
sbatch slurm/job.sbatch <stage_module> <dataset> <chunk> [extra args...]
```

* `<stage_module>` is the Python module path under `prep.`, e.g.
  `stage_2a_to_lerobot.droid`. The job runs
  `python -m prep.<stage_module> --dataset <dataset> --chunk <chunk> --resume`.
* `<dataset>` is one of `droid`, `agibot2026`, `robomind`, `bridge`.
* `<chunk>` is a non-negative integer.

Logs land in `${USAM_REPO}/logs/<job-name>-<jobid>/std{out,err}.log` for the
new pipeline sbatch files, and `${USAM_REPO}/logs/usam-<jobid>.{out,err}` for
the legacy `job.sbatch`.

## Optional: HF upload (legacy)

The pipeline writes everything locally and does NOT upload to HF Hub. If you
do want to mirror outputs to the Hub, the existing tooling in `prep/_hub.py`
and `prep/stage_6_upload.py` is unchanged — start a long-lived
`CommitScheduler` on a login node (`python -m prep.stage_6_upload --watch
$USAM_SCRATCH/staged/<dataset> --repo-prefix <org>/usam-`) and it will pick
up new files. This path is not exercised by the per-dataset sbatch files.

## Required environment variables

Set these in `~/.bashrc` or a per-cluster `.envrc`:

| Variable                | Required | Purpose                                       |
| ----------------------- | -------- | --------------------------------------------- |
| `USAM_REPO`             | yes      | path to USAM checkout                         |
| `USAM_SIF`              | yes (for legacy `job.sbatch`) | Singularity image path     |
| `USAM_SCRATCH`          | yes      | scratch root for staged outputs               |
| `USAM_HF_HOME`          | no       | HF cache (default `$USAM_SCRATCH/hf_cache`)   |
| `USAM_PYTHON`           | no       | python interpreter inside the sbatch          |
| `USAM_NUM_WORKERS_2A`   | no       | stage_2a CPU worker count (default 8)         |
| `USAM_NUM_GPUS`         | no       | stage_2c/4 GPU count (default 0 = auto)       |
| `USAM_WORKERS_PER_GPU`  | no       | stage_2c/4 oversubscription (default 1)       |
| `USAM_DINOV3_CKPT`      | no       | DINOv3 checkpoint id/path                     |
| `USAM_DA3_CKPT`         | no       | Depth-Anything-V3 checkpoint id/path          |
| `USAM_CONDA_ENV`        | no       | conda env to activate outside Singularity     |
| `USAM_LOG_LEVEL`        | no       | python log level (default INFO)               |

## Estimating local space

Quick summary; full per-modality breakdown is in
[`docs/DISK_SPACE_ESTIMATE.md`](../docs/DISK_SPACE_ESTIMATE.md).

| Dataset    | Episodes (≈) | Chunks (256/chunk) | Per-chunk peak | Per-dataset total |
| ---------- | ------------ | ------------------ | -------------- | ----------------- |
| DROID      |    76 000    |    297             |    22 GiB      |    6.4 TiB        |
| AgiBot2026 | 1 000 000    |  3 907             |    63 GiB      |   >100 TiB        |
| RoboMIND   |    30 000    |    118             |    44 GiB      |    4.0 TiB        |
| Bridge V2  |    60 000    |    235             |     7.5 GiB    |    1.6 TiB        |
| **Total**  |              |                    |                | **>110 TiB**      |

These assume uncompressed ``camera_*.npy`` staging (the default; assemble +
mp4 encoding shrinks RGB ~15×). For AgiBot, subset by embodiment if your
scratch can't hold the full uncompressed footprint.

## Troubleshooting

* **Job exits 124 but is not requeued.** Check that `scontrol requeue` ran
  in `logs/usam-prep-<ds>-<jobid>/stdout.log`. Some sites disable
  user-initiated requeue; ask the admin to enable `JobRequeue=1`.
* **Job killed at 4h with exit 137 instead of 124.** SIGKILL fired because
  SIGUSR1 was missed. Our wrapper treats 137 as "requeue" too. Confirm
  `--signal=B:USR1@600` is present.
* **Chunk re-runs unnecessarily.** Check that the previous run actually
  wrote `<chunk>/. pipeline_complete`. If stage_5 validation failed, the
  marker is intentionally absent and the chunk re-runs.
* **Out of disk on `$USAM_SCRATCH`.** The 4 datasets together can hit
  ~14 TiB. Process one dataset at a time, or symlink each dataset's
  `staged/<dataset>` to a separate volume.

## Who to contact

* Slurm template, env, signal handling, requeue: **pipeline-engineer**.
* Per-source converter behaviour (DROID, AgiBot, RoboMIND, Bridge):
  **data-engineer**.
* Singularity image build, pip pinning: **infra-engineer**.

## Pre-flight checklist before a large submission

1. `python -c "import prep.run_pipeline"` succeeds inside the runtime env.
2. `bash scripts/prep_render_sbatch.sh` regenerated the 4 sbatch files.
3. `sbatch --test-only slurm/pipeline_droid.sbatch` reports a sane start time.
4. `$USAM_SCRATCH` has at least the per-dataset budget free (see the table
   above). Resolve with `df -h $USAM_SCRATCH`.
5. Local dry-run worked: `bash scripts/prep_run_local.sh --dataset droid
   --output-root /tmp/usam --max-chunks 1`.
