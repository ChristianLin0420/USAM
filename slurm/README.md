# USAM Slurm operational guide

Phase A pipeline runs on the T1 tier: 8×A100, preemptible, 4 h walltime.
This directory holds the universal job template (`job.sbatch`), the env
bootstrap (`env.sh`), and the operational runbook (this file).

## TL;DR

```bash
# One-time per cluster setup
export USAM_REPO=/home/$USER/USAM
export USAM_SIF=/home/$USER/usam_prep.sif
export HUGGINGFACE_TOKEN=$(cat ~/.cache/huggingface/token)

# Submit one chunk
sbatch slurm/job.sbatch stage_2a_to_lerobot.droid droid 7

# Submit many — let the dispatcher manage them (Phase 2)
python -m prep.dispatch --max-pending 64
```

## Where the upload daemon lives

**The CommitScheduler (`prep._hub.make_commit_scheduler`) MUST run on a
login node, not inside a Slurm job.** Reasons:

1. The login node has a stable network identity. Slurm jobs are short-lived
   and each requeue would restart the scheduler from scratch, breaking its
   internal "last commit" state.
2. The scheduler holds an HF API token in memory. Tokens leak less when
   bound to a single shell on a known machine.
3. Slurm GPU jobs should not waste their walltime on uploads. The upload
   daemon is IO-bound and needs no GPU.

Operational pattern:

```bash
# Run in tmux on the login node, persists across SSH sessions.
tmux new -s usam-uploader
python -m prep.stage_6_upload --watch /scratch/$USER/usam --repo-prefix <org>/usam-
# Detach: Ctrl-b d
```

## How preemption + requeue works

```
   sbatch ─▶ Slurm scheduler ─▶ allocates 8×A100 for 3h55m
              │
              ▼
   bash job.sbatch
              │   #SBATCH --signal=B:USR1@600
              │   #SBATCH --requeue
              │
              ├─▶ singularity exec ... python -m prep.<stage> &
              │     │
              │     ▼
              │   prep._base.CheckpointedJob.run()
              │     ├── installs SIGUSR1 handler
              │     ├── per-episode loop
              │     │     │
              │     │     ▼
              │     │   between episodes: if _stop_requested ─▶ flush, sys.exit(124)
              │
              ▼
              wait $PYPID
              │
   ┌──────────┴──────────┐
   │                     │
 EXIT == 124          EXIT != 124
   │                     │
   ▼                     ▼
 scontrol requeue       exit EXIT
   │                     │
   ▼                     ▼
 Slurm resubmits        end (Slurm marks as completed/failed
 the same job id        based on exit code)
   │
   ▼
 next allocation reuses the
 SAME chunk dir, sees the
 done/<hash>.ok markers, skips
 already-finished episodes.
```

Key invariants:

* Slurm sends `SIGUSR1` to the **batch shell** 600 s before walltime, because
  of the `B:` prefix in `--signal=B:USR1@600`. Without `B:`, only `srun`'s
  immediate children get signalled and our nested `singularity exec` would
  miss it.
* `prep._base.CheckpointedJob.run` does not raise from the signal handler;
  it sets a flag and checks it between episodes. This avoids the
  classic "exception swallowed by the converter library" failure mode.
* Idempotency is per-episode, not per-shard. The marker file
  `<chunk_dir>/done/<hash>.ok` is the single source of truth. Only after a
  shard write succeeds do we touch its episodes' markers.
* Exit code `124` is the agreed-upon "please requeue" code. Any other
  non-zero exit is a real failure: the bash wrapper does **not** requeue,
  Slurm marks the job FAILED, and the dispatcher will retry later (with an
  exponential backoff schedule).

## Submitting jobs

The template signature is:

```bash
sbatch slurm/job.sbatch <stage_module> <dataset> <chunk> [extra args...]
```

* `<stage_module>` is the Python module path under `prep.`, e.g.
  `stage_2a_to_lerobot.droid`. The job runs
  `python -m prep.<stage_module> --dataset <dataset> --chunk <chunk> --resume`
  (Wave F: one A100 node per dataset).
* `<dataset>` is one of `droid`, `agibot2026`, `rh20t`, `robomind`,
  `bridge`, `oxe_auge`.
* `<chunk>` is a non-negative integer.

Logs land in `${USAM_REPO}/logs/usam-<jobid>.{out,err}`.

## MAX_PENDING and dispatcher behaviour

The dispatcher (`prep/dispatch.py`, Phase 2) keeps the queue at most
**`MAX_PENDING=64`** USAM jobs at a time. The default is conservative: most
T1 sites cap a single user at 100–200 concurrent jobs and we want headroom
for other users. Override with:

```bash
python -m prep.dispatch --max-pending 96
```

The dispatcher polls every 60 s, queries `squeue -u $USER -h -o %i` to count
USAM jobs, and only submits when the count is below the cap. It reads
`manifests/<source>__<stage>.parquet` to find chunks whose dependencies are
`done` and whose own status is `pending`.

## Required environment variables

Set these in `~/.bashrc` or a per-cluster `.envrc`:

| Variable            | Required | Purpose                                       |
| ------------------- | -------- | --------------------------------------------- |
| `USAM_REPO`         | yes      | path to USAM checkout                         |
| `USAM_SIF`          | yes      | path to `usam_prep.sif` Singularity image     |
| `HUGGINGFACE_TOKEN` | login only | HF token; only the upload daemon reads it   |
| `USAM_SCRATCH`      | no       | scratch root (default `/scratch/$USER/usam`)  |
| `USAM_HF_HOME`      | no       | HF cache (default `$USAM_SCRATCH/hf_cache`)   |
| `USAM_CONDA_ENV`    | no       | conda env to activate outside Singularity     |
| `USAM_LOG_LEVEL`    | no       | python log level (default INFO)               |

## Troubleshooting

* **Job exits 124 but is not requeued.** Check that `scontrol requeue` ran
  in `logs/usam-<jobid>.out`. Some sites disable user-initiated requeue;
  ask the admin to enable `JobRequeue=1`.
* **Job killed at 4h with exit 0 instead of 124.** The python process did
  not catch SIGUSR1 fast enough. Confirm `--signal=B:USR1@600` is in the
  sbatch header — without `B:` only srun children are signalled.
* **`huggingface_hub.errors.HfHubHTTPError 413`.** The chunk exceeded
  per-file size or per-folder file count. Check `prep._hub.validate_chunk`
  output — it logs the offending file/count and refuses the upload.
* **`Dataset.push_to_hub` raised RuntimeError.** Working as intended; see
  `prep._hub.reject_push_to_hub`. Use `upload_chunk_final` instead.

## Who to contact

* Slurm template, env, signal handling, requeue: **pipeline-engineer**
  (this file's owner).
* Per-source converter behaviour (DROID, AgiBot, RH20T, RoboMIND, Bridge,
  OXE-AugE): **data-engineer**.
* Singularity image build, pip pinning: **infra-engineer**.
* Anything routed via `team lead` if it crosses agents.

## Pre-flight checklist before a large submission

1. `singularity exec $USAM_SIF python -c "import prep, usam"` succeeds.
2. `sbatch --test-only slurm/job.sbatch <stage> <source> 0` reports a sane
   start time.
3. `python -m prep._hub --validate /scratch/$USER/usam/<source>/<stage>/chunk-000`
   reports `ok=True` on at least one fully-converted chunk.
4. The CommitScheduler is running in tmux on the login node (check with
   `tmux ls` and `tail -f` its log).
