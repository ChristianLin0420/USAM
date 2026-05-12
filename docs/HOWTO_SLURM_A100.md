# HOWTO — Slurm A100 data preprocessing (T1)

This guide covers running the USAM Phase A pipeline on the T1 tier
(8×A100 per job, 4 h preemptible windows). The plan reference is
[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §10 (Slurm
integration); the operational runbook lives at
[`slurm/README.md`](../slurm/README.md).

---

## 1. Build the prep image

```bash
docker build -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .
singularity build usam_prep.sif docker-daemon://usam:prep-a100
```

The prep image installs `requirements/base.txt` + `requirements/prep.txt`
and explicitly **does not** install `flash-attn` (not needed for prep)
or `transformer-engine` (no FP8 hardware on A100). It also enables
`HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_HIGH_PERFORMANCE=1` for fast
Hub downloads.

Copy `usam_prep.sif` to a stable path the Slurm batch script can find,
typically `~/usam_prep.sif`.

---

## 2. Where the upload daemon lives

**The CommitScheduler runs on a login node, NEVER inside Slurm.**

Reasons (cited verbatim from
[`slurm/README.md`](../slurm/README.md) §"Where the upload daemon lives"):

1. The login node has a stable network identity. Slurm jobs are
   short-lived and each requeue would restart the scheduler from
   scratch, breaking its internal "last commit" state.
2. The scheduler holds an HF API token in memory. Tokens leak less
   when bound to a single shell on a known machine.
3. Slurm GPU jobs should not waste their walltime on uploads. The
   upload daemon is IO-bound and needs no GPU.

Operational pattern (run in tmux on the login node so it persists
across SSH sessions):

```bash
tmux new -s usam-uploader
python -m prep.stage_6_upload \
  --watch /scratch/$USER/usam \
  --repo-prefix <org>/usam-
# Detach with Ctrl-b d
```

`<org>` is the team's HF organization name (e.g. `usam-team`); the
upload daemon picks up new chunks and pushes them to
`<org>/usam-<source>` repos.

---

## 3. Submit jobs

The universal Slurm template signature is:

```bash
sbatch slurm/job.sbatch <stage_module> <source> <chunk> [extra args...]
```

* `<stage_module>` is the Python module path under `prep.`, e.g.
  `stage_2a_to_lerobot.droid`. The job runs
  `python -m prep.<stage_module> --source <source> --chunk <chunk> --resume`.
* `<source>` is one of `droid`, `agibot2026`, `rh20t`, `robomind`,
  `bridge`.
* `<chunk>` is a non-negative integer (0-indexed).

Example: convert chunk 7 of DROID:

```bash
sbatch slurm/job.sbatch stage_2a_to_lerobot.droid droid 7
```

There is no `scripts/prep_submit_slurm.sh` shipped today — the team-lead
plans to add a thin wrapper around the dispatcher (see §5 below) but
direct `sbatch` invocation is the binding contract until it lands.

Logs land at `${USAM_REPO}/logs/usam-<jobid>.{out,err}`.

---

## 4. Preemption + requeue flow

The plumbing is summarized in
[`slurm/README.md`](../slurm/README.md) §"How preemption + requeue works".

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
            │     │   if _stop_requested ─▶ flush, sys.exit(124)
            ▼
            wait $PYPID
            │
   ┌────────┴────────┐
   │                 │
EXIT == 124      EXIT != 124
   │                 │
   ▼                 ▼
 scontrol requeue   exit EXIT
```

Key invariants:

* The `B:` prefix in `--signal=B:USR1@600` is mandatory — without it,
  only `srun`'s direct children get the signal and our nested
  `singularity exec` would miss it.
* `prep._base.CheckpointedJob.run` (`prep/_base.py:350-390`) does not
  raise from inside the signal handler; it sets a flag and checks it
  between episodes. This avoids the classic "exception swallowed by
  the converter library" failure.
* Idempotency is **per-episode, not per-shard**. The marker file
  `<chunk_dir>/done/<hash>.ok` is the single source of truth. Only after
  a shard write succeeds do we touch its episodes' markers.
* Exit code `124` is the agreed-upon "please requeue" code (see
  `prep._base.PREEMPT_EXIT_CODE`). Any other non-zero exit is a real
  failure: the bash wrapper does **not** requeue, Slurm marks the job
  FAILED, and the dispatcher retries later with exponential backoff.

---

## 5. Dispatcher and `MAX_PENDING`

The dispatcher (`prep/dispatch.py`) keeps the queue at most
**`MAX_PENDING=64`** USAM jobs at a time. This default is conservative;
most T1 sites cap a single user at 100–200 concurrent jobs and we want
headroom for other users. Tunable via the CLI flag:

```bash
python -m prep.dispatch --max-pending 96
```

The dispatcher polls every 60 s, queries `squeue -u $USER -h -o %i` to
count USAM jobs, and only submits when the count is below the cap. It
reads `manifests/<source>__<stage>.parquet` to find chunks whose
dependencies are `done` and whose own status is `pending`. Run it as a
long-lived process (e.g. inside the same tmux session as the upload
daemon).

---

## 6. HF Hub upload contracts

Hard rules enforced by `prep/_hub.py:50-160`:

* **`Dataset.push_to_hub` is forbidden.** Always use
  `huggingface_hub.upload_large_folder` (or the
  `CommitScheduler` for incremental uploads). Calling
  `prep._hub.reject_push_to_hub` raises a `RuntimeError` with a pointer
  to the right API.
* **≤ 1000 files per chunk.** Enforced by `MAX_FILES_PER_CHUNK = 1000`
  in `prep/_hub.py:50`.
* **≤ 5 GiB per file.** Enforced by `MAX_BYTES_PER_FILE = 5 * 1024**3`.
* `prep._hub.validate_chunk(folder)` runs the pre-flight check; never
  upload a chunk without a successful `ChunkValidation(ok=True)`.
* The Slurm jobs only write to local scratch. They do not talk to the
  Hub. The login-node `CommitScheduler` reconciles the chunk dirs back
  to the Hub.

---

## 7. Required environment variables

Set these on the login node (in `~/.bashrc` or a per-cluster `.envrc`):

| Variable | Required | Purpose |
|---|---|---|
| `USAM_REPO` | yes | path to USAM checkout |
| `USAM_SIF` | yes | path to `usam_prep.sif` Singularity image |
| `HUGGINGFACE_TOKEN` | login-node only | HF token; only the upload daemon reads it |
| `USAM_SCRATCH` | no | scratch root (default `/scratch/$USER/usam`) |
| `USAM_HF_HOME` | no | HF cache (default `$USAM_SCRATCH/hf_cache`) |
| `USAM_LOG_LEVEL` | no | Python log level (default `INFO`) |

`slurm/env.sh` reads these and exports the derived values
(`HF_HOME`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`) to every Slurm
allocation.

---

## 8. Pre-flight checklist

Before launching a large submission:

1. `singularity exec $USAM_SIF python -c "import prep, usam"` succeeds.
2. `sbatch --test-only slurm/job.sbatch <stage> <source> 0` reports a
   sane start time.
3. `python -m prep._hub --validate /scratch/$USER/usam/<source>/<stage>/chunk-000`
   reports `ok=True` on at least one fully-converted chunk.
4. The CommitScheduler is running in tmux on the login node (`tmux ls`
   and `tail -f` its log).

If any of those fail, fix before submitting more jobs — Phase A
preemption recovery is robust but a misconfigured environment will
chew through Slurm walltime making no forward progress.
