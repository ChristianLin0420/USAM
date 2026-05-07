#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Universal preemptible Slurm template for USAM Phase A pipeline jobs.
#
# Invariants (do not weaken without coordinating with prep/_base.py):
#   --signal=B:USR1@600  : Slurm sends SIGUSR1 to the *batch* shell 10 min
#                          before walltime expires (the `B:` is mandatory --
#                          without it the signal goes to srun's children only).
#   --requeue            : on preempt or node failure, the job re-enters the
#                          queue automatically. Combined with the per-episode
#                          idempotency in prep/_base.py, no episode is lost.
#
# Exit code contract with prep/_base.py:
#   0   : successful completion of this chunk
#   124 : graceful preemption mid-run; request a requeue
#   other : real failure, do not requeue
#
# Usage (from a login node):
#   sbatch slurm/job.sbatch <stage> <source> <chunk> [extra args...]
# e.g.
#   sbatch slurm/job.sbatch stage_2a_to_lerobot.droid droid 7
#
#SBATCH --job-name=usam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8
#SBATCH --time=03:55:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# slurm/env.sh sets HF_HOME, PYTHONUNBUFFERED, OMP_NUM_THREADS, the path to
# $USAM_SIF (Singularity image), and $USAM_REPO (repo checkout).
# Users override anything they need via `export FOO=bar` before sbatch'ing.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${USAM_REPO}/logs"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <stage_module> <source> <chunk> [extra args...]" >&2
    echo "  e.g. $0 stage_2a_to_lerobot.droid droid 7" >&2
    exit 2
fi
STAGE="$1"
SOURCE="$2"
CHUNK="$3"
shift 3
EXTRA=("$@")

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
# Slurm fires SIGUSR1 at the bash batch process. We forward it to the python
# child PID, then wait for the child to exit (the child handles SIGUSR1 in
# prep._base.CheckpointedJob.run, flushes the in-flight episode, and exits 124).
# Once the child returns 124 we issue scontrol requeue and exit so Slurm
# re-queues us instead of marking the job as failed.

PYPID=""
preempt_handler() {
    if [[ -n "${PYPID}" ]] && kill -0 "${PYPID}" 2>/dev/null; then
        echo "[$(date -Is)] [job.sbatch] caught SIGUSR1; forwarding to PID ${PYPID}"
        kill -USR1 "${PYPID}" 2>/dev/null || true
    else
        echo "[$(date -Is)] [job.sbatch] caught SIGUSR1; no python child running"
    fi
}
trap preempt_handler USR1

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[$(date -Is)] [job.sbatch] launching stage=${STAGE} source=${SOURCE} chunk=${CHUNK}"
echo "[$(date -Is)] [job.sbatch] SLURM_JOB_ID=${SLURM_JOB_ID:-?} node=$(hostname)"

# We background the singularity invocation so the bash trap can fire on the
# parent shell. ``wait $PYPID`` then returns the child exit code, which we
# inspect to decide whether to requeue.
singularity exec --nv \
    --bind /scratch:/scratch \
    --bind /home:/home \
    "${USAM_SIF}" \
    bash -lc "cd ${USAM_REPO} && python -u -m prep.${STAGE} \
        --source ${SOURCE} --chunk ${CHUNK} --resume ${EXTRA[*]:-}" &
PYPID=$!

# `wait` is interruptible by signals; on SIGUSR1 the trap fires, the python
# child catches the forwarded signal, flushes, and exits 124. Then `wait`
# returns the child's exit code in $?.
set +e
wait "${PYPID}"
EXIT=$?
set -e

echo "[$(date -Is)] [job.sbatch] python child exited with ${EXIT}"

if [[ ${EXIT} -eq 124 ]]; then
    echo "[$(date -Is)] [job.sbatch] graceful preemption; requesting requeue of ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    # Slurm will reschedule us; exit 0 so this attempt is recorded as clean.
    exit 0
fi

exit "${EXIT}"
