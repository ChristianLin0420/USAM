#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Per-dataset preparation pipeline sbatch template.
# This file is the *source*; the 4 concrete files
# (``slurm/pipeline_{droid,agibot2026,robomind,bridge}.sbatch``) are rendered
# from this template by ``scripts/prep_render_sbatch.sh`` with the
# ``robomind`` placeholder replaced. Edit this file, not the rendered
# ones — the renderer regenerates them.
#
# Each rendered sbatch is independent: submit 4 of them (one per dataset) in
# parallel via:
#
#   for d in droid agibot2026 robomind bridge; do
#       sbatch slurm/pipeline_${d}.sbatch
#   done
#
# Each job:
#   - holds 8 GPUs for up to 4 hours
#   - runs ``python -m prep.run_pipeline --dataset <d>`` under timeout 3.95h
#   - on SIGUSR1@600 → graceful exit 124 → ``scontrol requeue``
#   - on exit 0      → dataset finished, no requeue
#   - on other exit  → real failure; Slurm marks FAILED and does NOT requeue
#
# Resume comes from the ``.pipeline_complete`` markers under
# ``$USAM_SCRATCH/<dataset>/chunk-NNN/``: on requeue the orchestrator scans
# them and starts from the first un-complete chunk.
#
#SBATCH --job-name=usam-prep-robomind
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=./logs/%x-%j/stdout.log
#SBATCH --error=./logs/%x-%j/stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

DATASET="robomind"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${USAM_REPO}/logs"

# ---------------------------------------------------------------------------
# SIGUSR1 forwarding
# ---------------------------------------------------------------------------
# Slurm fires SIGUSR1 at the bash batch process 600 s before walltime.
# Forward it to the python child PID so PipelineOrchestrator's handler runs.
PYPID=""
preempt_handler() {
    if [[ -n "${PYPID}" ]] && kill -0 "${PYPID}" 2>/dev/null; then
        echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] caught SIGUSR1; forwarding to PID ${PYPID}"
        kill -USR1 "${PYPID}" 2>/dev/null || true
    else
        echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] caught SIGUSR1; no python child running"
    fi
}
trap preempt_handler USR1

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] launching dataset=${DATASET}"
echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] SLURM_JOB_ID=${SLURM_JOB_ID:-?} node=$(hostname)"

OUTPUT_ROOT="${USAM_SCRATCH}/staged"

# The 3.95h timeout fires ~3 minutes before SIGUSR1@600 in case Slurm's signal
# is delayed; in either case the orchestrator's chunk-boundary check exits 124.
timeout 3.95h \
    srun --jobid="${SLURM_JOB_ID}" \
    "${USAM_PYTHON:-python}" -u -m prep.run_pipeline \
        --dataset "${DATASET}" \
        --output-root "${OUTPUT_ROOT}" \
        --num-workers-2a "${USAM_NUM_WORKERS_2A:-8}" \
        --num-gpus "${USAM_NUM_GPUS:-0}" \
        --workers-per-gpu "${USAM_WORKERS_PER_GPU:-1}" \
        --dinov3-ckpt "${USAM_DINOV3_CKPT:-facebook/dinov3-vitl16-pretrain-lvd1689m}" \
        --da3-ckpt "${USAM_DA3_CKPT:-depth-anything/DA3MONO-LARGE}" \
        --resume &
PYPID=$!

set +e
wait "${PYPID}"
EXIT=$?
set -e

echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] python child exited with ${EXIT}"

if [[ ${EXIT} -eq 124 ]] || [[ ${EXIT} -eq 137 ]]; then
    # 124 = orchestrator's graceful preempt; 137 = SIGKILL from timeout.
    echo "[$(date -Is)] [pipeline_${DATASET}.sbatch] preempt/timeout exit; requesting requeue of ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

exit "${EXIT}"
