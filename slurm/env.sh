# SPDX-License-Identifier: MIT
# shellcheck shell=bash
#
# slurm/env.sh — environment bootstrap for USAM Phase A Slurm jobs.
#
# Sourced by slurm/job.sbatch on every job launch. Also useful for
# interactive `salloc` debug sessions:
#   $ salloc --gres=gpu:a100:1 --time=00:30:00
#   $ source slurm/env.sh
#
# # Required environment variables (caller must export these in ~/.bashrc
# # or pass via sbatch --export=ALL,FOO=bar)
#
#   USAM_REPO         absolute path to the USAM checkout on the cluster
#                     (e.g. /home/<user>/USAM)
#   USAM_SIF          absolute path to the prep Singularity image
#                     (e.g. /home/<user>/usam_prep.sif)
#   HUGGINGFACE_TOKEN HF API token (read from a file in ~/.cache, never
#                     committed to the repo). Only needed on login nodes
#                     running the CommitScheduler — Slurm jobs themselves
#                     do not talk to the Hub.
#
# # Optional
#
#   USAM_SCRATCH      where stage outputs land before upload
#                     (default: /scratch/${USER}/usam)
#   USAM_HF_HOME      cache root for HF datasets/models
#                     (default: ${USAM_SCRATCH}/hf_cache)
#   USAM_LOG_LEVEL    python logging level (default: INFO)

set -u

# ---------------------------------------------------------------------------
# Module loads
# ---------------------------------------------------------------------------
# Different sites have different module systems; gracefully no-op if `module`
# is not defined. Add cluster-specific loads here as needed.
if command -v module >/dev/null 2>&1; then
    module purge          || true
    module load cuda/12.4 || true
    module load singularity || true
fi

# ---------------------------------------------------------------------------
# Conda activation (only if a conda env is needed outside the .sif)
# ---------------------------------------------------------------------------
# In production we run inside the Singularity image, so this is rarely needed.
# Set $USAM_CONDA_ENV to "<name>" to force-activate.
if [[ -n "${USAM_CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck source=/dev/null
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${USAM_CONDA_ENV}"
    else
        echo "[env.sh] WARNING: USAM_CONDA_ENV='${USAM_CONDA_ENV}' set but conda not on PATH" >&2
    fi
fi

# ---------------------------------------------------------------------------
# Required path checks
# ---------------------------------------------------------------------------
: "${USAM_REPO:?USAM_REPO must be set (path to the USAM checkout)}"
: "${USAM_SIF:?USAM_SIF must be set (path to usam_prep.sif)}"

# ---------------------------------------------------------------------------
# Scratch and cache directories
# ---------------------------------------------------------------------------
export USAM_SCRATCH="${USAM_SCRATCH:-/scratch/${USER}/usam}"
export USAM_HF_HOME="${USAM_HF_HOME:-${USAM_SCRATCH}/hf_cache}"
mkdir -p "${USAM_SCRATCH}" "${USAM_HF_HOME}"

export HF_HOME="${USAM_HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${USAM_HF_HOME}/hub"
export HF_DATASETS_CACHE="${USAM_HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${USAM_HF_HOME}/transformers"

# ---------------------------------------------------------------------------
# HF transfer / xet — fast Hub IO. These are imported by prep/_hub.py.
# ---------------------------------------------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# ---------------------------------------------------------------------------
# Python runtime knobs
# ---------------------------------------------------------------------------
# OMP_NUM_THREADS=1 prevents BLAS oversubscription with multiple DataLoader
# workers; PYTHONUNBUFFERED=1 makes prints visible immediately in slurm logs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export USAM_LOG_LEVEL="${USAM_LOG_LEVEL:-INFO}"

# Ensure prep.* and usam.* are importable from inside the singularity image
# even when the user's working dir is somewhere else.
export PYTHONPATH="${USAM_REPO}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Echo for debuggability
# ---------------------------------------------------------------------------
echo "[env.sh] USAM_REPO=${USAM_REPO}"
echo "[env.sh] USAM_SIF=${USAM_SIF}"
echo "[env.sh] USAM_SCRATCH=${USAM_SCRATCH}"
echo "[env.sh] HF_HOME=${HF_HOME}"
echo "[env.sh] OMP_NUM_THREADS=${OMP_NUM_THREADS}"

set +u
