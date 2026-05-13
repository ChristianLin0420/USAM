#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Run the unified USAM Phase A pipeline locally for ONE dataset on the dev box.
# This is the same Python entry the Slurm sbatch files use
# (``slurm/pipeline_<dataset>.sbatch``) — running it locally is the simplest
# way to validate a config / converter / depth model against real raw data.
#
# Stages run in-process per chunk: 2a -> 2c -> 3 -> 4 -> 5 (validate).
# Resume: ``<output_root>/<dataset>/chunk-NNN/.pipeline_complete`` markers.
#
# Usage:
#   bash scripts/prep_run_local.sh --dataset droid \
#        --output-root /scratch/usam/staged \
#        [--config configs/data/droid.yaml] \
#        [--raw-root /scratch/usam/droid/raw] \
#        [--start-chunk 0] [--max-chunks 1]
#
# To do a stage-0 raw download first, run ``python -m prep.stage_0_download.<dataset>``
# manually (the orchestrator only handles stages 2a..5).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET=""
OUTPUT_ROOT=""
CONFIG=""
RAW_ROOT=""
START_CHUNK=0
MAX_CHUNKS=""
NUM_WORKERS_2A=8
NUM_GPUS=0
WORKERS_PER_GPU=1
DINOV3_CKPT="${USAM_DINOV3_CKPT:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
DA3_CKPT="${USAM_DA3_CKPT:-depth-anything/DA3MONO-LARGE}"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)         DATASET="$2"; shift 2 ;;
    --source)          DATASET="$2"; shift 2 ;;  # legacy alias
    --output-root|--out) OUTPUT_ROOT="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --raw-root|--raw)  RAW_ROOT="$2"; shift 2 ;;
    --start-chunk|--chunk) START_CHUNK="$2"; shift 2 ;;
    --max-chunks)      MAX_CHUNKS="$2"; shift 2 ;;
    --num-workers-2a)  NUM_WORKERS_2A="$2"; shift 2 ;;
    --num-gpus)        NUM_GPUS="$2"; shift 2 ;;
    --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
    --dinov3-ckpt)     DINOV3_CKPT="$2"; shift 2 ;;
    --da3-ckpt)        DA3_CKPT="$2"; shift 2 ;;
    --python)          PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$DATASET" ]] && { echo "--dataset is required" >&2; exit 2; }
[[ -z "$OUTPUT_ROOT" ]] && { echo "--output-root is required" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"

ARGS=(
  --dataset "$DATASET"
  --output-root "$OUTPUT_ROOT"
  --start-chunk "$START_CHUNK"
  --num-workers-2a "$NUM_WORKERS_2A"
  --num-gpus "$NUM_GPUS"
  --workers-per-gpu "$WORKERS_PER_GPU"
  --dinov3-ckpt "$DINOV3_CKPT"
  --da3-ckpt "$DA3_CKPT"
  --resume
)
[[ -n "$CONFIG" ]]     && ARGS+=(--config "$CONFIG")
[[ -n "$RAW_ROOT" ]]   && ARGS+=(--raw-root "$RAW_ROOT")
[[ -n "$MAX_CHUNKS" ]] && ARGS+=(--max-chunks "$MAX_CHUNKS")

echo "[prep_run_local] dataset=$DATASET output_root=$OUTPUT_ROOT start_chunk=$START_CHUNK"
exec "$PYTHON" -m prep.run_pipeline "${ARGS[@]}"
