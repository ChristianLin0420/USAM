#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Smoke training on a single 8x A40 node. Runs 100 steps on the tiny_droid
# golden fixture with the usam_350m_smoke config.
#
# Usage:
#   bash scripts/train_smoke_a40.sh                          # 8 GPU smoke (100 steps)
#   bash scripts/train_smoke_a40.sh --device cpu --max_steps 5  # CPU plumbing only
#   bash scripts/train_smoke_a40.sh --auto_oom_reduce        # halve bs on OOM
#
# The script auto-detects whether torchrun should fan out to multiple GPUs
# (it does so iff CUDA_VISIBLE_DEVICES isn't restricted to a single GPU).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ----- defaults --------------------------------------------------------------
TRAIN_CFG="configs/train/stage_b1_pretrain.yaml"
MODEL_CFG="configs/model/usam_350m_smoke.yaml"
DATA_DIR="tests/golden_data/tiny_droid"
MAX_STEPS=100
DEVICE="auto"
EXTRA_ARGS=()

# ----- arg parse -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         TRAIN_CFG="$2"; shift 2;;
    --model)          MODEL_CFG="$2"; shift 2;;
    --data)           DATA_DIR="$2"; shift 2;;
    --max_steps)      MAX_STEPS="$2"; shift 2;;
    --device)         DEVICE="$2"; shift 2;;
    --auto_oom_reduce)
                      EXTRA_ARGS+=("--auto_oom_reduce"); shift;;
    *)                EXTRA_ARGS+=("$1"); shift;;
  esac
done

# ----- device autodetect -----------------------------------------------------
NPROC=1
if [[ "$DEVICE" != "cpu" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    NPROC=${NPROC:-1}
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    fi
  fi
fi

echo "USAM smoke train | nproc=$NPROC | device=$DEVICE | max_steps=$MAX_STEPS"
echo "  config: $TRAIN_CFG"
echo "  model:  $MODEL_CFG"
echo "  data:   $DATA_DIR"

CMD=(python -m usam.train
     --config "$TRAIN_CFG"
     --model "$MODEL_CFG"
     --data "$DATA_DIR"
     --max_steps "$MAX_STEPS"
     --device "$DEVICE"
     "${EXTRA_ARGS[@]}")

if [[ "$DEVICE" == "cpu" || "$NPROC" -le 1 ]]; then
  exec "${CMD[@]}"
fi

# Multi-GPU: torchrun.
exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m usam.train \
  --config "$TRAIN_CFG" \
  --model "$MODEL_CFG" \
  --data "$DATA_DIR" \
  --max_steps "$MAX_STEPS" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
