#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# H200 burst-training launcher. Generates an `sbatch` invocation for the
# 500-GPU pretraining job described in plan §6.2 and §10. Does NOT submit
# anything — the team lead reviews and runs `sbatch` manually.
#
# Usage:
#   bash scripts/train_h200.sh stage_b1   # writes runs/h200_stage_b1.sbatch
#   bash scripts/train_h200.sh stage_b2   # writes runs/h200_stage_b2.sbatch
#
# After review:
#   sbatch runs/h200_stage_b1.sbatch

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-stage_b1}"
case "$STAGE" in
  stage_b1) TRAIN_CFG="configs/train/stage_b1_pretrain.yaml";;
  stage_b2) TRAIN_CFG="configs/train/stage_b2_finetune.yaml";;
  *)
    echo "unknown stage: $STAGE  (expected stage_b1 or stage_b2)" >&2
    exit 1
    ;;
esac

MODEL_CFG="configs/model/usam_1_4b.yaml"
DATA_REPO_DEFAULT="datasets/usam_pretrain_mixture"
DATA_REPO="${DATA_REPO:-$DATA_REPO_DEFAULT}"
NODES=125              # 500 H200 / 4 per node
GPUS_PER_NODE=4
WALLTIME="${WALLTIME:-7-00:00:00}"
PARTITION="${PARTITION:-h200}"
ACCOUNT="${ACCOUNT:-usam}"
JOB_NAME="usam_${STAGE}"

OUT_DIR="runs"
mkdir -p "$OUT_DIR"
SBATCH_PATH="$OUT_DIR/h200_${STAGE}.sbatch"

cat > "$SBATCH_PATH" <<'__SBATCH__'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:__GPUS_PER_NODE__
#SBATCH --cpus-per-task=64
#SBATCH --time=__WALLTIME__
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

# NCCL hygiene for H200 + Mellanox fabric.
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# Pre-emption-safe: forward USR1 to the python process so it can flush.
term_handler() {
  echo "[bash] caught USR1 → forwarding to python"
  kill -USR1 "${PYPID:-0}" 2>/dev/null || true
  wait "${PYPID:-0}" || true
  EXIT=$?
  if [[ $EXIT -eq 124 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit $EXIT
}
trap term_handler USR1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500
export RDZV_ID="$SLURM_JOB_ID"

srun --kill-on-bad-exit=1 \
  bash -c '
    torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=__GPUS_PER_NODE__ \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --rdzv_id=$RDZV_ID \
      -m usam.train \
      --config __TRAIN_CFG__ \
      --model __MODEL_CFG__ \
      --data __DATA_REPO__ \
      --output_dir runs/__JOB_NAME__-$SLURM_JOB_ID
  ' &
PYPID=$!
wait "$PYPID"
EXIT=$?
[[ $EXIT -eq 124 ]] && scontrol requeue "$SLURM_JOB_ID"
exit $EXIT
__SBATCH__

# Substitute the placeholders.
sed -i \
  -e "s|__JOB_NAME__|${JOB_NAME}|g" \
  -e "s|__PARTITION__|${PARTITION}|g" \
  -e "s|__ACCOUNT__|${ACCOUNT}|g" \
  -e "s|__NODES__|${NODES}|g" \
  -e "s|__GPUS_PER_NODE__|${GPUS_PER_NODE}|g" \
  -e "s|__WALLTIME__|${WALLTIME}|g" \
  -e "s|__TRAIN_CFG__|${TRAIN_CFG}|g" \
  -e "s|__MODEL_CFG__|${MODEL_CFG}|g" \
  -e "s|__DATA_REPO__|${DATA_REPO}|g" \
  "$SBATCH_PATH"

echo "Wrote $SBATCH_PATH"
echo "Review and submit with:"
echo "  sbatch $SBATCH_PATH"
echo
echo "Sanity: this script does NOT submit anything itself."
