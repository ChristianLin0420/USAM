#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# =============================================================================
# End-to-end DROID smoke pipeline: download → convert → cache → train + wandb.
# Validates the post-flow-removal codebase on real DROID data, exercising the
# parallel prep paths (stage_2c --workers-per-gpu, stage_4 --workers-per-gpu)
# and the GPU-pinned training loop with pinned-memory dataloader.
#
# Run from repo root:
#   HF_TOKEN=... WANDB_API_KEY=... bash scripts/e2e_droid_smoke.sh
#
# Tunable via env vars:
#   N_SHARDS         (default 10)  DROID TFDS shards to download (~870 MB each)
#   STAGE2C_WORKERS  (default 2)   DA3 depth instances per physical GPU
#   STAGE4_WORKERS   (default 4)   DINOv3 instances per physical GPU
#   MAX_STEPS        (default 2000) training steps
#   PREP_IMAGE       (default usam:prep-a100) container with TFDS + DA3 + DINOv3
#   TRAIN_IMAGE      (default usam:prep-a100) container for training (uses prep image since T0 has CUDA-version mismatch)
#
# What the script does:
#   1. Refuses to start if HF_TOKEN / WANDB_API_KEY are missing.
#   2. Wipes /localhome/local-chrislin/USAM/data/usam and runs/, wandb/.
#   3. gsutil cp the first N_SHARDS DROID tfrecord files + 3 metadata files.
#   4. Rename shards to -of-NSHARDS and truncate dataset_info.json.
#   5. stage_2a (single-process; mp.Pool deadlocks on TFDS warmup — known issue).
#   6. stage_2c depth, parallel via --workers-per-gpu STAGE2C_WORKERS.
#   7. Symlink depth into staged/ ep dirs (4 ../ levels up; common gotcha).
#   8. stage_4 DINOv3 cache, parallel via --workers-per-gpu STAGE4_WORKERS.
#   9. Assemble USAM-LeRobot v2.1 layout from staged + dino_cache.
#  10. Optional: HTML viz (set RUN_VIZ=1 to enable; default skipped).
#  11. 2000-step training on 8 GPU torchrun with wandb logging.
#  12. Tail wandb summary at the end.
#
# Exit codes:
#   0  every step passed
#   1+ first failing step's stage number (e.g., 5 = stage_2a, 11 = train)
# =============================================================================

set -euo pipefail

# ---------- config ----------
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/usam}"
DROID_ROOT="${DATA_ROOT}/droid"
RAW_TFDS="${DROID_ROOT}/raw/tfds/droid/1.0.0"
OUT_ROOT="${DROID_ROOT}/output"

N_SHARDS="${N_SHARDS:-10}"
STAGE2C_WORKERS="${STAGE2C_WORKERS:-2}"
STAGE4_WORKERS="${STAGE4_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-2000}"

PREP_IMAGE="${PREP_IMAGE:-usam:prep-a100}"
TRAIN_IMAGE="${TRAIN_IMAGE:-usam:prep-a100}"

LOG_PREFIX="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/runs/e2e_${LOG_PREFIX}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/run.log"; }
fail() { log "FAILED: $*"; exit "${1:-99}"; }

# ---------- preflight ----------
log "preflight: HF_TOKEN=${HF_TOKEN:+set} WANDB_API_KEY=${WANDB_API_KEY:+set}"
[[ -z "${HF_TOKEN:-}" ]]         && fail 1 "HF_TOKEN env var not set"
[[ -z "${WANDB_API_KEY:-}" ]]    && fail 1 "WANDB_API_KEY env var not set"
command -v ~/.local/bin/gsutil >/dev/null 2>&1 || command -v gsutil >/dev/null 2>&1 || {
    log "installing gsutil via pip --user"
    pip install --user --quiet gsutil
}
GSUTIL=$(command -v gsutil || echo "$HOME/.local/bin/gsutil")
docker image inspect "$PREP_IMAGE" >/dev/null 2>&1 || fail 1 "missing docker image $PREP_IMAGE"
docker image inspect "$TRAIN_IMAGE" >/dev/null 2>&1 || fail 1 "missing docker image $TRAIN_IMAGE"

# ---------- wipe ----------
log "step 0: wipe data + runs + wandb (preserving our LOG_DIR)"
# Exclude LOG_DIR from the runs/ wipe; otherwise the next `tee` call dies.
sudo find "${REPO_ROOT}/runs" -mindepth 1 -maxdepth 1 ! -path "${LOG_DIR}" -exec rm -rf {} + 2>/dev/null || true
sudo rm -rf "${DATA_ROOT}" "${REPO_ROOT}/wandb"/* 2>/dev/null || true
mkdir -p "${RAW_TFDS}" "${OUT_ROOT}" "${REPO_ROOT}/runs" "${REPO_ROOT}/wandb" "${LOG_DIR}"
log "  wiped → ${DATA_ROOT}, ${REPO_ROOT}/runs/* (kept ${LOG_DIR}), ${REPO_ROOT}/wandb"

# ---------- step 1: download N_SHARDS ----------
log "step 1: download $N_SHARDS DROID tfrecord shards (~$((N_SHARDS * 870 / 1000)) GB)"
SHARDS=""
for i in $(seq 0 $((N_SHARDS - 1))); do
    n=$(printf "%05d" "$i")
    SHARDS="$SHARDS gs://gresearch/robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-${n}-of-02048"
done
SHARDS="$SHARDS gs://gresearch/robotics/droid/1.0.0/dataset_info.json gs://gresearch/robotics/droid/1.0.0/features.json gs://gresearch/robotics/droid/1.0.0/CC-BY-4.0"
cd "$RAW_TFDS"
"$GSUTIL" -m cp $SHARDS . 2>&1 | tail -3 | tee -a "${LOG_DIR}/01_download.log" || fail 1 "gsutil download failed"
cd "$REPO_ROOT"
log "  downloaded $(ls "$RAW_TFDS"/*.tfrecord-* | wc -l) shards, $(du -sh "$RAW_TFDS" | cut -f1) total"

# ---------- step 2: rename + patch dataset_info.json ----------
log "step 2: rename shards 0..$((N_SHARDS-1)) to -of-$(printf '%05d' "$N_SHARDS") + truncate shardLengths"
cp "${RAW_TFDS}/dataset_info.json" "${RAW_TFDS}/dataset_info.json.bak"
for i in $(seq 0 $((N_SHARDS - 1))); do
    n=$(printf "%05d" "$i")
    nshards=$(printf "%05d" "$N_SHARDS")
    src="${RAW_TFDS}/r2d2_faceblur-train.tfrecord-${n}-of-02048"
    dst="${RAW_TFDS}/r2d2_faceblur-train.tfrecord-${n}-of-${nshards}"
    [[ -f "$src" ]] && mv "$src" "$dst"
done
python3 - <<PYEOF | tee -a "${LOG_DIR}/02_patch.log"
import json
p = "${RAW_TFDS}/dataset_info.json"
d = json.load(open(p))
d["splits"][0]["shardLengths"] = d["splits"][0]["shardLengths"][:${N_SHARDS}]
total_ep = sum(int(x) for x in d["splits"][0]["shardLengths"])
d["splits"][0]["numBytes"] = str(${N_SHARDS} * 870_000_000)
json.dump(d, open(p, "w"), indent=2)
print(f"truncated to {len(d['splits'][0]['shardLengths'])} shards = {total_ep} episodes")
PYEOF

# ---------- step 3: stage_2a (RLDS → staged ep_*) ----------
log "step 3: stage_2a (single-process; mp.Pool deadlocks on TFDS warmup)"
STEP_T0=$(date +%s)
docker run --rm \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${DATA_ROOT}:/workspace/output" \
    -e USAM_EPISODES_PER_CHUNK=512 \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    bash -c "pip install -q tensorflow-cpu gcsfs 2>&1 >/dev/null && python -m prep.stage_2a_to_lerobot \
        --dataset droid --chunk 0 \
        --staged-root /workspace/output/droid/output/staged \
        --rlds-data-dir /workspace/output/droid/raw/tfds \
        --num-workers 1 --resume" 2>&1 | tee -a "${LOG_DIR}/03_stage2a.log" | tail -50 \
    || fail 3 "stage_2a failed"
N_EPS=$(find "${OUT_ROOT}/staged" -maxdepth 4 -name 'ep_*' -type d 2>/dev/null | wc -l)
log "  staged ${N_EPS} episodes in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_EPS" -eq 0 ]] && fail 3 "stage_2a produced 0 episodes"

# ---------- step 4: stage_2c parallel depth ----------
log "step 4: stage_2c depth (--workers-per-gpu ${STAGE2C_WORKERS})"
STEP_T0=$(date +%s)
docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${DATA_ROOT}:/workspace/output" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_2c_compute_depth \
        --dataset droid --chunk 0 \
        --staged-root /workspace/output/droid/output/staged \
        --output-root /workspace/output/droid/output/depth \
        --workers-per-gpu "${STAGE2C_WORKERS}" \
        --resume 2>&1 | tee -a "${LOG_DIR}/04_stage2c.log" | tail -20 \
    || fail 4 "stage_2c failed"
N_DEPTH=$(find "${OUT_ROOT}/depth" -name 'depth_head_rgb.npy' 2>/dev/null | wc -l)
log "  wrote ${N_DEPTH} depth files in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_DEPTH" -lt "$N_EPS" ]] && log "  WARN: ${N_DEPTH} < ${N_EPS} depth files (some episodes skipped?)"

# ---------- step 5: symlink depth into staged ----------
log "step 5: symlink depth into staged ep dirs (4 ../ levels)"
sudo bash -c "cd '${OUT_ROOT}' && for ep in depth/droid/chunk-000/ep_*; do
    epname=\$(basename \"\$ep\")
    src_depth='../../../../depth/droid/chunk-000/'\$epname'/depth_head_rgb.npy'
    tgt_ep='staged/droid/chunk-000/'\$epname
    if [ -d \"\$tgt_ep\" ]; then ln -sfn \"\$src_depth\" \"\$tgt_ep/depth_head_rgb.npy\"; fi
done"
# Avoid `find | head -1` (SIGPIPE under pipefail). Just count + spot-check one ep.
N_LINKS=$(find "${OUT_ROOT}/staged" -name 'depth_head_rgb.npy' -print 2>/dev/null | wc -l)
[[ "$N_LINKS" -gt 0 ]] || fail 5 "no depth symlinks under ${OUT_ROOT}/staged"
FIRST_EP=$(find "${OUT_ROOT}/staged/droid/chunk-000" -mindepth 1 -maxdepth 1 -type d -name 'ep_*' -printf '%p\n' 2>/dev/null | sort | sed -n '1p')
[[ -e "${FIRST_EP}/depth_head_rgb.npy" ]] || fail 5 "symlink does not resolve: ${FIRST_EP}/depth_head_rgb.npy"
log "  ${N_LINKS} symlinks resolved (spot-check ${FIRST_EP##*/} OK)"

# ---------- step 6: stage_4 parallel DINO cache ----------
log "step 6: stage_4 DINOv3 cache (--workers-per-gpu ${STAGE4_WORKERS})"
STEP_T0=$(date +%s)
docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${DATA_ROOT}:/workspace/output" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_4_dino_cache \
        --dataset droid --chunk 0 \
        --staged-root /workspace/output/droid/output/staged \
        --output-root /workspace/output/droid/output/dino_cache \
        --dinov3-ckpt /opt/dinov3-cache/facebook/dinov3-vitl16-pretrain-lvd1689m \
        --dinov3-arch vit_l_16 \
        --source-fps 15 \
        --workers-per-gpu "${STAGE4_WORKERS}" \
        --cache-fps 5 2>&1 | tee -a "${LOG_DIR}/06_stage4.log" | tail -30 \
    || fail 6 "stage_4 failed"
N_SHARDS_OUT=$(find "${OUT_ROOT}/dino_cache" -name '*.safetensors' 2>/dev/null | wc -l)
log "  wrote ${N_SHARDS_OUT} safetensors shards in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_SHARDS_OUT" -eq 0 ]] && fail 6 "stage_4 produced 0 shards"

# ---------- step 7: assemble LeRobot layout ----------
log "step 7: assemble LeRobot v2.1 layout"
docker run --rm \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${DATA_ROOT}:/workspace/output" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_2a_to_lerobot._assemble \
        --dataset droid --chunk 0 \
        --staged-root /workspace/output/droid/output/staged \
        --dino-cache-root /workspace/output/droid/output/dino_cache \
        --out-root /workspace/output/droid/output/lerobot \
        --fps 15 --fps-features 5 2>&1 | tee -a "${LOG_DIR}/07_assemble.log" | tail -10 \
    || fail 7 "assemble failed"
N_KEPT=$(python3 -c "
import pyarrow.parquet as pq
print(pq.read_table('${OUT_ROOT}/lerobot/meta/episodes.parquet').num_rows)
" 2>/dev/null || echo "?")
log "  assembled (${N_KEPT} episodes kept after short-episode filter)"

# ---------- step 8 (optional): HTML viz ----------
if [[ "${RUN_VIZ:-0}" -eq 1 ]]; then
    log "step 8: HTML viz"
    docker run --rm --gpus all \
        -v "${REPO_ROOT}:/workspace/USAM" \
        -v "${OUT_ROOT}/staged/droid/chunk-000:/workspace/output/staged" \
        -v "${OUT_ROOT}/depth/droid/chunk-000:/workspace/output/depth" \
        -v "${OUT_ROOT}/viz:/workspace/output/viz" \
        -w /workspace/output \
        -e PYTHONPATH=/workspace/USAM \
        "$PREP_IMAGE" \
        python /workspace/USAM/tools/viz/dinov3_pca_gallery.py 2>&1 | tee -a "${LOG_DIR}/08_viz.log" | tail -10 \
        || fail 8 "viz failed"
    [[ -f "${OUT_ROOT}/viz/dinov3_chunk0/index.html" ]] || fail 8 "viz index.html missing"
    log "  index.html at ${OUT_ROOT}/viz/dinov3_chunk0/index.html"
fi

# ---------- step 9: training ----------
log "step 9: 2000-step training on 8 GPUs with wandb"
STEP_T0=$(date +%s)
docker run --rm --gpus all --shm-size=16g \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${DATA_ROOT}:/workspace/output" \
    -v "${REPO_ROOT}/runs:/workspace/USAM/runs" \
    -v "${REPO_ROOT}/wandb:/workspace/USAM/wandb" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-usam-smoke}" \
    -e USAM_VIZ_INTERVAL=100 \
    -e PYTHONPATH=/workspace/USAM \
    -w /workspace/USAM \
    "$TRAIN_IMAGE" \
    bash -c "pip install -q wandb 2>&1 >/dev/null && bash scripts/train_smoke_a40.sh \
        --config configs/train/stage_b1_pretrain.yaml \
        --model configs/model/usam_350m_smoke.yaml \
        --data /workspace/output/droid/output/lerobot \
        --max_steps ${MAX_STEPS} \
        --auto_oom_reduce" 2>&1 | tee -a "${LOG_DIR}/09_train.log" | tail -80 \
    || fail 9 "training failed"
log "  training completed in $(( $(date +%s) - STEP_T0 ))s"

# ---------- step 10: wandb summary ----------
log "step 10: wandb summary"
LATEST_RUN=$(ls -dt "${REPO_ROOT}/wandb/run-"* 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_RUN" ]]; then
    log "  wandb run dir: $LATEST_RUN"
    log "  run URL: $(grep 'wandb run:' "$LATEST_RUN/files/output.log" 2>/dev/null | head -1 || echo '?')"
    log "  last 5 step lines:"
    tail -5 "$LATEST_RUN/files/output.log" 2>/dev/null | sed 's/^/    /' | tee -a "${LOG_DIR}/run.log"
fi

log "E2E pipeline complete. Logs at ${LOG_DIR}"
exit 0
