#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# =============================================================================
# End-to-end RoboMIND smoke pipeline: HF download → extract → stage_2a (inline,
# first N_EPS) → stage_2c depth → stage_4 DINO → assemble → 200-step train.
# Purpose: verify the pipeline works on RoboMIND HDF5 inputs — NOT a real
# training run.
#
# Source: x-humanoid-robomind/RoboMIND on HF Hub (GATED — needs token-granted
# access). RoboMIND ships split tarballs (`*.tar.gz.part-aa/ab/...`) per task,
# so step 2 concatenates parts and untars before staging.
#
# Run from repo root:
#   HF_TOKEN=... WANDB_API_KEY=... bash scripts/e2e_robomind_smoke.sh
#
# Tunable via env vars:
#   HF_REPO   (default x-humanoid-robomind/RoboMIND)
#   EMB_DIR   (default h5_tienkung_gello_1rgb) — embodiment subdir name; the
#             converter's BGR-detect heuristic + ROBOMIND_CAMERA_MAP cover
#             tienkung / agilex / franka / ur out of the box.
#   TASK      (default close_the_drawer_under_the_combination_cabinet) —
#             task name to download from example_data/<EMB_DIR>/<TASK>.tar.gz.part-*
#   N_EPS     (default 8) — number of HDF5 trajectories to stage.
#   MAX_STEPS (default 200)
#   STAGE2C_WORKERS (default 2)
#   STAGE4_WORKERS  (default 4)
#   PREP_IMAGE  (default usam:prep-a100)
#   TRAIN_IMAGE (default usam:prep-a100)
#
# Exit codes: stage number on failure (1=preflight, 2=download/extract,
#             3=stage_2a, 4=stage_2c, 5=symlink, 6=stage_4, 7=assemble, 8=train).
# =============================================================================

set -euo pipefail

# ---------- config ----------
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/usam}"
ROBOMIND_ROOT="${DATA_ROOT}/robomind"
RAW_ROOT="${ROBOMIND_ROOT}/raw"
OUT_ROOT="${ROBOMIND_ROOT}/output"

# Default to the open AlayaNeW mirror of the Tien Kung subset — the upstream
# x-humanoid-robomind/RoboMIND repo is gated and not in the authorized list
# for our HF token. The mirror ships the same tar.gz.part-* layout under
# benchmark1_0_compressed/h5_tienkung_gello_1rgb/<task>.tar.gz.part-*.
HF_REPO="${HF_REPO:-AlayaNeW/RoboMIND_h5_tienkung_gello_1rgb}"
HF_TAR_SUBDIR="${HF_TAR_SUBDIR:-benchmark1_0_compressed/h5_tienkung_gello_1rgb}"
EMB_DIR="${EMB_DIR:-h5_tienkung_gello_1rgb}"
# NB: the AlayaNeW mirror is missing the `.part-aa` of several tasks
# (place_bread_plate_241203, close_trash_bin, place_bread_plate_241204_pro5).
# Those tasks fail to extract. place_bread_plate_1128 is the smallest task
# whose tarball is COMPLETE in the mirror (single .part-aa, ~6 GB).
TASK="${TASK:-place_bread_plate_1128}"
N_EPS="${N_EPS:-8}"
MAX_FRAMES_PER_EP="${MAX_FRAMES_PER_EP:-400}"   # truncate each ep to first N frames (smoke disk budget)
MAX_STEPS="${MAX_STEPS:-200}"
STAGE2C_WORKERS="${STAGE2C_WORKERS:-2}"
STAGE4_WORKERS="${STAGE4_WORKERS:-4}"

PREP_IMAGE="${PREP_IMAGE:-usam:prep-a100}"
TRAIN_IMAGE="${TRAIN_IMAGE:-usam:prep-a100}"

# RoboMIND ships frames at 25 Hz and the converter hard-codes that
# (`ROBOMIND_FPS = 25`); both stage_4 and the LeRobot assembler must
# agree, otherwise the dino-cache indices won't line up with the parquet.
NATIVE_FPS=25

LOG_PREFIX="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/runs/e2e_robomind_${LOG_PREFIX}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/run.log"; }
fail() { log "FAILED: $*"; exit "${1:-99}"; }

# ---------- preflight ----------
log "preflight: REPO=${HF_REPO} EMB_DIR=${EMB_DIR} TASK=${TASK} N_EPS=${N_EPS} HF_TOKEN=${HF_TOKEN:+set} WANDB_API_KEY=${WANDB_API_KEY:+set}"
[[ -z "${HF_TOKEN:-}" ]]      && fail 1 "HF_TOKEN required (RoboMIND is a gated dataset)"
[[ -z "${WANDB_API_KEY:-}" ]] && fail 1 "WANDB_API_KEY required for training step"
docker image inspect "$PREP_IMAGE"  >/dev/null 2>&1 || fail 1 "missing docker image $PREP_IMAGE"
docker image inspect "$TRAIN_IMAGE" >/dev/null 2>&1 || fail 1 "missing docker image $TRAIN_IMAGE"

# ---------- step 0: wipe ----------
log "step 0: wipe ${ROBOMIND_ROOT} and runs/wandb"
sudo find "${REPO_ROOT}/runs" -mindepth 1 -maxdepth 1 ! -path "${LOG_DIR}" -exec rm -rf {} + 2>/dev/null || true
sudo rm -rf "${ROBOMIND_ROOT}" "${REPO_ROOT}/wandb"/* 2>/dev/null || true
mkdir -p "${RAW_ROOT}/${EMB_DIR}" "${OUT_ROOT}/staged/robomind/chunk-000" \
         "${REPO_ROOT}/runs" "${REPO_ROOT}/wandb" "${LOG_DIR}"
log "  wiped → ${ROBOMIND_ROOT}, kept ${LOG_DIR}"

# ---------- step 1: HF snapshot of ONE task's split tarball parts ----------
log "step 1: snapshot_download(${HF_REPO}) — allow=${HF_TAR_SUBDIR}/${TASK}.tar.gz.part-*"
STEP_T0=$(date +%s)
docker run --rm \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_REPO="${HF_REPO}" \
    -e HF_TAR_SUBDIR="${HF_TAR_SUBDIR}" \
    -e EMB_DIR="${EMB_DIR}" \
    -e TASK="${TASK}" \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    bash -c "pip install -q huggingface_hub 2>&1 >/dev/null && python -u - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

repo = os.environ['HF_REPO']
subd = os.environ['HF_TAR_SUBDIR']
task = os.environ['TASK']
pattern = f'{subd}/{task}.tar.gz.part-*'
print(f'pulling pattern={pattern} from {repo}', flush=True)
path = snapshot_download(
    repo_id=repo,
    repo_type='dataset',
    allow_patterns=[pattern],
    local_dir='/workspace/output/robomind/raw/_dl',
    token=os.environ['HF_TOKEN'],
)
print('snapshot ->', path, flush=True)
PYEOF" 2>&1 | tee -a "${LOG_DIR}/01_hf_snapshot.log" | tail -10 \
    || fail 2 "HF snapshot_download failed (gated access? wrong task name?)"

DL_DIR="${RAW_ROOT}/_dl/${HF_TAR_SUBDIR}"
PARTS_COUNT=$(ls "${DL_DIR}/${TASK}.tar.gz.part-"* 2>/dev/null | wc -l)
[[ "$PARTS_COUNT" -eq 0 ]] && fail 2 "no tarball parts under ${DL_DIR}"
log "  fetched ${PARTS_COUNT} parts, $(du -sh "${DL_DIR}" 2>/dev/null | cut -f1) total"

# ---------- step 2: concat parts + extract ----------
log "step 2: concat .part-* and untar into ${RAW_ROOT}/${EMB_DIR}/"
STEP_T0=$(date +%s)
TARBALL="/tmp/${TASK}.tar.gz"
cat "${DL_DIR}/${TASK}.tar.gz.part-"* > "$TARBALL"
tar -xzf "$TARBALL" -C "${RAW_ROOT}/${EMB_DIR}/" 2>&1 | tail -5 | tee -a "${LOG_DIR}/02_extract.log" \
    || fail 2 "tar extract failed"
rm -f "$TARBALL"
# RoboMIND mirrors ship trajectories as `.../data/trajectory.hdf5`; the upstream
# converter's `rglob('*.h5')` doesn't match `.hdf5`. The inline stage_2a in
# step 3 matches both extensions explicitly; here we just count for the gate.
N_H5=$(find "${RAW_ROOT}/${EMB_DIR}" \( -name '*.h5' -o -name '*.hdf5' \) 2>/dev/null | wc -l)
[[ "$N_H5" -eq 0 ]] && fail 2 "0 HDF5 trajectories found in ${RAW_ROOT}/${EMB_DIR}"
log "  extracted ${N_H5} HDF5 trajectories in $(( $(date +%s) - STEP_T0 ))s"

# ---------- step 3: stage_2a INLINE (translates AlayaNeW schema → converter payload) ----------
log "step 3: stage_2a inline — staging first ${N_EPS} HDF5s of ${EMB_DIR} (max ${MAX_FRAMES_PER_EP} frames/ep)"
STEP_T0=$(date +%s)
docker run --rm \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -e EMB_DIR="${EMB_DIR}" \
    -e N_EPS="${N_EPS}" \
    -e MAX_FRAMES_PER_EP="${MAX_FRAMES_PER_EP}" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    bash -c "pip install -q h5py 2>&1 >/dev/null && python -u - <<'PYEOF'
import os, sys, time, json
import numpy as np
import cv2
import h5py
sys.path.insert(0, '/workspace/USAM')
from pathlib import Path
from prep.stage_2a_to_lerobot.robomind import RoboMINDConverter
from prep.stage_2a_to_lerobot.__main__ import _process_one
from prep._base import EpisodeRef

EMB = os.environ['EMB_DIR']
N   = int(os.environ['N_EPS'])
MAX_T = int(os.environ['MAX_FRAMES_PER_EP'])
RAW = Path('/workspace/output/robomind/raw')
OUT = Path('/workspace/output/robomind/output/staged/robomind/chunk-000')
OUT.mkdir(parents=True, exist_ok=True)

# Skip the _dl staging area used for HF snapshot. Match both *.h5 and *.hdf5
# (the AlayaNeW mirror packages trajectories as data/trajectory.hdf5).
h5s = sorted(
    p for p in (*RAW.rglob('*.h5'), *RAW.rglob('*.hdf5'))
    if '_dl' not in p.parts
)
print(f'found {len(h5s)} HDF5 trajectories under {RAW} (excluding _dl); staging first {min(N, len(h5s))}', flush=True)

def build_payload_alayanew(f, max_t):
    '''Translate AlayaNeW Tien Kung HDF5 layout into the converter's payload dict.

    AlayaNeW layout:
      /observations/rgb_images/camera_top   shape (T,) object  (JPEG bytes)
      /master/joint_position                shape (T, 16) f64  (teleop leader = action target)
      /puppet/joint_position                shape (T, 16) f64  (robot state)
      /language_raw                         shape (1,) object  (UTF-8 bytes)

    Converter expects:
      observations/images/<cam>  uint8 [T,H,W,3]
      observations/joint_position [T, D]
      actions [T, D]
      language_instruction
    '''
    cam_ds = f['observations/rgb_images/camera_top']
    T_full = len(cam_ds)
    T = min(T_full, max_t)
    # JPEG decode -> BGR uint8 (converter's BGR-detect heuristic then swaps to RGB).
    sample = cv2.imdecode(np.frombuffer(bytes(cam_ds[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    H, W = sample.shape[:2]
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    frames[0] = sample
    for t in range(1, T):
        img = cv2.imdecode(np.frombuffer(bytes(cam_ds[t]), dtype=np.uint8), cv2.IMREAD_COLOR)
        frames[t] = img

    puppet = np.asarray(f['puppet/joint_position'][:T], dtype=np.float32)
    master = np.asarray(f['master/joint_position'][:T], dtype=np.float32)

    raw_instr = f['language_raw'][()]
    if isinstance(raw_instr, np.ndarray) and raw_instr.size > 0:
        instr_bytes = raw_instr.flat[0]
    else:
        instr_bytes = raw_instr
    if isinstance(instr_bytes, (bytes, bytearray)):
        instr = instr_bytes.decode('utf-8', errors='ignore')
    else:
        instr = str(instr_bytes)

    return {
        'camera::head_camera': frames,                     # 'head_camera' aliases to 'head_rgb'
        'obs::joint_position': puppet,                     # robot state
        'actions': master,                                 # teleop leader command
        'language_instruction': np.array([instr.encode('utf-8')]),
    }

c = RoboMINDConverter(chunk=0, output_root=OUT, raw_root=RAW, drop_simulation=True)

# Monkey-patch convert_episode so the existing _process_one writer path is reused.
def _custom_convert(ref):
    with h5py.File(ref.raw_path, 'r') as f:
        payload = build_payload_alayanew(f, MAX_T)
    return c._payload_to_result(ref, payload)
c.convert_episode = _custom_convert

n_ok = n_fail = 0
t_start = time.time()
for i, h5_path in enumerate(h5s[:N]):
    ep_id = f'robomind_{EMB}_{h5_path.parent.parent.name}'   # parent.parent = trajectory dir (e.g. 1128_160221)
    ref = EpisodeRef(
        episode_id=ep_id,
        source='robomind',
        raw_path=str(h5_path),
        extra={'embodiment_dir': EMB},
    )
    t0 = time.time()
    try:
        _process_one(c, ref)
        n_ok += 1
        print(f'  {i+1}/{min(N, len(h5s))} OK {ep_id} ({time.time()-t0:.1f}s)', flush=True)
    except Exception as e:
        n_fail += 1
        import traceback
        print(f'  ep {i} FAILED {ep_id}: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
print(f'stage_2a inline: {n_ok} OK / {n_fail} fail in {time.time()-t_start:.1f}s', flush=True)
PYEOF" 2>&1 | tee -a "${LOG_DIR}/03_stage2a.log" | tail -50 \
    || fail 3 "stage_2a inline failed"

N_EPS_STAGED=$(find "${OUT_ROOT}/staged/robomind/chunk-000" -maxdepth 1 -name 'ep_*' -type d 2>/dev/null | wc -l)
log "  staged ${N_EPS_STAGED} episodes in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_EPS_STAGED" -eq 0 ]] && fail 3 "stage_2a produced 0 staged episodes"

# ---------- step 4: stage_2c parallel depth ----------
log "step 4: stage_2c depth (--workers-per-gpu ${STAGE2C_WORKERS})"
STEP_T0=$(date +%s)
docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_2c_compute_depth \
        --dataset robomind --chunk 0 \
        --staged-root /workspace/output/robomind/output/staged \
        --output-root /workspace/output/robomind/output/depth \
        --workers-per-gpu "${STAGE2C_WORKERS}" \
        --resume 2>&1 | tee -a "${LOG_DIR}/04_stage2c.log" | tail -20 \
    || fail 4 "stage_2c failed"
N_DEPTH=$(find "${OUT_ROOT}/depth" -name 'depth_head_rgb.npy' 2>/dev/null | wc -l)
log "  wrote ${N_DEPTH} depth files in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_DEPTH" -lt "$N_EPS_STAGED" ]] && log "  WARN: ${N_DEPTH} < ${N_EPS_STAGED} depth files"

# ---------- step 5: symlink depth into staged ----------
log "step 5: symlink depth into staged ep dirs"
sudo bash -c "cd '${OUT_ROOT}' && for ep in depth/robomind/chunk-000/ep_*; do
    epname=\$(basename \"\$ep\")
    src_depth='../../../../depth/robomind/chunk-000/'\$epname'/depth_head_rgb.npy'
    tgt_ep='staged/robomind/chunk-000/'\$epname
    if [ -d \"\$tgt_ep\" ]; then ln -sfn \"\$src_depth\" \"\$tgt_ep/depth_head_rgb.npy\"; fi
done"
N_LINKS=$(find "${OUT_ROOT}/staged" -name 'depth_head_rgb.npy' 2>/dev/null | wc -l)
[[ "$N_LINKS" -gt 0 ]] || fail 5 "no depth symlinks under ${OUT_ROOT}/staged"
FIRST_EP=$(find "${OUT_ROOT}/staged/robomind/chunk-000" -mindepth 1 -maxdepth 1 -type d -name 'ep_*' 2>/dev/null | sort | sed -n '1p')
[[ -e "${FIRST_EP}/depth_head_rgb.npy" ]] || fail 5 "symlink does not resolve: ${FIRST_EP}/depth_head_rgb.npy"
log "  ${N_LINKS} symlinks resolved (spot-check ${FIRST_EP##*/} OK)"

# ---------- step 6: stage_4 DINO cache ----------
log "step 6: stage_4 DINOv3 cache (--workers-per-gpu ${STAGE4_WORKERS}, source-fps=${NATIVE_FPS})"
STEP_T0=$(date +%s)
docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_4_dino_cache \
        --dataset robomind --chunk 0 \
        --staged-root /workspace/output/robomind/output/staged \
        --output-root /workspace/output/robomind/output/dino_cache \
        --dinov3-ckpt /opt/dinov3-cache/facebook/dinov3-vitl16-pretrain-lvd1689m \
        --dinov3-arch vit_l_16 \
        --source-fps "${NATIVE_FPS}" \
        --workers-per-gpu "${STAGE4_WORKERS}" \
        --cache-fps 5 2>&1 | tee -a "${LOG_DIR}/06_stage4.log" | tail -30 \
    || fail 6 "stage_4 failed"
N_SHARDS_OUT=$(find "${OUT_ROOT}/dino_cache" -name '*.safetensors' 2>/dev/null | wc -l)
log "  wrote ${N_SHARDS_OUT} safetensors shards in $(( $(date +%s) - STEP_T0 ))s"
[[ "$N_SHARDS_OUT" -eq 0 ]] && fail 6 "stage_4 produced 0 shards"

# ---------- step 7: assemble ----------
log "step 7: assemble (fps=${NATIVE_FPS}, fps_features=5)"
docker run --rm \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -w /workspace/USAM \
    "$PREP_IMAGE" \
    python -m prep.stage_2a_to_lerobot._assemble \
        --dataset robomind --chunk 0 \
        --staged-root /workspace/output/robomind/output/staged \
        --dino-cache-root /workspace/output/robomind/output/dino_cache \
        --out-root /workspace/output/robomind/output/lerobot \
        --fps "${NATIVE_FPS}" --fps-features 5 2>&1 | tee -a "${LOG_DIR}/07_assemble.log" | tail -10 \
    || fail 7 "assemble failed"
N_KEPT=$(python3 -c "
import pyarrow.parquet as pq
print(pq.read_table('${OUT_ROOT}/lerobot/meta/episodes.parquet').num_rows)
" 2>/dev/null || echo "?")
log "  assembled (${N_KEPT} episodes kept after short-episode filter)"

# ---------- step 8: training (smoke; very few steps) ----------
log "step 8: ${MAX_STEPS}-step smoke training on 8 GPUs with wandb"
STEP_T0=$(date +%s)
docker run --rm --gpus all --shm-size=16g \
    -v "${REPO_ROOT}:/workspace/USAM" \
    -v "${ROBOMIND_ROOT}:/workspace/output/robomind" \
    -v "${REPO_ROOT}/runs:/workspace/USAM/runs" \
    -v "${REPO_ROOT}/wandb:/workspace/USAM/wandb" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-usam-smoke}" \
    -e USAM_VIZ_INTERVAL=100 \
    -e PYTHONPATH=/workspace/USAM \
    -w /workspace/USAM \
    "$TRAIN_IMAGE" \
    bash -c "pip install -q wandb 2>&1 >/dev/null && torchrun --standalone --nproc_per_node=8 -m usam.train \
        --config /workspace/USAM/configs/train/stage_b1_pretrain.yaml \
        --model /workspace/USAM/configs/model/usam_350m_smoke.yaml \
        --data /workspace/output/robomind/output/lerobot \
        --max_steps ${MAX_STEPS} --log_every 25 --device auto --auto_oom_reduce" 2>&1 \
    | tee -a "${LOG_DIR}/08_train.log" | tail -80 \
    || fail 8 "training failed"
log "  training done in $(( $(date +%s) - STEP_T0 ))s"

log "ALL DONE: ${N_EPS_STAGED} eps → lerobot @ ${OUT_ROOT}/lerobot   (logs: ${LOG_DIR})"
