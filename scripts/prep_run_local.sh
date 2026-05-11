#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Run the full USAM Phase A pipeline locally for ONE source on the dev box.
#
# Sequence: download -> index -> 2a -> 2b -> 2c -> 3 -> 4 -> validate
# (Stage 6 / Hub upload is out-of-scope for the local path; use
#  scripts/prep_submit_slurm.sh + the long-lived CommitScheduler for that.)
#
# Usage:
#   bash scripts/prep_run_local.sh --source droid \
#        --config configs/data/droid.yaml \
#        --raw /scratch/usam/droid/raw \
#        --out /scratch/usam/droid/output \
#        [--chunk 0] [--dry-run] [--skip-download]
#
# Defaults to chunk 0; rerun with a different --chunk to process more.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE=""
CONFIG=""
RAW=""
OUT=""
CHUNK=0
DRY_RUN=0
SKIP_DOWNLOAD=0
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)       SOURCE="$2"; shift 2 ;;
    --config)       CONFIG="$2"; shift 2 ;;
    --raw)          RAW="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    --chunk)        CHUNK="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --python)       PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$SOURCE" ]] && { echo "--source is required" >&2; exit 2; }
[[ -z "$CONFIG" ]] && { echo "--config is required" >&2; exit 2; }
[[ -z "$RAW" ]] && { echo "--raw is required" >&2; exit 2; }
[[ -z "$OUT" ]] && { echo "--out is required" >&2; exit 2; }

mkdir -p "$RAW" "$OUT"

DRY_FLAG=""
[[ "$DRY_RUN" -eq 1 ]] && DRY_FLAG="--dry-run"

echo "[prep_run_local] source=$SOURCE chunk=$CHUNK dry_run=$DRY_RUN"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo "[prep_run_local] stage 0: download"
  "$PYTHON" -m "prep.stage_0_download.${SOURCE}" \
    --config "$CONFIG" --cache "$RAW" $DRY_FLAG
else
  echo "[prep_run_local] stage 0: SKIPPED"
fi

echo "[prep_run_local] stage 1: index"
"$PYTHON" -m prep.stage_1_index \
  --dataset "$SOURCE" --raw "$RAW" --out "$OUT" --chunk "$CHUNK"

# Stages 2a, 2b, 2c, 3, 4 are owned by data-engineer; entry points accept
# --dataset / --chunk / --resume (Wave F: one A100 node per dataset).
for STAGE in stage_2a_to_lerobot stage_2b_compute_flow stage_2c_compute_depth \
             stage_3_canonical stage_4_dino_cache; do
  echo "[prep_run_local] $STAGE"
  set +e
  "$PYTHON" -m "prep.${STAGE}" \
    --dataset "$SOURCE" --chunk "$CHUNK" --resume
  RC=$?
  set -e
  if [[ "$RC" -ne 0 && "$RC" -ne 124 ]]; then
    echo "[prep_run_local] $STAGE failed with exit $RC" >&2
    exit "$RC"
  fi
done

echo "[prep_run_local] stage 5: validate"
"$PYTHON" -m prep.stage_5_validate \
  --dataset "$SOURCE" --output-root "$OUT"

echo "[prep_run_local] done. Outputs at $OUT/$SOURCE"
