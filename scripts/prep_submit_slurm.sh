#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Submit the USAM Phase A dispatcher as a long-running process and (in a
# separate tmux session) the HF CommitScheduler upload daemon.
#
# IMPORTANT TOPOLOGY
# ------------------
# This script does NOT sbatch the dispatcher itself. The dispatcher is meant
# to live on a login node where it can call ``sbatch`` for individual chunks.
# Likewise the CommitScheduler upload daemon stays on the login node — it
# needs a stable network identity and must outlive any single batch job.
#
# Concretely:
#   1. Run THIS SCRIPT (prep_submit_slurm.sh) inside tmux on the login node.
#      It launches ``python -m prep.dispatch ...`` in the foreground.
#   2. In a SEPARATE tmux session on the login node, run:
#        python -m prep.stage_6_upload --watch --source <src> \
#          --output-root /scratch/$USER/usam
#      Repeat per source you want to mirror to the Hub.
#   3. The dispatcher ssh's into the cluster via ``sbatch`` to run actual
#      stage jobs under slurm/job.sbatch.
#
# Usage:
#   bash scripts/prep_submit_slurm.sh \
#        --output-root /scratch/$USER/usam \
#        --sources droid agibot2026 \
#        --chunks-per-source 32 \
#        [--max-pending 64] [--state /home/$USER/dispatch_state.json]
#
# The dispatcher writes its state to ``--state`` so a tmux restart resumes
# correctly.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT=""
SOURCES=()
CHUNKS_PER_SOURCE=1
MAX_PENDING="${USAM_MAX_PENDING:-64}"
STATE="dispatch_state.json"
POLL_SECONDS=60
PYTHON="${PYTHON:-python}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)        OUTPUT_ROOT="$2"; shift 2 ;;
    --sources)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do SOURCES+=("$1"); shift; done
      ;;
    --chunks-per-source)  CHUNKS_PER_SOURCE="$2"; shift 2 ;;
    --max-pending)        MAX_PENDING="$2"; shift 2 ;;
    --state)              STATE="$2"; shift 2 ;;
    --poll-seconds)       POLL_SECONDS="$2"; shift 2 ;;
    --python)             PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,33p' "$0"
      exit 0
      ;;
    *)                    EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -z "$OUTPUT_ROOT" ]] && { echo "--output-root is required" >&2; exit 2; }
[[ "${#SOURCES[@]}" -eq 0 ]] && { echo "--sources requires at least one source" >&2; exit 2; }

# We do not call sbatch ourselves — the dispatcher does. Sanity-check.
if ! command -v sbatch >/dev/null 2>&1; then
  echo "WARNING: 'sbatch' not on PATH. The dispatcher will fail to submit "\
       "real jobs, but you can still test with --launcher mock." >&2
fi

cat <<EOF
[prep_submit_slurm]
  output-root        = $OUTPUT_ROOT
  sources            = ${SOURCES[*]}
  chunks-per-source  = $CHUNKS_PER_SOURCE
  max-pending        = $MAX_PENDING
  state              = $STATE
  poll-seconds       = $POLL_SECONDS
  python             = $PYTHON

REMINDER: the CommitScheduler upload daemon is a SEPARATE long-running
process. From a different tmux session on the login node, run:
    $PYTHON -m prep.stage_6_upload --watch --source <src> \\
        --output-root $OUTPUT_ROOT
for every source you want mirrored to the Hub.
EOF

exec "$PYTHON" -m prep.dispatch \
  --output-root "$OUTPUT_ROOT" \
  --state "$STATE" \
  --sources "${SOURCES[@]}" \
  --chunks-per-source "$CHUNKS_PER_SOURCE" \
  --max-pending "$MAX_PENDING" \
  --poll-seconds "$POLL_SECONDS" \
  "${EXTRA_ARGS[@]}"
