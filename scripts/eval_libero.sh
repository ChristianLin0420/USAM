#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# scripts/eval_libero.sh — open-loop ADE evaluation wrapper.
#
# Usage:
#   scripts/eval_libero.sh <ckpt_path> [<output_path>] [extra args...]
#
# Examples:
#   scripts/eval_libero.sh runs/<run-id>/checkpoints/checkpoint_best.pt
#   scripts/eval_libero.sh foo.pt foo_metrics.json --seed 42 --n-samples 1000
#
# The script invokes ``python -m usam.inference.openloop`` with the
# canonical LIBERO eval config. It writes the JSON metrics to
# ``<output>`` (or to ``<ckpt-dir>/openloop_<ckpt>.json`` if omitted)
# and echoes the same JSON to stdout.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ckpt_path> [<output_path>] [extra args...]" >&2
  exit 2
fi

CKPT="$1"; shift
OUTPUT_ARGS=()
if [[ $# -ge 1 && "${1:0:1}" != "-" ]]; then
  OUTPUT_ARGS=(--output "$1")
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/eval/libero.yaml"

PYTHON="${PYTHON:-python}"
exec "${PYTHON}" -m usam.inference.openloop \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
