#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Render slurm/pipeline.sbatch.tmpl into one concrete file per dataset.
# Outputs: slurm/pipeline_{droid,agibot2026,robomind,bridge}.sbatch
#
# Idempotent: rerun any time you tweak the template.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEMPLATE="slurm/pipeline.sbatch.tmpl"
DATASETS=(droid agibot2026 robomind bridge)

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

for ds in "${DATASETS[@]}"; do
    out="slurm/pipeline_${ds}.sbatch"
    sed "s/__DATASET__/${ds}/g" "$TEMPLATE" > "$out"
    chmod +x "$out"
    echo "rendered $out"
done
