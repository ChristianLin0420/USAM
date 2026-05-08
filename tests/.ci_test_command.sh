#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# CI test command — single source of truth for what runs on PR / push CI.
# Invoked by .github/workflows/ci.yaml:
#
#     - name: Unit tests
#       run: bash tests/.ci_test_command.sh
#
# We filter the suite to CPU-runnable tests only:
#
#   gpu_1    — requires a single GPU
#   gpu_8    — requires 8 GPUs
#   network  — needs HF Hub / GCS access
#   slow     — kept for nightly, skipped on PR CI
#
# The four markers are declared in pyproject.toml [tool.pytest.ini_options].
# Adding ``-x`` keeps PR-time feedback fast: the first failure aborts the run.
#
# This file is owned by the test-engineer; infra-engineer wires only the
# workflow file. Keep the marker filter and the pyproject declaration in
# sync.

set -euo pipefail

# Allow the caller to override pytest (e.g. for running under a specific
# conda env locally). Default to the current PATH.
PYTEST="${PYTEST:-pytest}"

exec "$PYTEST" tests/unit/ \
    -m "not gpu_1 and not gpu_8 and not network and not slow" \
    -x
