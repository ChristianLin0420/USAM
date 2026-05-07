# SPDX-License-Identifier: MIT
"""Pytest bootstrap.

Ensures the repository root is on ``sys.path`` so ``import usam`` works
without a prior ``pip install -e .``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
