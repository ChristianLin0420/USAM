# SPDX-License-Identifier: MIT
"""pytest fixtures for USAM unit + integration tests.

If the LFS-tracked ``tests/golden_data/tiny_droid`` fixture is absent (as is
the case before test-engineer's Wave 4 lands), we synthesize a small
DROID-shaped stand-in via :mod:`tests.golden_data._synthesize_tiny_droid`. The
synthetic fixture is *byte-identical* across runs (seeded RNG) so test runs
are deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_ROOT = _REPO_ROOT / "tests" / "golden_data"
_TINY_DROID = _GOLDEN_ROOT / "tiny_droid"


# Ensure repo root is on sys.path so `from usam... import ...` works without an
# editable install. The pyproject ships a `setuptools` build but we don't want
# to require `pip install -e .` to run tests.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _real_fixture_present() -> bool:
    """True only if the LFS-materialized fixture is present *and* non-empty."""
    if not _TINY_DROID.exists():
        return False
    info = _TINY_DROID / "meta" / "info.json"
    return info.exists() and info.stat().st_size > 0


@pytest.fixture(scope="session")
def tiny_droid_root() -> Path:
    """Path to a working ``tiny_droid`` fixture.

    Materializes a synthetic stand-in if the LFS fixture is missing.
    """
    if _real_fixture_present():
        return _TINY_DROID

    # Synthesize. Import is lazy so a broken synthesizer doesn't break
    # collection of unrelated tests.
    from tests.golden_data._synthesize_tiny_droid import synthesize_tiny_droid

    return synthesize_tiny_droid(_TINY_DROID)
