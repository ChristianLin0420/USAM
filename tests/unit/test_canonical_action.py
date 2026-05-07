# SPDX-License-Identifier: MIT
"""Unit tests for the action canonicalization stage.

Verifies that for every non-stub embodiment in ``prep/embodiment.json``:
    1. ``canonicalize_action`` returns a finite ``[T, 7]`` tensor
    2. ``validate_action_canonical`` accepts it (bounds OK)
    3. The values are inside the documented ranges

Phase 2 stubs are checked to raise ``NotImplementedError`` cleanly so that a
silent contract drift trips the test rather than silently shipping all-zero
canonical actions for those embodiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prep.stage_3_canonical import (
    canonicalize_action,
    load_embodiment_registry,
    validate_action_canonical,
)


def _tiny_droid_action_native(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((30, 7), dtype=np.float32)
    a[:, 0:3] = rng.uniform(-0.5, 0.5, size=(30, 3))
    a[:, 3:6] = rng.uniform(-1.0, 1.0, size=(30, 3))
    a[:, 6] = rng.uniform(0.0, 1.0, size=(30,))
    return a


def test_droid_canonical_action_finite_and_in_bounds() -> None:
    a_native = _tiny_droid_action_native()
    a_canon = canonicalize_action(a_native, "droid_franka")
    assert a_canon.shape == (30, 7)
    assert np.isfinite(a_canon).all()
    # bounds
    assert (np.abs(a_canon[:, 0:3]) <= 2.0).all(), "lin vel oob"
    assert (np.abs(a_canon[:, 3:6]) <= np.pi).all(), "ang vel oob"
    assert ((a_canon[:, 6] >= 0.0) & (a_canon[:, 6] <= 1.0)).all(), "gripper oob"
    # validator agrees
    validate_action_canonical(a_canon)


def test_droid_canonical_action_passthrough_preserves_values() -> None:
    """DROID's rule is a pure passthrough — values must round-trip identically."""
    a_native = _tiny_droid_action_native(seed=11)
    a_canon = canonicalize_action(a_native, "droid_franka")
    np.testing.assert_array_almost_equal(a_canon, a_native, decimal=6)


def test_validator_rejects_oob_values() -> None:
    bad = np.zeros((4, 7), dtype=np.float32)
    bad[0, 0] = 5.0  # lin vel out of bounds
    with pytest.raises(AssertionError):
        validate_action_canonical(bad)


def test_validator_rejects_non_finite() -> None:
    bad = np.zeros((4, 7), dtype=np.float32)
    bad[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        validate_action_canonical(bad)


def test_phase2_stubs_raise_not_implemented_error() -> None:
    reg = load_embodiment_registry()
    stubs = [name for name, rule in reg.items() if rule.is_stub]
    assert len(stubs) >= 1, "expected some Phase 2 stub embodiments in registry"
    for name in stubs:
        rule = reg[name]
        a = np.zeros((4, rule.native_dim), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            canonicalize_action(a, name)


def test_registry_round_trip_every_known_embodiment() -> None:
    """For each registered embodiment, either canonicalize cleanly or raise
    NotImplementedError. Anything else is a contract drift."""
    reg = load_embodiment_registry()
    assert "droid_franka" in reg
    for name, rule in reg.items():
        a_native = np.zeros((5, max(rule.native_dim, 7)), dtype=np.float32)
        if name == "droid_franka":
            a_native = _tiny_droid_action_native()[:5]
        try:
            canon = canonicalize_action(a_native, name)
        except NotImplementedError:
            assert rule.is_stub, f"{name}: NotImplementedError but not marked as stub"
            continue
        else:
            assert canon.shape == (a_native.shape[0], 7)
            validate_action_canonical(canon)


def test_registry_contains_droid_franka() -> None:
    reg = load_embodiment_registry()
    assert "droid_franka" in reg
    rule = reg["droid_franka"]
    assert rule.kind == "ee_velocity_passthrough"
    assert rule.native_dim == 7
    assert rule.ee_dim == 7
    assert not rule.is_stub
