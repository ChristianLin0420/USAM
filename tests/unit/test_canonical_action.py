# SPDX-License-Identifier: MIT
"""Unit tests for the action canonicalization stage.

Verifies that for **every** embodiment in ``prep/embodiment.json``:
    1. ``canonicalize_action`` returns a finite ``[T, 7]`` tensor
    2. ``validate_action_canonical`` accepts it (bounds OK)
    3. The values are inside the documented ranges

After Phase 2 there are zero stub embodiments — all six (DROID, AgiBot G1,
RH20T, RoboMIND Tien Kung, Bridge WidowX, OXE-AugE generic) round-trip with
real rules. The legacy "expects NotImplementedError on stubs" assertion is
gone; the new ``test_no_phase2_stubs_remain`` flips it: zero stubs allowed.
"""

from __future__ import annotations

import math
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


def _tiny_pose_stream(T: int, seed: int = 11) -> np.ndarray:
    """A small ``[T, 7]`` pose stream: smooth pos + rotvec + gripper.

    Used for ``ee_pose_finite_diff`` (RH20T) round-trip. The pose values are
    chosen so the finite-difference velocity comfortably fits the canonical
    bounds.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T, dtype=np.float32)
    pos = np.stack([0.1 * t, -0.05 * t, 0.02 * t], axis=1)
    rot = np.stack([0.05 * t, -0.05 * t, 0.10 * t], axis=1)
    grip = rng.uniform(0.0, 1.0, size=(T,)).astype(np.float32)
    return np.concatenate([pos, rot, grip[:, None]], axis=1)


def _tiny_prefilled_native(T: int, native_dim: int, seed: int = 13) -> np.ndarray:
    """Action native where the converter has pre-filled canonical EE in cols 0..6."""
    rng = np.random.default_rng(seed)
    a = np.zeros((T, native_dim), dtype=np.float32)
    a[:, 0:3] = rng.uniform(-0.5, 0.5, size=(T, 3))
    a[:, 3:6] = rng.uniform(-1.0, 1.0, size=(T, 3))
    a[:, 6] = rng.uniform(0.0, 1.0, size=(T,))
    if native_dim > 7:
        a[:, 7:] = rng.uniform(-0.1, 0.1, size=(T, native_dim - 7))
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


def test_no_phase2_stubs_remain() -> None:
    """After Phase 2, no embodiment should still be marked as a stub.

    This is the inverse of the Phase 1 ``test_phase2_stubs_raise_*`` check —
    Phase 2 has implemented rules for every embodiment, so the registry must
    carry zero ``_phase2_stub: true`` entries. A reappearing stub indicates
    contract drift and the test should catch it.
    """
    reg = load_embodiment_registry()
    stubs = [name for name, rule in reg.items() if rule.is_stub]
    assert stubs == [], (
        f"expected zero Phase 2 stubs after data-engineer Phase 2; found {stubs}. "
        "If a new embodiment was added, fill in its rule in prep/embodiment.json "
        "and prep/stage_3_canonical.py rather than re-introducing a stub."
    )


def test_registry_round_trip_every_known_embodiment() -> None:
    """Every embodiment must canonicalize cleanly to a [T, 7] tensor in bounds.

    This is the Phase 2 strengthened version of the Phase 1 test — there is
    no NotImplementedError fallback; every rule must succeed. We supply
    realistic per-rule input shapes:

    * ``ee_velocity_passthrough`` (DROID, Bridge): random 7-D velocity stream.
    * ``ee_pose_finite_diff`` (RH20T): smooth pose stream.
    * ``joint_position_to_ee_finite_diff`` / ``joint_delta_to_ee_finite_diff``
      / ``manifest_per_source`` (RoboMIND, AgiBot, OXE-AugE): the converter
      contract pre-fills cols 0..6 with canonical EE; we mirror that here.
    """
    reg = load_embodiment_registry()
    assert "droid_franka" in reg
    assert len(reg) == 6, f"expected 6 embodiments, got {sorted(reg)}"

    T = 8
    for name, rule in reg.items():
        if rule.kind == "ee_velocity_passthrough":
            # Use a random 7-D velocity stream within bounds.
            a_native = _tiny_droid_action_native()[:T]
        elif rule.kind == "ee_pose_finite_diff":
            a_native = _tiny_pose_stream(T)
        elif rule.kind in (
            "joint_position_to_ee_finite_diff",
            "joint_delta_to_ee_finite_diff",
            "manifest_per_source",
        ):
            # Converter pre-fills cols 0..6; pad to native_dim.
            native_dim = max(rule.native_dim, 7)
            a_native = _tiny_prefilled_native(T, native_dim)
        else:
            raise AssertionError(f"unhandled rule kind {rule.kind!r} for {name}")

        canon = canonicalize_action(a_native, name)
        assert canon.shape == (T, 7), f"{name}: bad shape {canon.shape}"
        assert np.isfinite(canon).all(), f"{name}: non-finite output"
        validate_action_canonical(canon)
        # Tighter bounds (per docs/AGENT_CHARTER.md): joint angles in [-π, π],
        # EE position in [-2 m, 2 m]. Our canonical schema stores EE velocity
        # rather than position, so the EE-position bound applies to the linear
        # velocity component (m/s), and the joint-angle bound to angular vel.
        assert (np.abs(canon[:, 0:3]) <= 2.0 + 1e-4).all(), f"{name}: lin vel oob"
        assert (np.abs(canon[:, 3:6]) <= math.pi + 1e-4).all(), f"{name}: ang vel oob"
        assert ((canon[:, 6] >= -1e-4) & (canon[:, 6] <= 1.0 + 1e-4)).all(), (
            f"{name}: gripper out of [0, 1]"
        )


def test_registry_contains_droid_franka() -> None:
    reg = load_embodiment_registry()
    assert "droid_franka" in reg
    rule = reg["droid_franka"]
    assert rule.kind == "ee_velocity_passthrough"
    assert rule.native_dim == 7
    assert rule.ee_dim == 7
    assert not rule.is_stub


def test_rh20t_pose_diff_zero_motion_yields_zero_velocity() -> None:
    """RH20T's finite-diff rule on a constant pose must yield zero velocity."""
    T = 6
    a_native = np.zeros((T, 7), dtype=np.float32)
    a_native[:, 0:3] = 0.1  # constant position
    a_native[:, 3:6] = 0.05  # constant rotvec
    a_native[:, 6] = 0.5  # constant gripper
    canon = canonicalize_action(a_native, "rh20t_franka")
    np.testing.assert_array_almost_equal(canon[:, 0:6], 0.0, decimal=5)
    np.testing.assert_array_almost_equal(canon[:, 6], 0.5, decimal=5)


def test_prefilled_rules_are_passthrough_on_first_seven_columns() -> None:
    """Joint-stream rules must be a scaled passthrough on cols 0..6."""
    for name in ("agibot_g1", "robomind_tien_kung", "oxe_auge_generic"):
        reg = load_embodiment_registry()
        rule = reg[name]
        a_native = _tiny_prefilled_native(5, max(rule.native_dim, 7))
        canon = canonicalize_action(a_native, name)
        # The default scale in embodiment.json is [1, 1, 1, 1, 1, 1, 1] and
        # the values fall inside the canonical bounds, so the result equals
        # the first 7 columns up to the clip operation (which is a no-op here).
        np.testing.assert_array_almost_equal(canon, a_native[:, :7], decimal=5)
