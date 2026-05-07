# SPDX-License-Identifier: MIT
"""Stage 3: action canonicalization.

Converts each embodiment's ``action_native`` into a unified ``action_canonical_ee``
of length 7: ``[lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z, gripper]``.

Phase 1 implements the DROID rule (a simple passthrough since DROID's native
action is already a 7-D EE-velocity vector). The other embodiments are
placeholders that ``raise NotImplementedError`` with a clear message; they will
be filled in during Phase 2.

Bounds (joint angles in ``[-pi, pi]``, EE position/velocity in ``[-2, 2]``) are
enforced post-hoc by ``validate_action_canonical``; converters should not
silently clip — they must produce values inside the bounds by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np


_EMBODIMENT_PATH = Path(__file__).resolve().parent / "embodiment.json"


@dataclass(frozen=True)
class CanonicalRule:
    """One row from ``embodiment.json``.

    Attributes
    ----------
    name : str
    kind : str
        One of ``ee_velocity_passthrough``, ``joint_to_ee_fk_stub``,
        ``manifest_per_source``.
    native_dim : int
    ee_dim : int
        Always 7 for USAM-canonical-EE.
    params : dict
        Kind-specific parameters.
    is_stub : bool
        True if this is a Phase 2 placeholder.
    """

    name: str
    kind: str
    native_dim: int
    ee_dim: int
    params: Mapping[str, object]
    is_stub: bool = False


def load_embodiment_registry(path: Path | None = None) -> Dict[str, CanonicalRule]:
    """Parse ``embodiment.json`` into a dict of rules.

    Parameters
    ----------
    path : Path | None
        Optional override; defaults to ``prep/embodiment.json``.

    Returns
    -------
    dict[str, CanonicalRule]
    """
    p = path or _EMBODIMENT_PATH
    assert p.exists(), f"missing embodiment registry at {p}"
    raw = json.loads(p.read_text())
    out: Dict[str, CanonicalRule] = {}
    for name, body in raw["embodiments"].items():
        out[name] = CanonicalRule(
            name=name,
            kind=str(body["kind"]),
            native_dim=int(body["native_dim"]),
            ee_dim=int(body["ee_dim"]),
            params={k: v for k, v in body.items() if k not in {"kind", "native_dim", "ee_dim"}},
            is_stub=bool(body.get("_phase2_stub", False)),
        )
    return out


# ----- per-kind canonicalizers ------------------------------------------------

def _canon_ee_velocity_passthrough(
    action_native: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """DROID-style: action is already 7-D EE velocity + grip. Just slice + scale.

    Parameters
    ----------
    action_native : np.ndarray
        Shape ``[T, native_dim]`` fp32.
    params : Mapping
        Reads ``indices.linear_velocity_xyz``, ``indices.angular_velocity_xyz``,
        ``indices.gripper``, ``scale``.

    Returns
    -------
    np.ndarray
        ``[T, 7]`` fp32.
    """
    assert action_native.ndim == 2, f"expected [T, D], got {action_native.shape}"
    indices = params["indices"]  # type: ignore[index]
    lin = action_native[:, indices["linear_velocity_xyz"]]  # type: ignore[index]
    ang = action_native[:, indices["angular_velocity_xyz"]]  # type: ignore[index]
    grip_idx = indices["gripper"]  # type: ignore[index]
    grip = action_native[:, int(grip_idx)][:, None]
    canonical = np.concatenate([lin, ang, grip], axis=1).astype(np.float32)

    scale = np.asarray(params.get("scale", [1.0] * 7), dtype=np.float32)
    assert scale.shape == (7,), scale.shape
    return canonical * scale


def _canon_phase2_stub(action_native: np.ndarray, params: Mapping[str, object]) -> np.ndarray:
    raise NotImplementedError(
        "This embodiment's canonicalization is a Phase 2 deliverable. "
        "See docs/IMPLEMENTATION_PLAN.md §11.18."
    )


_KIND_DISPATCH: Dict[str, Callable[[np.ndarray, Mapping[str, object]], np.ndarray]] = {
    "ee_velocity_passthrough": _canon_ee_velocity_passthrough,
    "joint_to_ee_fk_stub": _canon_phase2_stub,
    "manifest_per_source": _canon_phase2_stub,
}


# ----- public API -------------------------------------------------------------

def canonicalize_action(
    action_native: np.ndarray,
    embodiment: str,
    registry: Dict[str, CanonicalRule] | None = None,
) -> np.ndarray:
    """Convert ``action_native`` to ``action_canonical_ee[T, 7]``.

    Parameters
    ----------
    action_native : np.ndarray
        Shape ``[T, native_dim]`` fp32. ``native_dim`` must match the registry.
    embodiment : str
        Embodiment key (e.g. ``"droid_franka"``).
    registry : dict | None
        Pre-loaded registry; if ``None`` we load the default.

    Returns
    -------
    np.ndarray
        ``[T, 7]`` fp32.
    """
    assert isinstance(action_native, np.ndarray)
    assert action_native.ndim == 2, action_native.shape
    reg = registry or load_embodiment_registry()
    if embodiment not in reg:
        raise KeyError(f"unknown embodiment {embodiment!r}; have {sorted(reg)}")
    rule = reg[embodiment]
    if rule.is_stub:
        raise NotImplementedError(
            f"embodiment {embodiment!r} is a Phase 2 stub; see prep.stage_3_canonical"
        )
    if action_native.shape[1] < rule.native_dim:
        raise ValueError(
            f"action_native has {action_native.shape[1]} dims but embodiment "
            f"{embodiment!r} expects at least {rule.native_dim}"
        )
    fn = _KIND_DISPATCH[rule.kind]
    out = fn(action_native[:, : rule.native_dim].astype(np.float32), rule.params)
    assert out.shape == (action_native.shape[0], 7), out.shape
    return out


def validate_action_canonical(action_canonical_ee: np.ndarray) -> None:
    """Hard-assert a canonical action chunk is well-formed.

    Raises
    ------
    AssertionError
        On NaN/Inf, on shape mismatch, or on out-of-bounds values:
        ``|lin_vel| <= 2.0``, ``|ang_vel| <= pi``, ``gripper in [0, 1]``.
    """
    assert isinstance(action_canonical_ee, np.ndarray)
    assert action_canonical_ee.ndim == 2 and action_canonical_ee.shape[1] == 7, (
        f"expected [T, 7], got {action_canonical_ee.shape}"
    )
    assert np.isfinite(action_canonical_ee).all(), "non-finite values in canonical action"
    lin = action_canonical_ee[:, 0:3]
    ang = action_canonical_ee[:, 3:6]
    grip = action_canonical_ee[:, 6]
    assert (np.abs(lin) <= 2.0 + 1e-4).all(), "linear velocity out of bounds"
    assert (np.abs(ang) <= math.pi + 1e-4).all(), "angular velocity out of bounds"
    assert ((grip >= -1e-4) & (grip <= 1.0 + 1e-4)).all(), "gripper out of [0, 1]"


__all__ = [
    "CanonicalRule",
    "canonicalize_action",
    "load_embodiment_registry",
    "validate_action_canonical",
]
