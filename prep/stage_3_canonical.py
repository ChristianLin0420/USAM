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

# All bounds applied here are documented in embodiment.json; we centralize
# clipping in a single helper so every kind has identical post-conditions.
_LIN_BOUND: float = 2.0
_ANG_BOUND: float = math.pi


def _clip_to_canonical_bounds(canon: np.ndarray) -> np.ndarray:
    """Clip the [T, 7] canonical-EE tensor to the documented bounds.

    Linear velocity to ±2 m/s, angular velocity to ±π rad/s, gripper to [0, 1].
    Clipping (rather than rejecting) is appropriate here because (a) finite-
    difference rules can spike on a single bad frame and (b) the validator in
    ``validate_action_canonical`` enforces the same bounds with a tiny epsilon,
    so we want to stay strictly inside.
    """
    assert canon.ndim == 2 and canon.shape[1] == 7, canon.shape
    out = canon.copy()
    out[:, 0:3] = np.clip(out[:, 0:3], -_LIN_BOUND, _LIN_BOUND)
    out[:, 3:6] = np.clip(out[:, 3:6], -_ANG_BOUND, _ANG_BOUND)
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    return out


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


def _canon_ee_pose_finite_diff(
    action_native: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """Pose-stream rule: native action is [pos_xyz, rotvec_xyz, gripper] at fps.

    We finite-difference (forward diff with last-frame replicate) to get a
    velocity-style canonical action. The gripper is NOT differentiated; it is
    passed through as-is and clipped to [0, 1].

    Used for RH20T (fps=10) and provided as a reusable building block for any
    embodiment whose native control loop ships pose targets rather than
    velocities.
    """
    assert action_native.ndim == 2, f"expected [T, D], got {action_native.shape}"
    indices = params["indices"]  # type: ignore[index]
    pos = action_native[:, indices["ee_position_xyz"]].astype(np.float32)  # type: ignore[index]
    rot = action_native[:, indices["ee_rotvec_xyz"]].astype(np.float32)  # type: ignore[index]
    grip = action_native[:, int(indices["gripper"])][:, None].astype(np.float32)  # type: ignore[index]

    fps = float(params.get("fps", 10))
    dt = 1.0 / max(fps, 1e-3)

    T = pos.shape[0]
    if T < 2:
        lin_vel = np.zeros_like(pos)
        ang_vel = np.zeros_like(rot)
    else:
        lin_vel = np.zeros_like(pos)
        ang_vel = np.zeros_like(rot)
        lin_vel[:-1] = (pos[1:] - pos[:-1]) / dt
        lin_vel[-1] = lin_vel[-2]
        # Wrap the rotvec delta into [-π, π] to handle ±π crossings.
        rot_delta = rot[1:] - rot[:-1]
        rot_delta = (rot_delta + math.pi) % (2 * math.pi) - math.pi
        ang_vel[:-1] = rot_delta / dt
        ang_vel[-1] = ang_vel[-2]

    scale = np.asarray(params.get("scale", [1.0] * 7), dtype=np.float32)
    assert scale.shape == (7,), scale.shape
    canon = np.concatenate([lin_vel, ang_vel, grip], axis=1) * scale
    return _clip_to_canonical_bounds(canon)


def _canon_joint_position_to_ee_finite_diff(
    action_native: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """Joint-position stream rule: native columns are joint targets, not EE.

    Pure FK is embodiment-specific (urdf needed). The contract here is that
    the converter has already produced an EE-frame velocity stream and stored
    it in the FIRST 7 padded columns of the parquet's ``action_native``; this
    rule then becomes a finite-difference passthrough on those 7 columns. If
    fewer than 7 columns are present we raise — that means the converter did
    not pre-compute the EE stream, which is a bug.

    Returns
    -------
    np.ndarray
        ``[T, 7]`` fp32, clipped to canonical bounds.
    """
    assert action_native.ndim == 2, f"expected [T, D], got {action_native.shape}"
    if action_native.shape[1] < 7:
        raise ValueError(
            "joint_position_to_ee_finite_diff expects the first 7 columns of "
            "action_native to be the converter-produced EE-velocity stream; "
            f"got only {action_native.shape[1]} columns. The converter for this "
            "embodiment must populate them upstream of stage_3."
        )
    canon = action_native[:, :7].astype(np.float32)
    scale = np.asarray(params.get("scale", [1.0] * 7), dtype=np.float32)
    assert scale.shape == (7,), scale.shape
    return _clip_to_canonical_bounds(canon * scale)


def _canon_joint_delta_to_ee_finite_diff(
    action_native: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """Bimanual joint-delta rule (AgiBot G1): converter pre-fills first 7 cols.

    Same contract as :func:`_canon_joint_position_to_ee_finite_diff`: the
    converter is responsible for producing the canonical EE stream and storing
    it in the first 7 padded columns of ``action_native``. This rule is then a
    scaled passthrough.
    """
    return _canon_joint_position_to_ee_finite_diff(action_native, params)


def _canon_manifest_per_source(
    action_native: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """OXE-AugE rule: per-sub-source manifest selects the format.

    For mixed collections we cannot dispatch by embodiment string alone — the
    OXE-AugE manifest carries a per-source action_format that the converter
    consumed to pre-fill the first 7 padded columns. Stage_3 therefore reads
    those 7 columns directly. Same passthrough contract as the joint-stream
    rules above.
    """
    return _canon_joint_position_to_ee_finite_diff(action_native, params)


_KIND_DISPATCH: Dict[str, Callable[[np.ndarray, Mapping[str, object]], np.ndarray]] = {
    "ee_velocity_passthrough": _canon_ee_velocity_passthrough,
    "ee_pose_finite_diff": _canon_ee_pose_finite_diff,
    "joint_position_to_ee_finite_diff": _canon_joint_position_to_ee_finite_diff,
    "joint_delta_to_ee_finite_diff": _canon_joint_delta_to_ee_finite_diff,
    "manifest_per_source": _canon_manifest_per_source,
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
        # Defensive fallback — the registry should no longer carry stubs after
        # Phase 2, but leave the message here in case someone re-introduces one.
        raise NotImplementedError(
            f"embodiment {embodiment!r} is marked as a Phase 2 stub; "
            "remove the _phase2_stub flag from prep/embodiment.json once a real "
            "rule is in place."
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``python -m prep.stage_3_canonical --dataset droid --chunk 0``.

    Per-chunk canonicalization driver. Walks ``<staged-root>/<dataset>/chunk-NNN/
    ep_*/action_native.npy``, applies the embodiment rule, and writes
    ``action_canonical_ee.npy`` next to the input. This is the per-A100-node
    entry point used by ``slurm/job.sbatch`` (Wave F).
    """
    import argparse
    import logging as _logging
    import os as _os

    parser = argparse.ArgumentParser(
        prog="prep.stage_3_canonical", description=__doc__
    )
    scratch_default = Path(_os.environ.get("USAM_SCRATCH", "/scratch/usam"))
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument(
        "--dataset",
        choices=("droid", "bridge", "agibot2026", "oxe_auge", "rh20t", "robomind"),
        help="Source name (one A100 node per dataset).",
    )
    ds.add_argument(
        "--source",
        dest="dataset",
        choices=("droid", "bridge", "agibot2026", "oxe_auge", "rh20t", "robomind"),
        help="(deprecated) use --dataset",
    )
    parser.add_argument("--chunk", required=True, type=int)
    parser.add_argument(
        "--staged-root",
        type=Path,
        default=scratch_default / "staged",
        help="Root containing <dataset>/chunk-NNN/ep_*/ directories.",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default=None,
        help="Override embodiment key. Defaults to the dataset name's canonical "
             "embodiment (e.g. droid -> droid_franka).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for parity; canonicalization is always per-episode idempotent.",
    )
    args = parser.parse_args(argv)
    _logging.basicConfig(level=_logging.INFO)
    log = _logging.getLogger("prep.stage_3_canonical")

    _DATASET_TO_EMBODIMENT = {
        "droid": "droid_franka",
        "bridge": "bridge_widowx",
        "agibot2026": "agibot_g1",
        "oxe_auge": "oxe_mixed",
        "rh20t": "rh20t_franka",
        "robomind": "robomind_franka",
    }
    embodiment = args.embodiment or _DATASET_TO_EMBODIMENT.get(args.dataset, args.dataset)
    registry = load_embodiment_registry()
    if embodiment not in registry:
        raise SystemExit(
            f"unknown embodiment {embodiment!r} for dataset {args.dataset!r}; "
            f"have {sorted(registry)}"
        )

    chunk_dir = args.staged_root / args.dataset / f"chunk-{args.chunk:03d}"
    if not chunk_dir.exists():
        log.warning("staged chunk dir %s does not exist; nothing to do", chunk_dir)
        return 0

    processed = 0
    for ep_dir in sorted(chunk_dir.glob("ep_*")):
        action_native_path = ep_dir / "action_native.npy"
        if not action_native_path.exists():
            log.debug("skipping %s (no action_native.npy)", ep_dir.name)
            continue
        action_native = np.load(action_native_path)
        canonical = canonicalize_action(action_native, embodiment, registry=registry)
        validate_action_canonical(canonical)
        np.save(ep_dir / "action_canonical_ee.npy", canonical)
        processed += 1
    log.info("canonicalized %d episode(s) under %s", processed, chunk_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())


__all__ = [
    "CanonicalRule",
    "canonicalize_action",
    "load_embodiment_registry",
    "validate_action_canonical",
]
