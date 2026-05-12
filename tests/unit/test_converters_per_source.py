# SPDX-License-Identifier: MIT
"""Unit tests for the Phase 2 per-source converters.

We exercise each converter's pure transform path against a tiny in-memory
mock raw input — no network, no TFDS, no HDF5. These tests verify:

* The class can be instantiated against an empty/missing raw_root without
  crashing (smoke import + ``list_episodes`` returns empty cleanly).
* The internal transform method (``_*_to_result`` / ``_table_to_result``)
  produces a ``ConversionResult`` whose ``action_canonical_ee`` is finite and
  in-bounds and whose ``action_native`` schema matches the converter's
  stage_3 contract (first 7 cols = canonical EE for the prefilled rules).
* RoboMIND's BGR-detection heuristic correctly classifies obvious BGR/RGB
  inputs and raises on the ambiguous case.

Total runtime is well under the 60 s budget — no I/O, all pure functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from prep._base import EpisodeRef
from prep.stage_2a_to_lerobot.agibot2026 import AgiBot2026Converter, AGIBOT_FPS
from prep.stage_2a_to_lerobot.bridge import BridgeConverter, BRIDGE_FPS
from prep.stage_2a_to_lerobot.rh20t import (
    RH20TConverter,
    RH20T_FPS,
    _select_nearest_indices,
)
from prep.stage_2a_to_lerobot.robomind import (
    RoboMINDConverter,
    ROBOMIND_FPS,
    _bgr_to_rgb,
    _detect_bgr,
)
from prep.stage_3_canonical import validate_action_canonical


# ---------------------------------------------------------------------------
# AgiBot 2026
# ---------------------------------------------------------------------------


def _make_tmp_dir(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_agibot_list_empty_when_no_raw(tmp_path: Path) -> None:
    """No raw_root -> empty enumeration, no crash."""
    out = _make_tmp_dir(tmp_path, "agibot_out")
    conv = AgiBot2026Converter(chunk=0, output_root=out, raw_root=None)
    assert list(conv.list_episodes()) == []


def test_agibot_table_to_result_promotes_instruction_segments(tmp_path: Path) -> None:
    """AgiBot's central job: instruction_segments -> level_1/level_2/level_3."""
    out = _make_tmp_dir(tmp_path, "agibot_out")
    conv = AgiBot2026Converter(chunk=0, output_root=out, raw_root=None)
    T = 12
    # observation.state[14:21] is the right-arm EE pose; pre-fill so the
    # finite-diff produces a sane velocity.
    state = np.zeros((T, 50), dtype=np.float32)
    t_axis = np.linspace(0, 1, T, dtype=np.float32)
    state[:, 14] = 0.05 * t_axis  # position-x sweep
    state[:, 17] = 0.02 * t_axis  # rotvec-x sweep
    state[:, 20] = 0.5  # gripper held
    cols: Dict[str, np.ndarray] = {
        "timestamp": np.arange(T, dtype=np.float32) / float(AGIBOT_FPS),
        "frame_index": np.arange(T, dtype=np.int64),
        "task": np.array(["pick the red cube"], dtype=object),
        "observation.state": state,
        "action": np.zeros((T, 24), dtype=np.float32),
        "instruction_segments": np.array(
            [
                json.dumps(
                    [
                        {"start": 0, "end": 5, "level_1": "pick", "level_2": "approach", "level_3": "x+"},
                        {"start": 5, "end": 12, "level_1": "pick", "level_2": "grasp", "level_3": "close"},
                    ]
                )
            ]
            * T,
            dtype=object,
        ),
    }
    ref = EpisodeRef(
        episode_id="agibot_test_ep_0",
        source="agibot2026",
        raw_path=str(tmp_path),
        extra={"episode_index": 0},
    )
    result = conv._table_to_result(0, cols, ref)
    assert result.action_canonical_ee.shape == (T, 7)
    validate_action_canonical(result.action_canonical_ee)
    assert result.instructions["level_1"] == ["pick"] * T
    assert result.instructions["level_2"][:5] == ["approach"] * 5
    assert result.instructions["level_2"][5:] == ["grasp"] * 7
    assert result.instructions["level_3"][0:5] == ["x+"] * 5
    assert result.instructions["level_3"][5:] == ["close"] * 7
    # subtask_label per-frame should mark the level_2 boundary at frame 5.
    sl = result.raw_meta["subtask_label_per_frame"]
    assert sl[5] is True
    assert sl[4] is False


def test_agibot_action_native_first_seven_cols_are_canonical_velocity(
    tmp_path: Path,
) -> None:
    """AgiBot's stage_3 contract: first 7 padded cols are canonical EE-velocity."""
    out = _make_tmp_dir(tmp_path, "agibot_out")
    conv = AgiBot2026Converter(chunk=0, output_root=out, raw_root=None)
    T = 6
    state = np.zeros((T, 50), dtype=np.float32)
    state[:, 14] = np.linspace(0, 0.06, T)  # right ee position-x
    state[:, 20] = 0.7  # gripper
    cols: Dict[str, np.ndarray] = {
        "timestamp": np.arange(T, dtype=np.float32) / float(AGIBOT_FPS),
        "observation.state": state,
        "action": np.zeros((T, 24), dtype=np.float32),
        "task": np.array(["place the cup"], dtype=object),
    }
    ref = EpisodeRef(
        episode_id="agibot_ep_42",
        source="agibot2026",
        raw_path=str(tmp_path),
        extra={"episode_index": 42},
    )
    result = conv._table_to_result(42, cols, ref)
    # The first 7 padded cols equal the canonical EE-velocity stream.
    np.testing.assert_array_almost_equal(
        result.action_native[:, :7], result.action_canonical_ee, decimal=5
    )


# ---------------------------------------------------------------------------
# RH20T
# ---------------------------------------------------------------------------


def test_rh20t_list_empty_when_no_raw(tmp_path: Path) -> None:
    out = _make_tmp_dir(tmp_path, "rh20t_out")
    conv = RH20TConverter(chunk=0, output_root=out, raw_root=None)
    assert list(conv.list_episodes()) == []


def test_rh20t_json_to_result_finite_diff_velocity(tmp_path: Path) -> None:
    out = _make_tmp_dir(tmp_path, "rh20t_out")
    conv = RH20TConverter(chunk=0, output_root=out, raw_root=None)
    T = 8
    frames = []
    for i in range(T):
        frames.append(
            {
                "timestamp": float(i) / RH20T_FPS,
                "tcp_pose": [0.05 * i, 0.0, 0.0, 0.01 * i, 0.0, 0.0],
                "gripper": 30.0,  # mm, normalized to ~30/85
                "ft": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [0.0, 0.1, 0.0, -0.5, 0.0, 0.5, 0.0],
            }
        )
    action_data = {"task_description": "wipe the table", "frames": frames}
    ep_dir = _make_tmp_dir(tmp_path, "rh20t_ep")
    ref = EpisodeRef(
        episode_id="rh20t_test_ep",
        source="rh20t",
        raw_path=str(ep_dir),
        extra={"config": "RH20T_cfg1"},
    )
    result = conv._json_to_result(ref, ep_dir, action_data)
    assert result.fps == RH20T_FPS
    assert result.action_canonical_ee.shape == (T, 7)
    validate_action_canonical(result.action_canonical_ee)
    # Constant-rate position sweep + constant rotvec sweep -> nonzero lin/ang vel.
    assert np.abs(result.action_canonical_ee[0, 0]) > 0.0  # x-velocity nonzero
    assert result.force_torque is not None
    assert result.force_torque.shape == (T, 6)


def test_rh20t_select_nearest_indices_basic() -> None:
    target = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    source = np.linspace(0.0, 2.0, 21, dtype=np.float32)  # 0.1 s grid
    out = _select_nearest_indices(target, source)
    assert out.shape == (5,)
    # Each target should land on its exact tick.
    np.testing.assert_array_equal(out, np.array([0, 5, 10, 15, 20]))


# ---------------------------------------------------------------------------
# RoboMIND
# ---------------------------------------------------------------------------


def test_robomind_detect_bgr_obvious_blue_skewed() -> None:
    """A blue-skewed frame should be flagged as BGR."""
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[..., 0] = 200  # channel 0 high (blue in BGR layout)
    frame[..., 2] = 100  # channel 2 lower (red in BGR layout)
    assert _detect_bgr(frame) is True


def test_robomind_detect_bgr_obvious_red_skewed() -> None:
    """A red-skewed frame should NOT be flagged as BGR (already RGB)."""
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[..., 0] = 100  # channel 0 (would be R if RGB layout)
    frame[..., 2] = 200  # channel 2 high
    assert _detect_bgr(frame) is False


def test_robomind_detect_bgr_ambiguous_raises() -> None:
    """Equal blue/red means we cannot decide -> hard error."""
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="ambiguous"):
        _detect_bgr(frame, threshold=5.0)


def test_robomind_bgr_to_rgb_swap_is_correct() -> None:
    """Swap is a pure last-axis reverse."""
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    swapped = _bgr_to_rgb(frame)
    np.testing.assert_array_equal(swapped[0, 0], np.array([30, 20, 10]))


def test_robomind_payload_to_result_swaps_bgr(tmp_path: Path) -> None:
    out = _make_tmp_dir(tmp_path, "robomind_out")
    conv = RoboMINDConverter(chunk=0, output_root=out, raw_root=None)
    T = 6
    # Synthesize a BGR-shaped camera buffer (blue dominant).
    cam = np.zeros((T, 8, 8, 3), dtype=np.uint8)
    cam[..., 0] = 200
    cam[..., 2] = 50
    payload: Dict[str, np.ndarray] = {
        "camera::head_cam": cam,
        "obs::joint_position": np.zeros((T, 14), dtype=np.float32),
        "obs::ee_pose": np.column_stack(
            [
                np.linspace(0, 0.05, T, dtype=np.float32),  # pos_x
                np.zeros(T, dtype=np.float32),
                np.zeros(T, dtype=np.float32),
                np.zeros(T, dtype=np.float32),  # rotvec_x
                np.zeros(T, dtype=np.float32),
                np.zeros(T, dtype=np.float32),
                np.full(T, 0.5, dtype=np.float32),  # gripper
            ]
        ),
        "actions": np.zeros((T, 14), dtype=np.float32),
        "language_instruction": np.array(["pick the bowl"], dtype=object),
    }
    ref = EpisodeRef(
        episode_id="robomind_test_ep",
        source="robomind",
        raw_path=str(tmp_path / "fake.h5"),
        extra={"embodiment_dir": "tien_kung"},
    )
    result = conv._payload_to_result(ref, payload)
    # After swap, channel 0 is the (former) blue value -> now interpreted as
    # red in the RGB output, i.e. result has its 0-channel = 50 (was red in BGR).
    head = result.cameras["head_rgb"]
    assert head.shape == cam.shape
    assert int(head[0, 0, 0, 0]) == 50  # was BGR's red (low)
    assert int(head[0, 0, 0, 2]) == 200  # was BGR's blue (high)
    validate_action_canonical(result.action_canonical_ee)


# ---------------------------------------------------------------------------
# Bridge V2
# ---------------------------------------------------------------------------


def test_bridge_list_empty_when_no_tfds(tmp_path: Path) -> None:
    out = _make_tmp_dir(tmp_path, "bridge_out")
    conv = BridgeConverter(chunk=0, output_root=out)
    # tfds may or may not be installed; either way the iterator should not
    # blow up at instantiation. We only check it's iterable.
    refs = list(conv.list_episodes())
    assert isinstance(refs, list)


def test_bridge_tf_episode_to_result_via_synthetic_steps(tmp_path: Path) -> None:
    out = _make_tmp_dir(tmp_path, "bridge_out")
    conv = BridgeConverter(chunk=0, output_root=out)
    T = 5

    class _StepIterable:
        def __init__(self, steps):
            self._steps = steps

        def as_numpy_iterator(self):
            return iter(self._steps)

    steps = []
    for i in range(T):
        steps.append(
            {
                "observation": {
                    "image_0": np.full((4, 4, 3), 100, dtype=np.uint8),
                    "image_2": np.full((4, 4, 3), 50, dtype=np.uint8),
                    "state": np.zeros(7, dtype=np.float32),
                },
                "action": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                "language_instruction": b"open the drawer",
            }
        )

    tf_episode = {"steps": _StepIterable(steps)}
    ref = EpisodeRef(
        episode_id="bridge_test_ep",
        source="bridge",
        raw_path="",
        extra={"episode_index": 7},
    )
    result = conv._tf_episode_to_result(tf_episode, ref)
    assert result.action_canonical_ee.shape == (T, 7)
    validate_action_canonical(result.action_canonical_ee)
    assert result.instructions["level_1"][0] == "open the drawer"
    # head_rgb and wrist_rgb both populated.
    assert "head_rgb" in result.cameras
    assert "wrist_rgb" in result.cameras


