# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`usam.losses`."""

from __future__ import annotations

import warnings
from dataclasses import fields

import pytest
import torch

from usam.losses import LossWeights, USAMUnifiedLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EXPECTED_KEYS = ("action", "rgb", "depth", "geom", "drift", "subtask")


def _zero_weights(**overrides: float) -> LossWeights:
    """Return ``LossWeights`` with everything 0 except ``overrides``."""
    base = {k: 0.0 for k in EXPECTED_KEYS}
    base.update(overrides)
    return LossWeights(**base)


def _make_loss(weights: LossWeights) -> USAMUnifiedLoss:
    """Build :class:`USAMUnifiedLoss` configured for the test latent dim D=64."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return USAMUnifiedLoss(
            weights=weights,
            geom_kwargs=dict(dim=64, hidden=16),
        )


def _make_predictions_targets() -> tuple[dict, dict]:
    """Make a self-consistent (predictions, targets) pair of test tensors."""
    torch.manual_seed(0)
    b, n_act, d_act = 2, 4, 7
    n_patch, d_lat = 8, 64

    predictions = {
        "action": torch.randn(b, n_act, d_act),
        "image": torch.randn(b, 3, n_patch, d_lat),
        "depth": torch.randn(b, 3, n_patch, d_lat),
        "geom": {
            "depth_dino_pred": torch.randn(b, 3, n_patch, d_lat),
            "rgb_dino_pred": torch.randn(b, 3, n_patch, d_lat),
        },
        "drift": torch.randn(b, 16),
        "subtask": torch.randn(b),
    }
    targets = {
        "action": torch.randn(b, n_act, d_act),
        "image": torch.randn(b, 3, n_patch, d_lat),
        "depth": torch.randn(b, 3, n_patch, d_lat),
        "drift": torch.randn(b, 16),
        "subtask": torch.randint(0, 2, (b,)).float(),
    }
    return predictions, targets


# ---------------------------------------------------------------------------
# LossWeights basics
# ---------------------------------------------------------------------------
def test_loss_weights_field_names() -> None:
    """The dataclass declares exactly the six named weights, in order."""
    names = tuple(f.name for f in fields(LossWeights))
    assert names == EXPECTED_KEYS


def test_loss_weights_defaults_match_plan() -> None:
    """Defaults must match plan §4.3."""
    w = LossWeights()
    assert w.action == 1.0
    assert w.rgb == 1.0
    assert w.depth == 0.3
    assert w.geom == 0.0
    assert w.drift == 0.1
    assert w.subtask == 0.1


# ---------------------------------------------------------------------------
# Output dict invariants
# ---------------------------------------------------------------------------
def test_per_loss_dict_keys_match_field_names() -> None:
    loss_fn = _make_loss(LossWeights())
    preds, tgts = _make_predictions_targets()
    total, per_loss = loss_fn(preds, tgts)
    assert set(per_loss.keys()) == set(EXPECTED_KEYS)
    assert total.dim() == 0


def test_total_is_weighted_sum() -> None:
    """``total`` must equal the dot product of weights and per-loss values."""
    loss_fn = _make_loss(LossWeights())
    preds, tgts = _make_predictions_targets()
    total, per_loss = loss_fn(preds, tgts)
    weights = loss_fn.weights.as_dict()
    expected = sum(float(weights[k]) * per_loss[k] for k in EXPECTED_KEYS)
    assert torch.allclose(total, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Single-component isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("only", EXPECTED_KEYS)
def test_only_one_weight_active(only: str) -> None:
    """When all weights are 0 except ``only``, total == per_loss[only]."""
    weights = _zero_weights(**{only: 1.0})
    loss_fn = _make_loss(weights)
    preds, tgts = _make_predictions_targets()
    total, per_loss = loss_fn(preds, tgts)
    assert torch.allclose(total, per_loss[only], atol=1e-6), (
        f"isolating {only}: total={total.item()} vs component={per_loss[only].item()}"
    )


# ---------------------------------------------------------------------------
# Aux-loss baseline reproduction
# ---------------------------------------------------------------------------
def test_baseline_reproduced_when_aux_zero() -> None:
    """``LossWeights(geom=0)`` reproduces a baseline that drops the aux term.

    We verify this by building the explicit baseline (manually summing the
    five non-aux components with their weights), then toggling ``geom``
    to nonzero and confirming the total moves by exactly the weighted
    geom component.
    """
    preds, tgts = _make_predictions_targets()

    # Baseline (no aux): default LossWeights already has geom = 0.
    baseline_weights = LossWeights()  # geom=0 by default
    loss_fn_baseline = _make_loss(baseline_weights)
    total_baseline, per_loss_baseline = loss_fn_baseline(preds, tgts)

    # Sum the five non-aux weighted contributions explicitly.
    weights = baseline_weights.as_dict()
    five = sum(
        float(weights[k]) * per_loss_baseline[k]
        for k in EXPECTED_KEYS
        if k != "geom"
    )
    assert torch.allclose(total_baseline, five, atol=1e-6), (
        "baseline total should be the sum of the five non-aux components"
    )

    # Now flip on the aux weight and verify the diff.
    aux_weights = LossWeights(geom=0.5)
    loss_fn_aux = _make_loss(aux_weights)
    total_aux, per_loss_aux = loss_fn_aux(preds, tgts)

    diff_expected = 0.5 * per_loss_aux["geom"]
    diff_observed = total_aux - sum(
        float(aux_weights.as_dict()[k]) * per_loss_aux[k]
        for k in EXPECTED_KEYS
        if k != "geom"
    )
    assert torch.allclose(diff_observed, diff_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Mask plumbing
# ---------------------------------------------------------------------------
def test_masks_are_applied() -> None:
    """An all-zero mask should drive masked components to zero."""
    weights = _zero_weights(action=1.0)
    loss_fn = _make_loss(weights)
    preds, tgts = _make_predictions_targets()
    masks = {"action": torch.zeros(2, 4)}  # mask out everything
    total, per_loss = loss_fn(preds, tgts, masks=masks)
    assert torch.allclose(per_loss["action"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(total, torch.zeros(()), atol=1e-6)


def test_image_alias_for_rgb_loss() -> None:
    """The ``image`` MMDiT key feeds into the ``rgb`` loss component."""
    weights = _zero_weights(rgb=1.0)
    loss_fn = _make_loss(weights)
    preds, tgts = _make_predictions_targets()
    # Replace 'image' with 'rgb' on both sides — must match the result.
    preds2 = dict(preds)
    tgts2 = dict(tgts)
    preds2["rgb"] = preds2.pop("image")
    tgts2["rgb"] = tgts2.pop("image")
    total_alias, per_loss_alias = loss_fn(preds, tgts)
    total_direct, per_loss_direct = loss_fn(preds2, tgts2)
    assert torch.allclose(per_loss_alias["rgb"], per_loss_direct["rgb"], atol=1e-6)
    assert torch.allclose(total_alias, total_direct, atol=1e-6)
