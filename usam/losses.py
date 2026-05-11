# SPDX-License-Identifier: MIT
"""Unified loss aggregator for USAM.

This module provides:

* :class:`LossWeights` — the eight scalar weights of the unified objective
  (see plan §11.9 and §4.3).
* :class:`USAMUnifiedLoss` — module that combines the four flow-matching
  losses (``action``, ``rgb``, ``depth``, ``flow``) consumed from the MMDiT
  prediction heads with the four auxiliary losses (``geom``, ``flow_act``,
  ``drift``, ``subtask``).

The four flow-matching losses are treated as MSE between the MMDiT
velocity prediction and the corresponding target velocity. LDA-1B's own
flow-matching primitive is structurally identical for the unweighted MSE
case, so this aggregator wraps it directly without re-implementing the
rectified-flow scheduling. The two USAM-specific auxiliary losses
(``geom`` and ``flow_act``) come from :mod:`usam.aux_heads`.

Predictions dictionary keys
---------------------------
The aggregator reads MMDiT outputs by key. The MMDiT exposes
``image_proj_out``, ``action_proj_out``, ``depth_proj_out``, ``flow_proj_out``.
The :meth:`USAMUnifiedLoss.forward` method accepts the ``forward`` output
dict whose entries are gated on the corresponding config flags:

* ``predictions["action"]`` — MMDiT action velocity prediction.
* ``predictions["image"]`` — MMDiT RGB velocity prediction (note: weight
  is named ``rgb`` to match the plan, the input is named ``image`` to
  match the MMDiT head).
* ``predictions["depth"]`` — MMDiT depth velocity prediction.
* ``predictions["flow"]`` — MMDiT flow velocity prediction.
* ``predictions["geom"]`` — dict ``{"depth_dino_pred", "rgb_dino_pred"}``
  passed straight to :class:`GeomConsistencyLoss`.
* ``predictions["flow_act"]`` — dict
  ``{"proprio", "action_chunk", "flow_dino_pred"}`` passed straight to
  :class:`FlowActionConsistencyLoss`.
* ``predictions["drift"]`` — predicted next embedding from the f_drift MLP.
* ``predictions["subtask"]`` — subtask completion logits ``[B, 1]`` or
  ``[B]``.

Targets dictionary keys
-----------------------
* ``targets["action"]``, ``targets["image"]``, ``targets["depth"]``,
  ``targets["flow"]`` — flow-matching targets.
* ``targets["drift"]`` — target embedding (from a fresh Conductor pass).
* ``targets["subtask"]`` — float labels in ``{0, 1}``.

Masks dictionary
----------------
Optional. If ``masks[key]`` is present and non-``None``, the loss is
masked-mean'd along the leading dims; otherwise an unmasked mean is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from usam.aux_heads import FlowActionConsistencyLoss, GeomConsistencyLoss


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    """Scalar weights for the unified loss.

    Defaults follow plan §4.3 (production):

    * ``action = 1.0``
    * ``rgb = 1.0``
    * ``depth = 0.3``
    * ``flow = 0.3``
    * ``geom = 0.0``       (ramped externally between 50K–100K steps)
    * ``flow_act = 0.0``   (same)
    * ``drift = 0.1``
    * ``subtask = 0.1``
    """

    action: float = 1.0
    rgb: float = 1.0
    depth: float = 0.3
    flow: float = 0.3
    geom: float = 0.0
    flow_act: float = 0.0
    drift: float = 0.1
    subtask: float = 0.1

    def as_dict(self) -> Dict[str, float]:
        """Return the weights as a plain ``dict`` keyed by field name."""
        return asdict(self)

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        """Tuple of field names in declaration order."""
        return tuple(f.name for f in fields(cls))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _masked_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    """MSE with optional broadcastable mask.

    Parameters
    ----------
    pred, target : Tensor
        Same shape. Real valued.
    mask : Tensor | None
        ``None`` for an unmasked mean, otherwise a tensor that is
        broadcastable to ``pred``. The mean is ``sum(mask * sq) / sum(mask)``.
    """
    sq = (pred - target).pow(2)
    if mask is None:
        return sq.mean()
    mask = mask.to(sq.dtype)
    # Broadcast against sq.
    while mask.dim() < sq.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1e-8)
    return (mask * sq).sum() / denom


def _flow_match_loss(
    pred: Tensor, target: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    """Velocity-matching loss used for the four MMDiT prediction heads.

    LDA-1B's flow-matching objective at the velocity level is an MSE
    between the network's predicted velocity and the linearly-interpolated
    target velocity. This helper wraps that MSE so the aggregator stays
    independent of the scheduler that produced the targets.
    """
    return _masked_mse(pred, target, mask)


def _bce_loss(logits: Tensor, label: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Binary cross entropy with optional mask. Logits must match label shape."""
    if logits.dim() > label.dim():
        logits = logits.squeeze(-1)
    assert logits.shape == label.shape, (
        f"logits {tuple(logits.shape)} vs label {tuple(label.shape)}"
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, label.to(logits.dtype), reduction="none"
    )
    if mask is None:
        return bce.mean()
    mask = mask.to(bce.dtype)
    denom = mask.sum().clamp_min(1e-8)
    return (mask * bce).sum() / denom


# ---------------------------------------------------------------------------
# Unified loss
# ---------------------------------------------------------------------------
class USAMUnifiedLoss(nn.Module):
    """Combine the eight USAM losses with configurable weights.

    Parameters
    ----------
    weights : LossWeights
        Per-component weights. Components whose weight is exactly ``0`` are
        still computed (so the per-loss log dict is always populated) but
        contribute zero to the total.
    use_gradnorm : bool
        Reserved for the GradNorm loss-balancer. Currently a no-op; the
        aggregator just exposes the flag for the training-engineer. Default
        ``False``.
    geom_kwargs, flow_act_kwargs : dict | None
        Optional constructor overrides for the two owned auxiliary heads.
        See :class:`usam.aux_heads.GeomConsistencyLoss` and
        :class:`usam.aux_heads.FlowActionConsistencyLoss`.

    Shapes (training contract)
    --------------------------
    The exact shapes are flexible — :func:`_flow_match_loss` only asks that
    ``pred`` and ``target`` match and that any mask broadcasts to them.

    Returns
    -------
    (total_loss, per_loss_dict) : tuple[Tensor, dict[str, Tensor]]
        ``per_loss_dict`` keys exactly match :class:`LossWeights` field
        names. Each per-loss entry is the **unweighted** loss; the
        ``total_loss`` is ``sum(weights[k] * per_loss_dict[k])``.
    """

    def __init__(
        self,
        weights: LossWeights,
        use_gradnorm: bool = False,
        geom_kwargs: Optional[Mapping[str, Any]] = None,
        flow_act_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(weights, LossWeights), "weights must be a LossWeights"
        self.weights = weights
        self.use_gradnorm = bool(use_gradnorm)

        self.geom_loss = GeomConsistencyLoss(**(dict(geom_kwargs) if geom_kwargs else {}))
        self.flow_act_loss = FlowActionConsistencyLoss(
            **(dict(flow_act_kwargs) if flow_act_kwargs else {})
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        predictions: Mapping[str, Any],
        targets: Mapping[str, Any],
        masks: Optional[Mapping[str, Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total and per-component losses.

        See module-level docstring for the keys consumed in each dict.
        """
        assert isinstance(predictions, Mapping), "predictions must be a Mapping"
        assert isinstance(targets, Mapping), "targets must be a Mapping"
        masks = masks or {}

        per_loss: Dict[str, Tensor] = {}

        # ---- Flow-matching heads ------------------------------------------------
        per_loss["action"] = self._flow_match(predictions, targets, masks, "action")
        # The plan calls this loss `rgb` even though the MMDiT head is
        # named `image`. We accept either key in `predictions` / `targets`
        # so existing LDA-1B call sites continue to work.
        per_loss["rgb"] = self._flow_match(
            predictions, targets, masks, "rgb", alt_keys=("image",)
        )
        per_loss["depth"] = self._flow_match(predictions, targets, masks, "depth")
        per_loss["flow"] = self._flow_match(predictions, targets, masks, "flow")

        # ---- Auxiliary: geometric consistency ----------------------------------
        # Skip the (expensive, shape-picky) compute when this component's
        # weight is zero — its contribution to the total is zero anyway.
        # This matters during the early ramp (geom=0 for steps < ramp_start)
        # and for smoke runs that explicitly disable an aux head.
        if "geom" in predictions and self.weights.geom != 0.0:
            geom_inputs = predictions["geom"]
            assert isinstance(geom_inputs, Mapping), (
                "predictions['geom'] must be a dict with depth_dino_pred + rgb_dino_pred"
            )
            per_loss["geom"] = self.geom_loss(
                geom_inputs["depth_dino_pred"], geom_inputs["rgb_dino_pred"]
            )
        else:
            per_loss["geom"] = self._zero(predictions)

        # ---- Auxiliary: flow-action consistency --------------------------------
        if "flow_act" in predictions and self.weights.flow_act != 0.0:
            fa_inputs = predictions["flow_act"]
            assert isinstance(fa_inputs, Mapping), (
                "predictions['flow_act'] must be a dict with proprio, action_chunk, flow_dino_pred"
            )
            per_loss["flow_act"] = self.flow_act_loss(
                fa_inputs["proprio"], fa_inputs["action_chunk"], fa_inputs["flow_dino_pred"]
            )
        else:
            per_loss["flow_act"] = self._zero(predictions)

        # ---- Auxiliary: f_drift regression -------------------------------------
        if "drift" in predictions and "drift" in targets:
            per_loss["drift"] = _masked_mse(
                predictions["drift"], targets["drift"], masks.get("drift")
            )
        else:
            per_loss["drift"] = self._zero(predictions)

        # ---- Auxiliary: subtask completion -------------------------------------
        if "subtask" in predictions and "subtask" in targets:
            per_loss["subtask"] = _bce_loss(
                predictions["subtask"], targets["subtask"], masks.get("subtask")
            )
        else:
            per_loss["subtask"] = self._zero(predictions)

        # Sanity: keys exactly match the LossWeights fields.
        expected = set(LossWeights.field_names())
        assert set(per_loss.keys()) == expected, (
            f"per_loss keys mismatch: extras={set(per_loss) - expected}, "
            f"missing={expected - set(per_loss)}"
        )

        # ---- Weighted sum -------------------------------------------------------
        weights = self.weights.as_dict()
        # Use any per-loss tensor as a template for zero (gives us a tensor
        # on the right device / dtype for the running sum).
        template = next(iter(per_loss.values()))
        total = torch.zeros((), dtype=template.dtype, device=template.device)
        for k, v in per_loss.items():
            w = float(weights[k])
            if w == 0.0:
                continue
            total = total + w * v

        return total, per_loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _zero(predictions: Mapping[str, Any]) -> Tensor:
        """Produce a zero scalar on the same device/dtype as some prediction."""
        # Find any tensor to lift the zero onto its device + dtype. We
        # search both top-level prediction values and (one level down) any
        # dict entries that may exist there.
        for v in predictions.values():
            if isinstance(v, Tensor):
                return torch.zeros((), dtype=v.dtype, device=v.device)
            if isinstance(v, Mapping):
                for vv in v.values():
                    if isinstance(vv, Tensor):
                        return torch.zeros((), dtype=vv.dtype, device=vv.device)
        # Fallback: CPU float zero.
        return torch.zeros(())

    @staticmethod
    def _flow_match(
        predictions: Mapping[str, Any],
        targets: Mapping[str, Any],
        masks: Mapping[str, Optional[Tensor]],
        primary_key: str,
        alt_keys: Tuple[str, ...] = (),
    ) -> Tensor:
        """Lookup ``primary_key`` (or any of ``alt_keys``) in pred/target dicts."""
        pred = USAMUnifiedLoss._lookup(predictions, primary_key, alt_keys)
        tgt = USAMUnifiedLoss._lookup(targets, primary_key, alt_keys)
        if pred is None or tgt is None:
            return USAMUnifiedLoss._zero(predictions)
        # Mask is matched on the same key in `masks`, then alt_keys.
        mask = masks.get(primary_key)
        if mask is None:
            for ak in alt_keys:
                if ak in masks:
                    mask = masks[ak]
                    break
        return _flow_match_loss(pred, tgt, mask)

    @staticmethod
    def _lookup(
        d: Mapping[str, Any], primary_key: str, alt_keys: Tuple[str, ...]
    ) -> Optional[Tensor]:
        if primary_key in d:
            v = d[primary_key]
            return v if isinstance(v, Tensor) else None
        for k in alt_keys:
            if k in d:
                v = d[k]
                return v if isinstance(v, Tensor) else None
        return None


__all__ = ["LossWeights", "USAMUnifiedLoss"]
