# SPDX-License-Identifier: MIT
"""LoRA factory for DINOv3 attention modules.

This module provides a thin LoRA implementation tailored to USAM's needs:

* It wraps an existing ``nn.Linear`` (typically the fused QKV projection or
  separate Q/K/V projections inside a DINOv3 attention block) without
  modifying its weights.
* It supports **multiple modality paths** sharing the same frozen base. We
  use this so that the same DINOv3 backbone can handle RGB, depth, and
  optical flow with cheap modality-specific deltas.
* The base ``nn.Linear`` is frozen by ``apply_lora``; only the LoRA A/B
  matrices receive gradients.

Reference: https://github.com/RobvanGastel/dinov3-finetune
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Module names targeted by ``apply_lora`` for DINOv3 attention blocks.
# ---------------------------------------------------------------------------
DINOV3_QKV_TARGETS: tuple[str, ...] = ("attention.query", "attention.key", "attention.value")
"""Default Q/K/V projection module names for HF DINOv3 transformers."""


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` wrapped with one LoRA path per modality id.

    The forward computes ``base(x) + scale * (B_m A_m x)`` for the
    currently selected modality ``m``. When ``modality_id is None`` (or the
    LoRA matrices are zero), the wrapper is mathematically identical to the
    base linear.

    Parameters
    ----------
    base : nn.Linear
        Pre-existing linear layer. Its weights are frozen in-place
        (``requires_grad_(False)``); the wrapper holds it as a submodule.
    r : int
        LoRA rank. Must be a positive integer (typically 8).
    modality_ids : sequence of str
        Identifiers for the modality-specific LoRA paths to instantiate.
        Each id gets its own ``A`` and ``B`` matrix.
    alpha : float, optional
        LoRA scaling factor. Effective scale is ``alpha / r``. Defaults to
        ``r`` (i.e. unit scale), matching the dinov3-finetune recipe.
    dropout : float, optional
        Dropout applied to ``A x`` before the up-projection. Defaults to
        ``0.0``.

    Notes
    -----
    * ``A`` is initialized with Kaiming-uniform like the original LoRA paper
      and ``B`` is zero-initialized so that at step 0 the wrapped module
      reproduces the base output bit-exactly.
    * The base layer's parameters are *frozen*. Adding more parameter
      updates after construction (e.g. by calling
      ``base.weight.requires_grad_(True)`` from outside) is not supported.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        modality_ids: Sequence[str],
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear), f"LoRALinear expects nn.Linear base, got {type(base)}"
        assert r > 0, f"LoRA rank must be positive, got {r}"
        assert len(modality_ids) > 0, "At least one modality_id is required"
        assert len(set(modality_ids)) == len(modality_ids), "modality_ids must be unique"

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / self.r
        self.modality_ids = tuple(modality_ids)

        # Freeze the base in-place. The wrapper still owns trainable LoRA params.
        for p in self.base.parameters():
            p.requires_grad_(False)

        # One A, one B per modality. Stored as ParameterDicts so they show
        # up in ``named_parameters`` with informative keys.
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for mid in self.modality_ids:
            a = nn.Parameter(torch.empty(self.r, self.in_features))
            b = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A[mid] = a
            self.lora_B[mid] = b

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, modality_id: str | None = None) -> Tensor:  # noqa: D401
        """Apply the wrapped linear plus the modality-specific LoRA delta.

        Parameters
        ----------
        x : Tensor
            Shape ``[..., in_features]``.
        modality_id : str or None
            One of ``self.modality_ids``. When ``None`` only the base
            forward is returned, which is useful for sanity-checking
            zero-LoRA equivalence.

        Returns
        -------
        Tensor
            Shape ``[..., out_features]``.
        """
        assert x.shape[-1] == self.in_features, (
            f"input feature dim mismatch: expected {self.in_features}, got {x.shape[-1]}"
        )
        out = self.base(x)
        if modality_id is None:
            return out
        if modality_id not in self.lora_A:
            raise KeyError(
                f"unknown modality_id={modality_id!r}; expected one of {self.modality_ids}"
            )
        a = self.lora_A[modality_id]
        b = self.lora_B[modality_id]
        # x [..., in], a [r, in] -> [..., r]; then b [out, r] -> [..., out]
        delta = self.dropout(x @ a.t()) @ b.t()
        return out + self.scaling * delta

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only the trainable LoRA parameters (A and B)."""
        for p in self.lora_A.parameters():
            yield p
        for p in self.lora_B.parameters():
            yield p

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, modalities={self.modality_ids}"
        )


# ---------------------------------------------------------------------------
# Apply-lora helper
# ---------------------------------------------------------------------------
def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Return ``(parent_module, child_attr)`` for ``model.<dotted_name>``."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    r: int,
    target_module_names: Sequence[str],
    modality_ids: Sequence[str],
    alpha: float | None = None,
    dropout: float = 0.0,
) -> dict[str, list[LoRALinear]]:
    """Wrap every linear sub-module whose name ends with one of the given suffixes.

    The function walks ``model.named_modules()`` and for each ``nn.Linear``
    whose name endswith one of ``target_module_names`` (e.g. ``"attention.query"``,
    ``"attention.key"``, ``"attention.value"``, or simply ``"qkv"``), it
    replaces the linear in its parent with a :class:`LoRALinear` that holds
    one LoRA path per modality id.

    Parameters
    ----------
    model : nn.Module
        The model to mutate in place. The base linears are frozen; all
        other parameters of ``model`` are left untouched (caller is
        responsible for whatever broader freeze policy applies).
    r : int
        LoRA rank.
    target_module_names : sequence of str
        Module name suffixes to wrap. The match is on the **fully qualified
        dotted name** ending with one of the entries, so passing
        ``["query", "key", "value"]`` will catch every Q/K/V projection.
    modality_ids : sequence of str
        E.g. ``["rgb", "depth", "flow"]``. One LoRA path per id.
    alpha : float, optional
        Forwarded to :class:`LoRALinear`.
    dropout : float, optional
        Forwarded to :class:`LoRALinear`.

    Returns
    -------
    dict[str, list[LoRALinear]]
        Mapping ``modality_id -> [LoRALinear, ...]``. The list contains
        every wrapper containing a path for that modality (i.e. every
        wrapper this function created). Useful for building parameter
        groups per modality if needed; also used by tests to check that
        gradients flow only into LoRA params.
    """
    assert r > 0, f"LoRA rank must be positive, got {r}"
    assert len(target_module_names) > 0, "target_module_names must be non-empty"
    assert len(modality_ids) > 0, "modality_ids must be non-empty"

    # Snapshot names first; we mutate the module tree below. The match is
    # on the **last dotted segment** of the qualified name, so passing
    # ``["query", "key", "value"]`` matches ``encoder.layer.0.attention.query``
    # but not ``encoder.layer.0.attention.subquery``.
    targets: list[tuple[str, nn.Linear]] = []
    suffix_set = set(target_module_names)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name:
            continue
        leaf = name.split(".")[-1]
        if leaf in suffix_set:
            targets.append((name, module))

    wrappers: dict[str, list[LoRALinear]] = {mid: [] for mid in modality_ids}
    for name, linear in targets:
        parent, child_attr = _resolve_parent(model, name)
        wrapper = LoRALinear(
            base=linear,
            r=r,
            modality_ids=modality_ids,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, child_attr, wrapper)
        for mid in modality_ids:
            wrappers[mid].append(wrapper)

    return wrappers


__all__ = ["LoRALinear", "apply_lora", "DINOV3_QKV_TARGETS"]
