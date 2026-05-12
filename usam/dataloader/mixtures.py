# SPDX-License-Identifier: MIT
"""Per-source sampling weights for the USAM Tier-1 mixture.

This module defines the canonical weight table used by :mod:`usam.dataloader.usam_lerobot`
when multiple per-source datasets are concatenated. The weights below are the
defaults from ``docs/IMPLEMENTATION_PLAN.md §11.12`` and were chosen so each
source contributes a sensible chunk despite huge size differences.

Phase 1 only references the DROID slot; the remaining 5 entries are kept here
so Phase-2 wiring is a single config change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class SourceMixture:
    """One row in the source-mixture table.

    Parameters
    ----------
    name : str
        Short source key (``"droid"``, ``"agibot2026"``, ...). Must be unique
        within a mixture list.
    repo_id : str
        HuggingFace repo id. Format ``"<org>/usam-<source>"``.
    weight : float
        Non-negative sampling weight. The dataloader normalizes weights to sum
        to 1 before drawing samples.
    """

    name: str
    repo_id: str
    weight: float


DEFAULT_TIER1_MIX: List[SourceMixture] = [
    SourceMixture("droid", "<org>/usam-droid", 0.15),
    SourceMixture("agibot2026", "<org>/usam-agibot2026", 0.30),
    SourceMixture("rh20t", "<org>/usam-rh20t", 0.20),
    SourceMixture("robomind", "<org>/usam-robomind", 0.15),
    SourceMixture("bridge", "<org>/usam-bridge", 0.10),
]


PHASE1_DROID_ONLY_MIX: List[SourceMixture] = [
    SourceMixture("droid", "<org>/usam-droid", 1.0),
]


def normalize_weights(mixture: Sequence[SourceMixture]) -> torch.Tensor:
    """Return a 1-D fp32 tensor of normalized sampling weights.

    Parameters
    ----------
    mixture : sequence of SourceMixture
        Input mixture. Must contain at least one element with positive weight.

    Returns
    -------
    torch.Tensor
        1-D fp32 tensor of length ``len(mixture)`` summing to 1.0.
    """
    assert len(mixture) > 0, "empty mixture"
    raw = torch.tensor([m.weight for m in mixture], dtype=torch.float32)
    assert (raw >= 0).all(), "weights must be non-negative"
    total = float(raw.sum())
    assert total > 0, "mixture weights sum to zero"
    return raw / total


def make_weighted_sampler(
    per_dataset_lengths: Sequence[int],
    mixture: Sequence[SourceMixture],
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.utils.data.WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` over a concatenated dataset.

    Parameters
    ----------
    per_dataset_lengths : sequence of int
        ``len(dataset_i)`` in the same order as ``mixture``.
    mixture : sequence of SourceMixture
        Source mixture; ``len(mixture) == len(per_dataset_lengths)``.
    num_samples : int
        How many indices to draw per epoch.
    generator : torch.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        Sampler that draws indices into the concatenated dataset such that the
        empirical source distribution matches ``mixture``.
    """
    from torch.utils.data import WeightedRandomSampler

    assert len(per_dataset_lengths) == len(mixture)
    assert all(L > 0 for L in per_dataset_lengths), "empty sub-dataset"
    assert num_samples > 0

    src_w = normalize_weights(mixture)
    # within source: uniform over its own length; cross-source: source weight
    weights: List[float] = []
    for L, w in zip(per_dataset_lengths, src_w.tolist()):
        per_item = w / L
        weights.extend([per_item] * L)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def mixture_summary(
    mixture: Sequence[SourceMixture],
    per_dataset_lengths: Sequence[int] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Human-readable summary of a mixture for logging.

    Returns a dict ``{name: {"weight": float, "items": int | None}}``.
    """
    out: Dict[str, Dict[str, float]] = {}
    src_w = normalize_weights(mixture)
    for i, m in enumerate(mixture):
        out[m.name] = {
            "weight": float(src_w[i]),
            "items": float(per_dataset_lengths[i]) if per_dataset_lengths else float("nan"),
        }
    return out


def filter_mixture(
    mixture: Iterable[SourceMixture], names: Iterable[str]
) -> List[SourceMixture]:
    """Keep only entries whose ``name`` is in ``names`` and renormalize.

    Returns a new list with weights re-normalized to sum to 1.
    """
    keep = [m for m in mixture if m.name in set(names)]
    assert len(keep) > 0, f"no mixture entries match {list(names)}"
    total = sum(m.weight for m in keep)
    assert total > 0
    return [SourceMixture(m.name, m.repo_id, m.weight / total) for m in keep]
