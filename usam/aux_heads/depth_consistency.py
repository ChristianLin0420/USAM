# SPDX-License-Identifier: MIT
"""Depth-RGB geometric consistency loss.

The :class:`GeomConsistencyLoss` enforces that the **predicted depth-DINO**
latents and the **predicted RGB-DINO** latents agree on their per-patch
ordering of "near-field-ness". Two frozen, lightweight components decode
each modality into a per-patch scalar:

* ``dav2_distill_mlp`` — a 2-layer MLP (~50K params) distilled from
  Depth-Anything-V2 in Phase A.5. Maps a depth-DINO latent ``[D]`` to a
  scalar inverse-depth proxy.
* ``nearfield_proto`` — a frozen "near-field" RGB prototype (~50K params,
  modelled as a 2-layer head followed by a unit-norm prototype). Maps an
  RGB-DINO latent ``[D]`` to a cosine similarity against the prototype.

The two scalar streams are compared with a **differentiable Spearman rank
correlation** (soft-rank via sigmoid pairwise comparisons; see
:func:`soft_rank`). The returned loss is ``-rho``: a negative correlation
becomes a positive loss, perfect agreement is ``-1``, perfect anti-agreement
is ``+1``. The result is bounded in ``[-1, 1]``.

Reference for the soft-rank trick: Blondel et al., 2020 (Fast Differentiable
Sorting and Ranking) and the simpler sigmoid-pairwise relaxation widely used
for differentiable Spearman. We hand-roll the relaxation here so we don't
take a dependency on ``differentiable-rank``. (If a future iteration prefers
the Blondel formulation, ``differentiable-rank`` would need to be added to
``requirements/train.txt`` — flagged for the infra-engineer.)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Soft-rank / Soft-Spearman primitives
# ---------------------------------------------------------------------------
def soft_rank(x: Tensor, tau: float = 1.0) -> Tensor:
    """Differentiable rank approximation via sigmoid pairwise comparisons.

    For a 1-D vector ``x`` of length ``N``, the (1-indexed) soft rank of
    element ``i`` is

    .. math::
       r_i = 1 + \\sum_{j \\neq i} \\sigma\\big((x_j - x_i)/\\tau\\big).

    As ``tau -> 0`` this converges to the true rank; for finite ``tau`` it
    is a smooth surrogate amenable to autograd. Output ranks are floats in
    ``[1, N]``.

    Parameters
    ----------
    x : Tensor
        Input of shape ``[..., N]``.
    tau : float
        Temperature. Must be positive. Smaller = sharper ranks but noisier
        gradient; ``1.0`` is a robust default for inputs roughly in the
        unit range.

    Returns
    -------
    Tensor
        Same shape as ``x``; soft rank values in ``[1, N]``.
    """
    assert tau > 0, "tau must be positive"
    assert x.dim() >= 1, "soft_rank expects at least a 1-D tensor"
    # Pairwise differences: out[..., i, j] = x[..., j] - x[..., i].
    diffs = x.unsqueeze(-2) - x.unsqueeze(-1)
    # sigmoid((x_j - x_i) / tau) -> high when j > i, low when j < i.
    s = torch.sigmoid(diffs / tau)
    # Sum over j; subtract the j == i term (which is 0.5) and add 1 for
    # 1-based ranking.
    rank = s.sum(dim=-1) - 0.5 + 1.0
    # The constant 1 just shifts the ranks; for Spearman we de-mean
    # afterwards so it does not matter, but keeping it makes the values
    # interpretable for callers.
    assert rank.shape == x.shape
    return rank


def soft_spearman(a: Tensor, b: Tensor, tau: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Differentiable Spearman rank correlation between ``a`` and ``b``.

    Both inputs must have the same shape ``[..., N]``; the correlation is
    computed along the last dimension and the leading dimensions are
    averaged into a scalar.

    Parameters
    ----------
    a, b : Tensor
        Real-valued tensors with matching shape. Last dimension is the
        sample dimension across which ranks are computed.
    tau : float
        Temperature passed to :func:`soft_rank`.
    eps : float
        Numerical floor for the correlation denominator.

    Returns
    -------
    Tensor
        Scalar mean correlation in approximately ``[-1, 1]``. The bound is
        ``[-1, 1]`` exactly in the limit ``tau -> 0``; for finite ``tau``
        the values are slightly contracted but stay strictly inside.
    """
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    assert a.shape[-1] >= 2, "need at least 2 samples for a rank correlation"

    ra = soft_rank(a, tau=tau)
    rb = soft_rank(b, tau=tau)

    ra_c = ra - ra.mean(dim=-1, keepdim=True)
    rb_c = rb - rb.mean(dim=-1, keepdim=True)

    num = (ra_c * rb_c).sum(dim=-1)
    den = torch.sqrt((ra_c.pow(2).sum(dim=-1) + eps) * (rb_c.pow(2).sum(dim=-1) + eps))
    rho = num / den
    return rho.mean()


# ---------------------------------------------------------------------------
# Frozen depth / RGB heads
# ---------------------------------------------------------------------------
class _FrozenDepthDistill(nn.Module):
    """Frozen 2-layer MLP that decodes a depth-DINO latent to an inverse-depth
    proxy.

    Parameters are initialised then immediately frozen
    (``requires_grad=False``). When :class:`GeomConsistencyLoss` receives a
    real ``dav2_distill_ckpt`` path the weights are loaded from that file;
    otherwise the random initialisation is used. The total parameter count
    targets ~50K.
    """

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        # 768*64 + 64 + 64 + 1 ≈ 49K parameters at D=768, hidden=64.
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., D] -> [..., 1] -> [...]
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class _FrozenNearfieldProto(nn.Module):
    """Frozen near-field RGB prototype.

    Projects an RGB-DINO latent ``[D]`` to a small embedding ``[H]`` and
    returns its cosine similarity with a learned (then frozen) prototype.

    Parameters
    ----------
    dim : int
        Latent dimensionality.
    hidden : int
        Embedding dimensionality. Default 64 -> ~49K parameters at D=768.
    """

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # Prototype is a unit-norm direction in the embedding space.
        self.prototype = nn.Parameter(torch.randn(hidden))

    def forward(self, x: Tensor) -> Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.fc2(h)
        # Cosine similarity with the prototype.
        h_n = torch.nn.functional.normalize(h, dim=-1)
        p_n = torch.nn.functional.normalize(self.prototype, dim=0)
        return (h_n * p_n).sum(dim=-1)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Public loss
# ---------------------------------------------------------------------------
class GeomConsistencyLoss(nn.Module):
    """Soft-Spearman geometric consistency loss ``L_geom``.

    Implements §11.7 of ``docs/IMPLEMENTATION_PLAN.md``:

    * ``decode_depth(depth_dino_pred)`` — frozen DAv2-distill MLP,
      latent -> inverse-depth proxy per patch.
    * ``nearfield_cos(rgb_dino_pred)`` — frozen RGB near-field prototype,
      latent -> cosine similarity per patch.
    * Compare both per-patch streams via differentiable Spearman rank
      correlation.

    The returned loss is ``-rho``: low (negative) values indicate strong
    rank agreement, the desired outcome. Bounded in ``[-1, 1]``.

    Parameters
    ----------
    dim : int
        Latent dimensionality ``D`` (must match the Tri-DINO embed_dim).
    hidden : int
        Width of the two frozen heads. The ~50K-param targets in the plan
        come from ``hidden=64`` at ``dim=768``.
    dav2_distill_ckpt : str | Path | None
        Optional path to a saved ``state_dict`` for the DAv2-distill MLP.
        If ``None``, a warning is emitted and random weights are used.
        This branch is intended for unit testing only.
    nearfield_proto_ckpt : str | Path | None
        Same, for the near-field prototype.
    tau : float
        Soft-rank temperature.

    Notes
    -----
    The DAv2-distill MLP and near-field prototype have ``requires_grad=False``
    on every parameter as soon as :meth:`__init__` returns; gradients flow
    only through ``depth_dino_pred`` / ``rgb_dino_pred``.
    """

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 64,
        dav2_distill_ckpt: Optional[str | Path] = None,
        nearfield_proto_ckpt: Optional[str | Path] = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim > 0, "dim must be positive"
        assert hidden > 0, "hidden must be positive"
        assert tau > 0, "tau must be positive"

        self.dim = dim
        self.hidden = hidden
        self.tau = tau

        self.dav2_mlp = _FrozenDepthDistill(dim=dim, hidden=hidden)
        self.nearfield_proto = _FrozenNearfieldProto(dim=dim, hidden=hidden)

        if dav2_distill_ckpt is not None:
            state = torch.load(str(dav2_distill_ckpt), map_location="cpu")
            self.dav2_mlp.load_state_dict(state)
        else:
            warnings.warn(
                "GeomConsistencyLoss: no `dav2_distill_ckpt` supplied; the "
                "DAv2-distill MLP is randomly initialised. This is only "
                "appropriate for unit tests.",
                stacklevel=2,
            )

        if nearfield_proto_ckpt is not None:
            state = torch.load(str(nearfield_proto_ckpt), map_location="cpu")
            self.nearfield_proto.load_state_dict(state)
        else:
            warnings.warn(
                "GeomConsistencyLoss: no `nearfield_proto_ckpt` supplied; the "
                "near-field prototype is randomly initialised. This is only "
                "appropriate for unit tests.",
                stacklevel=2,
            )

        # Freeze immediately and verify.
        _freeze(self.dav2_mlp)
        _freeze(self.nearfield_proto)
        assert all(not p.requires_grad for p in self.dav2_mlp.parameters()), (
            "dav2_mlp must be frozen"
        )
        assert all(not p.requires_grad for p in self.nearfield_proto.parameters()), (
            "nearfield_proto must be frozen"
        )

    def train(self, mode: bool = True) -> "GeomConsistencyLoss":
        """Override: keep the frozen sub-modules in eval mode regardless."""
        super().train(mode)
        self.dav2_mlp.eval()
        self.nearfield_proto.eval()
        return self

    def forward(self, depth_dino_pred: Tensor, rgb_dino_pred: Tensor) -> Tensor:
        """Compute the geometric consistency loss.

        Parameters
        ----------
        depth_dino_pred : Tensor
            Predicted depth-DINO latents. Accepts ``[B, T, N, D]`` (the
            canonical training shape) or any leading-dim layout that ends
            in ``[..., N, D]`` with ``N`` patches.
        rgb_dino_pred : Tensor
            Predicted RGB-DINO latents with the same shape.

        Returns
        -------
        Tensor
            Scalar loss in ``[-1, 1]``.
        """
        assert depth_dino_pred.shape == rgb_dino_pred.shape, (
            f"shape mismatch: depth={tuple(depth_dino_pred.shape)} "
            f"rgb={tuple(rgb_dino_pred.shape)}"
        )
        assert depth_dino_pred.dim() >= 2, (
            "expected at least [N, D]; got "
            f"{tuple(depth_dino_pred.shape)}"
        )
        assert depth_dino_pred.shape[-1] == self.dim, (
            f"latent dim mismatch: expected {self.dim}, got {depth_dino_pred.shape[-1]}"
        )
        assert depth_dino_pred.shape[-2] >= 2, (
            "need at least 2 patch tokens for a rank correlation"
        )

        # Decode each modality to a per-patch scalar -> shape [..., N].
        d_scalar = self.dav2_mlp(depth_dino_pred)
        s_scalar = self.nearfield_proto(rgb_dino_pred)

        # Spearman is computed across the patch dimension (last) and any
        # leading batch / time dims are averaged.
        rho = soft_spearman(d_scalar, s_scalar, tau=self.tau)
        return -rho


__all__ = ["GeomConsistencyLoss", "soft_rank", "soft_spearman"]
