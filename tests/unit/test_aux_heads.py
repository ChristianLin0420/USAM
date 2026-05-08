# SPDX-License-Identifier: MIT
"""Unit tests for the USAM auxiliary heads.

We exercise both heads at a very small ``D`` so the numeric gradient checks
in :func:`torch.autograd.gradcheck` finish in seconds:

* ``GeomConsistencyLoss`` is constructed with ``dim=64``; the depth /
  near-field heads are randomly initialised (the test instantiates
  with ``dav2_distill_ckpt=None`` and silences the resulting warning).
* ``FlowActionConsistencyLoss`` uses ``flow_dim=64`` to match.

The test batch / time / patch / latent shape ``[B=2, T=3, N=8, D=64]`` is
the fixture mandated by the agent prompt.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from usam.aux_heads import (
    FlowActionConsistencyLoss,
    GeomConsistencyLoss,
    soft_rank,
    soft_spearman,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_geom_loss(dim: int = 64, hidden: int = 16, tau: float = 1.0) -> GeomConsistencyLoss:
    """Build a randomly-initialised ``GeomConsistencyLoss`` for testing."""
    with warnings.catch_warnings():
        # The constructor explicitly warns when ckpt paths are None — that's
        # the expected unit-test path.
        warnings.simplefilter("ignore")
        return GeomConsistencyLoss(dim=dim, hidden=hidden, tau=tau)


def _make_flow_act_loss(
    proprio_dim: int = 4,
    chunk_dim: int = 6,
    hidden: int = 16,
    flow_dim: int = 64,
) -> FlowActionConsistencyLoss:
    return FlowActionConsistencyLoss(
        proprio_dim=proprio_dim,
        action_chunk_dim=chunk_dim,
        hidden=hidden,
        flow_dim=flow_dim,
    )


# ---------------------------------------------------------------------------
# Shape / scalar sanity
# ---------------------------------------------------------------------------
def test_geom_loss_returns_scalar() -> None:
    loss_fn = _make_geom_loss()
    depth = torch.randn(2, 3, 8, 64)
    rgb = torch.randn(2, 3, 8, 64)
    out = loss_fn(depth, rgb)
    assert out.dim() == 0, f"expected scalar, got {tuple(out.shape)}"
    assert torch.isfinite(out)


def test_flow_act_loss_returns_scalar() -> None:
    loss_fn = _make_flow_act_loss()
    proprio = torch.randn(2, 4)
    action_chunk = torch.randn(2, 6)
    flow_pred = torch.randn(2, 3, 8, 64)
    out = loss_fn(proprio, action_chunk, flow_pred)
    assert out.dim() == 0, f"expected scalar, got {tuple(out.shape)}"
    assert torch.isfinite(out)
    assert (out >= 0).item(), "MSE loss must be non-negative"


# ---------------------------------------------------------------------------
# Soft-rank primitive sanity
# ---------------------------------------------------------------------------
def test_soft_rank_monotone() -> None:
    """For widely-spaced inputs, soft rank converges to the true rank.

    The implementation's formula r_i = 1 + Σ σ((x_j - x_i)/τ) counts elements
    *larger* than x_i, so rank 1 = largest. Spearman = Pearson(ranks) is
    invariant under global rank reversal, so the loss math is unaffected; we
    just assert the convention here.
    """
    x = torch.tensor([0.0, 10.0, 5.0, 20.0])
    r = soft_rank(x, tau=0.1)
    # Largest (20.0) -> rank 1, then 10.0 -> 2, 5.0 -> 3, 0.0 -> 4.
    rounded = r.round().long().tolist()
    assert rounded == [4, 2, 3, 1]


def test_soft_spearman_perfect_agreement() -> None:
    a = torch.linspace(0.0, 1.0, 16)
    rho = soft_spearman(a, a, tau=0.1)
    assert rho.item() > 0.99


def test_soft_spearman_perfect_disagreement() -> None:
    a = torch.linspace(0.0, 1.0, 16)
    rho = soft_spearman(a, a.flip(0), tau=0.1)
    assert rho.item() < -0.99


# ---------------------------------------------------------------------------
# Frozen-component assertion (charter requirement)
# ---------------------------------------------------------------------------
def test_geom_loss_frozen_components() -> None:
    loss_fn = _make_geom_loss()
    for p in loss_fn.dav2_mlp.parameters():
        assert not p.requires_grad, (
            "GeomConsistencyLoss.dav2_mlp parameter is unexpectedly trainable"
        )
    for p in loss_fn.nearfield_proto.parameters():
        assert not p.requires_grad, (
            "GeomConsistencyLoss.nearfield_proto parameter is unexpectedly trainable"
        )


def test_geom_loss_train_keeps_subs_in_eval() -> None:
    """Calling .train() on the parent must not flip the frozen children."""
    loss_fn = _make_geom_loss()
    loss_fn.train(True)
    assert not loss_fn.dav2_mlp.training
    assert not loss_fn.nearfield_proto.training


def test_geom_loss_no_grad_into_frozen_params() -> None:
    """Gradients into the frozen heads' parameters must be ``None``."""
    loss_fn = _make_geom_loss()
    depth = torch.randn(2, 8, 64, requires_grad=True)
    rgb = torch.randn(2, 8, 64, requires_grad=True)
    out = loss_fn(depth, rgb)
    out.backward()
    for p in loss_fn.dav2_mlp.parameters():
        assert p.grad is None, "frozen dav2_mlp received a gradient"
    for p in loss_fn.nearfield_proto.parameters():
        assert p.grad is None, "frozen nearfield_proto received a gradient"


# ---------------------------------------------------------------------------
# gradcheck (CPU, fp64, tiny input)
# ---------------------------------------------------------------------------
def test_geom_loss_gradcheck() -> None:
    torch.manual_seed(0)
    loss_fn = _make_geom_loss(dim=64, hidden=16, tau=1.0).double()

    depth = torch.randn(2, 3, 8, 64, dtype=torch.float64, requires_grad=True)
    rgb = torch.randn(2, 3, 8, 64, dtype=torch.float64, requires_grad=True)

    def fn(d: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return loss_fn(d, r)

    assert torch.autograd.gradcheck(fn, (depth, rgb), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_flow_act_loss_gradcheck() -> None:
    torch.manual_seed(0)
    loss_fn = _make_flow_act_loss(
        proprio_dim=4, chunk_dim=6, hidden=16, flow_dim=64
    ).double()

    proprio = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    action = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    flow = torch.randn(2, 3, 8, 64, dtype=torch.float64, requires_grad=True)

    def fn(
        p: torch.Tensor, a: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        return loss_fn(p, a, f)

    assert torch.autograd.gradcheck(fn, (proprio, action, flow), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Consistent vs inconsistent sanity (smoke)
# ---------------------------------------------------------------------------
def test_geom_loss_consistent_lower_than_inconsistent() -> None:
    """A pair where depth and RGB scalars share rank order must give a
    lower loss than a pair where the ordering is reversed.

    Construction: pre-compute the two heads' per-patch scalars on a random
    bank of latents, then re-order the RGB latents so their scalar ranks
    align with the depth scalar's ranks. The "inconsistent" pair flips the
    ordering of the RGB latents to invert the rank correlation.
    """
    torch.manual_seed(123)
    loss_fn = _make_geom_loss(dim=64, hidden=16, tau=0.5)
    loss_fn.eval()

    n, d = 12, 64
    depth = torch.randn(1, n, d)
    rgb_bank = torch.randn(1, n, d)

    with torch.no_grad():
        depth_scalar = loss_fn.dav2_mlp(depth)[0]   # [N]
        rgb_scalar = loss_fn.nearfield_proto(rgb_bank)[0]  # [N]

    depth_order = torch.argsort(depth_scalar)
    rgb_order = torch.argsort(rgb_scalar)

    # Permute the RGB bank so its rank order matches depth's exactly.
    perm_consistent = torch.empty_like(depth_order)
    perm_consistent[depth_order] = rgb_order
    rgb_consistent = rgb_bank[:, perm_consistent]

    # Inconsistent: reverse the rank order.
    perm_inconsistent = torch.empty_like(depth_order)
    perm_inconsistent[depth_order.flip(0)] = rgb_order
    rgb_inconsistent = rgb_bank[:, perm_inconsistent]

    loss_consistent = loss_fn(depth, rgb_consistent).item()
    loss_inconsistent = loss_fn(depth, rgb_inconsistent).item()
    assert loss_consistent < loss_inconsistent, (
        f"expected consistent < inconsistent, got {loss_consistent} vs {loss_inconsistent}"
    )


def test_flow_act_loss_consistent_lower_than_inconsistent() -> None:
    """Loss should be lower when ``g_phi`` already predicts something close
    to the empirical ``flow_magnitude`` than when the two disagree.

    The fixed flow decoder is ``mean_n((flow_n @ decode_w)^2)``. If we set
    ``flow = α * decode_w`` (broadcast across patches and time), then
    ``decode_w`` is unit-norm so ``flow @ decode_w == α`` per patch and the
    decoded magnitude is ``α^2``. We pick ``α`` such that ``α^2`` lands at
    ``g_phi``'s bias output (the "consistent" target) and far above it
    (the "inconsistent" target).
    """
    torch.manual_seed(0)
    loss_fn = _make_flow_act_loss(
        proprio_dim=4, chunk_dim=6, hidden=16, flow_dim=64
    )
    loss_fn.eval()

    proprio = torch.zeros(2, 4)
    action = torch.zeros(2, 6)
    with torch.no_grad():
        bias_pred = loss_fn.g_phi(torch.cat([proprio, action], dim=-1))  # [B]
    # We want α^2 == bias_pred, so α = sqrt(|bias_pred|) carrying the sign
    # of bias_pred via post-hoc sign of α (square removes it). Clamp away
    # from 0 so the inconsistent case can move to a much larger magnitude.
    target_mag = bias_pred.detach().clamp_min(1e-3)  # [B]
    alpha = target_mag.sqrt().reshape(2, 1, 1, 1)

    decode_w = loss_fn.decode_weight  # [D]
    flow_consistent = alpha * decode_w.reshape(1, 1, 1, -1).expand(2, 3, 8, -1).contiguous()
    flow_inconsistent = (alpha + 50.0) * decode_w.reshape(1, 1, 1, -1).expand(2, 3, 8, -1).contiguous()

    loss_consistent = loss_fn(proprio, action, flow_consistent).item()
    loss_inconsistent = loss_fn(proprio, action, flow_inconsistent).item()
    assert loss_consistent < loss_inconsistent, (
        f"expected consistent < inconsistent, got {loss_consistent} vs {loss_inconsistent}"
    )


def test_geom_loss_warns_when_ckpts_missing() -> None:
    """Documented: the constructor warns when no checkpoints are passed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GeomConsistencyLoss(dim=8, hidden=4)
    msgs = [str(w.message) for w in caught]
    assert any("dav2_distill_ckpt" in m for m in msgs)
    assert any("nearfield_proto_ckpt" in m for m in msgs)


@pytest.mark.parametrize("bad_shape", [(2, 8, 32), (2, 8, 128)])
def test_geom_loss_dim_mismatch_raises(bad_shape: tuple[int, ...]) -> None:
    loss_fn = _make_geom_loss(dim=64)
    bad = torch.randn(*bad_shape)
    rgb = torch.randn(*bad_shape)
    with pytest.raises(AssertionError):
        loss_fn(bad, rgb)
