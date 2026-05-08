# SPDX-License-Identifier: MIT
"""Forward-dynamics consistency loss between action and optical-flow latents.

The :class:`FlowActionConsistencyLoss` enforces that the **action chunk**
and the **predicted flow-DINO latents** agree on how much the world should
move. Concretely:

* A learned 2-layer MLP ``g_phi(proprio, action_chunk) -> scalar`` predicts
  the average flow magnitude that the action chunk should produce given the
  current proprioceptive state.
* A fixed (non-trainable, but differentiable) decode head reads the
  flow-DINO patch tokens and returns the mean magnitude of the V channel
  of an HSV-style flow encoding. This is the empirical target.
* Loss = MSE between the prediction and the empirical magnitude.

The fixed decode head is implemented as a per-channel mixing buffer that is
*differentiable through* but holds frozen weights — so gradients flow back
into ``flow_dino_pred`` while the head itself is not trained. The buffer is
seeded deterministically (with a fixed RNG) so the head behaves identically
across processes / workers without needing a checkpoint file.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Fixed flow-magnitude decode head
# ---------------------------------------------------------------------------
def _make_fixed_decode_weight(dim: int, seed: int = 0xF10) -> Tensor:
    """Deterministic ``[dim]`` mixing weight used by the fixed flow decoder.

    The head computes ``relu(<x, w>)`` per patch token, then averages over
    patches. ``w`` is sampled with a fixed RNG so the result is identical
    across hosts / processes without a checkpoint file. Returned tensor
    has unit L2-norm so the magnitude is on a comparable scale across
    different ``dim``.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, generator=g)
    return w / w.norm().clamp_min(1e-8)


def flow_magnitude(flow_dino_pred: Tensor, decode_weight: Tensor) -> Tensor:
    """Return mean per-sample flow magnitude using a fixed decode head.

    Parameters
    ----------
    flow_dino_pred : Tensor
        Flow-DINO predictions of shape ``[B, ..., N, D]``. The last two
        dims are patches and latent. Any number of leading dims is fine.
    decode_weight : Tensor
        ``[D]`` mixing weights produced by :func:`_make_fixed_decode_weight`.

    Returns
    -------
    Tensor
        ``[B, ...]`` flow magnitudes (the leading dims of the input minus
        the patch and latent dims).

    Notes
    -----
    The "decode the patch tokens to mean HSV-V magnitude" specification in
    plan §11.8 is implemented here as a deterministic non-negative scalar
    per patch followed by a mean over the patch dimension. We square the
    inner product so the result is everywhere non-negative *and* smooth
    (avoiding the ReLU kink at zero, which would make
    :func:`torch.autograd.gradcheck` flaky). The MSE objective normalises
    the scale across batches naturally.
    """
    assert flow_dino_pred.dim() >= 2, "expected [..., N, D]"
    assert decode_weight.dim() == 1, "decode_weight must be [D]"
    assert flow_dino_pred.shape[-1] == decode_weight.shape[0], (
        f"latent dim mismatch: pred {flow_dino_pred.shape[-1]} vs decode {decode_weight.shape[0]}"
    )
    # [..., N, D] @ [D] -> [..., N]
    per_patch = (flow_dino_pred @ decode_weight).pow(2)
    # Mean over the patch dim -> [...]
    return per_patch.mean(dim=-1)


# ---------------------------------------------------------------------------
# g_phi MLP
# ---------------------------------------------------------------------------
class _GPhiMLP(nn.Module):
    """2-layer MLP predicting a scalar flow magnitude from
    ``(proprio, flat_action_chunk)``.
    """

    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Public loss
# ---------------------------------------------------------------------------
class FlowActionConsistencyLoss(nn.Module):
    """Forward-action consistency loss ``L_flow-act``.

    Implements §11.8 of ``docs/IMPLEMENTATION_PLAN.md``.

    Parameters
    ----------
    proprio_dim : int
        Width of the proprioceptive state vector. Default 50 to match
        USAM-LeRobot v2.1.
    action_chunk_dim : int
        Flat dimensionality of the action chunk (``D_action * chunk_len``).
        Default ``7 * 16`` matching the canonical EE chunk.
    hidden : int
        MLP hidden width. Default 256.
    flow_dim : int
        Latent dimensionality ``D`` of the flow-DINO predictions. Used to
        size the fixed decode head.
    decode_seed : int
        RNG seed for the fixed decode head. Identical seeds across
        processes give identical decoded targets.

    Notes
    -----
    * ``g_phi`` is trainable.
    * The flow decode head is held as a registered buffer
      (``self.decode_weight``). It is not updated by the optimiser, but
      gradients propagate through it back into ``flow_dino_pred``.
    * Pass ``flow_dino_pred.detach()`` to :meth:`forward` if a caller wants
      to block that gradient — by default we don't, because the consistency
      objective should also pull the flow-DINO head toward the action's
      predicted magnitude.
    """

    def __init__(
        self,
        proprio_dim: int = 50,
        action_chunk_dim: int = 7 * 16,
        hidden: int = 256,
        flow_dim: int = 768,
        decode_seed: int = 0xF10,
    ) -> None:
        super().__init__()
        assert proprio_dim > 0, "proprio_dim must be positive"
        assert action_chunk_dim > 0, "action_chunk_dim must be positive"
        assert hidden > 0, "hidden must be positive"
        assert flow_dim > 0, "flow_dim must be positive"

        self.proprio_dim = proprio_dim
        self.action_chunk_dim = action_chunk_dim
        self.flow_dim = flow_dim

        self.g_phi = _GPhiMLP(in_dim=proprio_dim + action_chunk_dim, hidden=hidden)

        decode_weight = _make_fixed_decode_weight(flow_dim, seed=decode_seed)
        self.register_buffer("decode_weight", decode_weight, persistent=True)

    def forward(
        self,
        proprio: Tensor,
        action_chunk: Tensor,
        flow_dino_pred: Tensor,
    ) -> Tensor:
        """Compute the forward-action consistency loss.

        Parameters
        ----------
        proprio : Tensor
            ``[B, proprio_dim]`` proprioceptive state vectors.
        action_chunk : Tensor
            ``[B, chunk_len, action_dim]`` or ``[B, action_chunk_dim]``
            action chunks. The tensor is flattened along all dims after
            the batch.
        flow_dino_pred : Tensor
            Flow-DINO predictions of shape ``[B, ..., N, D]``. Leading
            extras (e.g. a time dim) are averaged into a per-batch
            scalar before MSE.

        Returns
        -------
        Tensor
            Scalar MSE loss.
        """
        assert proprio.dim() == 2, f"proprio must be [B, D_state], got {tuple(proprio.shape)}"
        assert proprio.shape[-1] == self.proprio_dim, (
            f"proprio_dim mismatch: expected {self.proprio_dim}, got {proprio.shape[-1]}"
        )
        assert flow_dino_pred.dim() >= 3, (
            f"flow_dino_pred must be [B, ..., N, D], got {tuple(flow_dino_pred.shape)}"
        )
        assert flow_dino_pred.shape[0] == proprio.shape[0], (
            "batch mismatch between proprio and flow_dino_pred"
        )

        b = proprio.shape[0]
        action_flat = action_chunk.reshape(b, -1)
        assert action_flat.shape[-1] == self.action_chunk_dim, (
            f"action_chunk_dim mismatch: expected {self.action_chunk_dim}, "
            f"got {action_flat.shape[-1]}"
        )

        # g_phi prediction.
        x = torch.cat([proprio, action_flat], dim=-1)
        pred = self.g_phi(x)  # [B]

        # Empirical magnitude from flow predictions.
        target = flow_magnitude(flow_dino_pred, self.decode_weight)  # [B, ...]
        # Reduce any extra leading dims (e.g. time) to per-sample scalar.
        if target.dim() > 1:
            target = target.flatten(start_dim=1).mean(dim=-1)
        assert pred.shape == target.shape, (
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )

        return torch.nn.functional.mse_loss(pred, target)


__all__ = ["FlowActionConsistencyLoss", "flow_magnitude"]
