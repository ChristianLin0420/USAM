# SPDX-License-Identifier: MIT
"""Subtask-completion classifier head.

Given the current Conductor [EOS] embedding ``e_t`` plus a 16-frame
sliding window of ``[obs_cls, proprio]`` features, the classifier
predicts a logit for "the current subtask just completed". A positive
logit triggers a Plan-KV-Cache refresh (see
:func:`usam.conductor.drift.should_refresh`).

Training signal: AgiBot World 2026 ships ``instruction_segments`` —
boundaries between coarse subtasks like "approach", "grasp", "lift".
We label the boundary frame ``+1`` and all other frames ``-1``, then
train with BCE-with-logits (``L_subtask`` in §4.3 of the plan).

Architecture
------------
Two-layer MLP on a concatenation of:

* ``e_t``: ``[B, e_dim]`` — current Conductor embedding (or its
  projection from :class:`Conductor.e_proj_dim`).
* ``window_obs``: ``[B, window, obs_dim]`` — sliding window of
  RGB-DINO [CLS] tokens. Pooled by mean across the window axis.
* ``window_proprio``: ``[B, window, proprio_dim]`` — proprio history,
  also mean-pooled.

The pooling keeps the parameter count small and the latency negligible
even at 30 Hz; the temporal structure inside the window is mostly
redundant with the Conductor's own [EOS] anyway.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SubtaskCompletionHead(nn.Module):
    """Lightweight 2-layer MLP predicting ``P(subtask_completed)``.

    Parameters
    ----------
    e_dim : int
        Dim of ``e_t`` (e.g. 64 if you pass the Conductor's ``e_proj``).
    obs_dim : int
        Dim of one frame's RGB-DINO [CLS] (768 for ViT-B/14).
    proprio_dim : int
        Dim of one frame's proprio vector (50 in USAM-LeRobot).
    window : int
        Number of frames in the sliding window. Default 16.
    hidden : int
        Width of the MLP hidden layer. Default 128.
    """

    def __init__(
        self,
        e_dim: int = 64,
        obs_dim: int = 768,
        proprio_dim: int = 50,
        window: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        assert e_dim > 0, f"e_dim must be positive, got {e_dim}"
        assert obs_dim > 0, f"obs_dim must be positive, got {obs_dim}"
        assert proprio_dim > 0, f"proprio_dim must be positive, got {proprio_dim}"
        assert window > 0, f"window must be positive, got {window}"
        assert hidden > 0, f"hidden must be positive, got {hidden}"

        self.e_dim = int(e_dim)
        self.obs_dim = int(obs_dim)
        self.proprio_dim = int(proprio_dim)
        self.window = int(window)

        in_dim = e_dim + obs_dim + proprio_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def forward(
        self,
        e_t: Tensor,
        window_obs: Tensor,
        window_proprio: Tensor,
    ) -> Tensor:
        """Compute the subtask-completion logit.

        Parameters
        ----------
        e_t : Tensor
            ``[B, e_dim]`` (or ``[B, 1, e_dim]``) current embedding.
        window_obs : Tensor
            ``[B, window, obs_dim]`` RGB-DINO [CLS] history.
        window_proprio : Tensor
            ``[B, window, proprio_dim]`` proprio history.

        Returns
        -------
        Tensor
            ``[B, 1]`` logits. Positive ⇒ subtask complete.
        """
        if e_t.dim() == 3 and e_t.shape[1] == 1:
            e_t = e_t.squeeze(1)
        assert e_t.dim() == 2, f"e_t must be [B, D] or [B,1,D], got {tuple(e_t.shape)}"
        assert window_obs.dim() == 3, (
            f"window_obs must be [B, W, D_obs], got {tuple(window_obs.shape)}"
        )
        assert window_proprio.dim() == 3, (
            f"window_proprio must be [B, W, D_p], got {tuple(window_proprio.shape)}"
        )
        assert e_t.shape[-1] == self.e_dim
        assert window_obs.shape[-1] == self.obs_dim
        assert window_proprio.shape[-1] == self.proprio_dim
        assert window_obs.shape[1] == self.window, (
            f"window mismatch: cfg={self.window}, got={window_obs.shape[1]}"
        )
        assert window_proprio.shape[1] == self.window, (
            f"window mismatch: cfg={self.window}, got={window_proprio.shape[1]}"
        )
        assert window_obs.shape[0] == window_proprio.shape[0] == e_t.shape[0]

        obs_pool = window_obs.mean(dim=1)  # [B, obs_dim]
        prop_pool = window_proprio.mean(dim=1)  # [B, proprio_dim]
        h = torch.cat([e_t, obs_pool, prop_pool], dim=-1)
        h = self.act(self.fc1(h))
        return self.fc2(h)


__all__ = ["SubtaskCompletionHead"]
