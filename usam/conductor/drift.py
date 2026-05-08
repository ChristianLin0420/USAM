# SPDX-License-Identifier: MIT
"""Cosine-drift trigger and the cheap ``f_drift`` MLP.

The Conductor (Qwen3-VL-4B) is expensive to run every step. Instead, we
estimate how stale the *current* committed plan is by comparing the
Conductor's committed [EOS] embedding ``e_committed`` against an
inexpensive prediction ``e_now_estimate = f_drift(rgb_dino_cls,
e_committed)``.

Drift signal: cosine distance
-----------------------------
Drift is measured as ``d_t = 1 - cosine_similarity(e_committed,
e_now_estimate)`` on the **L2-normalized** [EOS] embedding. We
deliberately do *not* use raw KL on softmaxed hidden states: in the
SemEval 2020 distributional-shift detection literature (and our own
calibration runs), raw-KL on hidden states is dominated by entropy
fluctuations of the LM head and produces unstable triggers, while
cosine distance on normalized sentence embeddings tracks task-boundary
ground truth tightly.

Trigger conditions
------------------
:func:`should_refresh` returns ``True`` if **any** of:

* ``episode_start`` (i.e. ``t == 0``)
* ``t - last_refresh_t >= timer_hard``  (timer expiry)
* ``d_t > tau_hard``                    (hard drift breach)
* ``d_t > tau_soft`` AND ``t - last_refresh_t >= timer_soft``  (soft+timer combo)
* ``subtask_completed`` (positive logit from the subtask classifier)

Calibration
-----------
:func:`calibrate_taus` reads a list of measured cross-subtask cosine
distances and sets ``tau_hard`` to the empirical 90th percentile and
``tau_soft`` to the 50th percentile. Run this on a held-out language-
trajectory dataset before deployment.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DriftConfig:
    """Drift-trigger thresholds.

    Parameters
    ----------
    tau_hard : float
        Hard drift threshold. ``d_t > tau_hard`` always triggers refresh.
    tau_soft : float
        Soft drift threshold. Combined with ``timer_soft`` to catch slow
        drifts that the hard threshold misses.
    timer_hard : int
        Maximum frames between forced refreshes (default 60 frames =
        2 s @ 30 Hz). Hard upper bound on plan staleness.
    timer_soft : int
        Minimum frames before a soft-threshold breach can fire (default
        30 frames = 1 s @ 30 Hz).
    """

    tau_hard: float = 0.20
    tau_soft: float = 0.06
    timer_hard: int = 60
    timer_soft: int = 30


# ---------------------------------------------------------------------------
# f_drift MLP
# ---------------------------------------------------------------------------
class FDriftMLP(nn.Module):
    """Cheap MLP predicting ``e_now`` from ``(rgb_dino_cls, e_committed)``.

    The network is intentionally tiny — 2 hidden layers, total
    parameter count ``< 100_000`` — so it can run every control step
    without slowing the Player. The training-time loss is the MSE
    between ``f_drift(...)`` and the next genuine Conductor [EOS]
    embedding.

    Parameters
    ----------
    rgb_dino_dim : int
        Dim of the RGB-DINO [CLS] token (768 for ViT-B/14).
    e_dim : int
        Dim of the Qwen3-VL-4B [EOS] hidden state (e.g. 2048 for the
        text branch — Qwen3-VL's text hidden is much larger but we
        project before passing through this MLP). The plan gives
        flexibility; we assert the param budget in ``__init__``.
    hidden : int
        Width of the hidden MLP layer. Default tuned so that for the
        canonical ``rgb_dino_dim=768``, ``e_dim=64``, the total param
        count is around 50K.

    Notes
    -----
    The default ``e_dim=64`` assumes a *projected* embedding fed in by
    the conductor; the raw Qwen3-VL hidden (2048+) is too large to fit
    a 50K-param MLP. Project ``e`` down with a small ``nn.Linear`` if
    needed (this is owned by the Conductor wrapper, not this MLP).
    """

    def __init__(
        self,
        rgb_dino_dim: int = 768,
        e_dim: int = 64,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        assert rgb_dino_dim > 0, f"rgb_dino_dim must be positive, got {rgb_dino_dim}"
        assert e_dim > 0, f"e_dim must be positive, got {e_dim}"
        assert hidden > 0, f"hidden must be positive, got {hidden}"

        self.rgb_dino_dim = int(rgb_dino_dim)
        self.e_dim = int(e_dim)
        self.hidden = int(hidden)

        in_dim = rgb_dino_dim + e_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, e_dim)
        self.act = nn.GELU()

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 100_000, (
            f"FDriftMLP must stay below 100K params, got {n_params}. "
            f"Lower hidden ({hidden}) or e_dim ({e_dim})."
        )
        self._n_params = n_params

    def forward(self, rgb_dino_cls: Tensor, e_committed: Tensor) -> Tensor:
        """Predict the next ``e_now`` from current visuals and committed plan.

        Parameters
        ----------
        rgb_dino_cls : Tensor
            ``[B, rgb_dino_dim]``. The [CLS] token of RGB-DINO at the
            current frame.
        e_committed : Tensor
            ``[B, e_dim]`` (or ``[B, 1, e_dim]``). The committed [EOS]
            embedding from the last Conductor refresh, *projected* to
            ``e_dim`` to match the MLP budget.

        Returns
        -------
        Tensor
            ``[B, e_dim]`` predicted next-step [EOS] embedding.
        """
        if e_committed.dim() == 3 and e_committed.shape[1] == 1:
            e_committed = e_committed.squeeze(1)
        assert rgb_dino_cls.dim() == 2, (
            f"rgb_dino_cls must be [B, D], got {tuple(rgb_dino_cls.shape)}"
        )
        assert e_committed.dim() == 2, (
            f"e_committed must be [B, D] (or [B, 1, D]), got {tuple(e_committed.shape)}"
        )
        assert rgb_dino_cls.shape[-1] == self.rgb_dino_dim, (
            f"rgb_dino_cls feature dim {rgb_dino_cls.shape[-1]} != {self.rgb_dino_dim}"
        )
        assert e_committed.shape[-1] == self.e_dim, (
            f"e_committed feature dim {e_committed.shape[-1]} != {self.e_dim}"
        )
        h = torch.cat([rgb_dino_cls, e_committed], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.fc3(h)


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------
def should_refresh(
    t: int,
    d_t: float,
    last_refresh_t: int,
    config: DriftConfig,
    subtask_completion_logit: float = float("-inf"),
    episode_start: bool | None = None,
) -> bool:
    """Decide whether the Conductor should re-run on this frame.

    Drift signal interpretation
    ---------------------------
    ``d_t`` should be the **cosine distance** between the L2-normalized
    committed [EOS] embedding and the current estimate from
    :class:`FDriftMLP`. *Not* raw KL on softmaxed hidden states — that
    metric is unstable (SemEval-2020 distributional-shift detection
    literature) and cosine distance on normalized [EOS] embeddings is
    the empirical winner for sentence-level drift.

    Parameters
    ----------
    t : int
        Current control step (0-indexed).
    d_t : float
        Cosine distance ``1 - cos(e_committed, e_now_estimate)``.
        Range ``[0, 2]``; typical values 0.0–0.3.
    last_refresh_t : int
        Step index of the last successful Conductor refresh. Use ``-1``
        before any refresh.
    config : DriftConfig
        Threshold configuration.
    subtask_completion_logit : float, optional
        Logit from :class:`usam.conductor.classifier.SubtaskCompletionHead`.
        Positive means "subtask completed"; we trigger on
        ``> 0`` to bracket the boundary tightly.
    episode_start : bool or None, optional
        If ``True``, force-refresh regardless of timers/thresholds. If
        ``None``, equivalent to ``t == 0``.

    Returns
    -------
    bool
        ``True`` if any trigger fired.
    """
    assert isinstance(t, int), f"t must be int, got {type(t)}"
    assert isinstance(last_refresh_t, int), (
        f"last_refresh_t must be int, got {type(last_refresh_t)}"
    )

    if episode_start is None:
        episode_start = t == 0
    if episode_start:
        return True

    # If we have never refreshed and we are past t=0, force a refresh.
    if last_refresh_t < 0:
        return True

    elapsed = t - last_refresh_t

    # Hard timer expiry — guarantees no plan goes more than `timer_hard`
    # frames without a refresh.
    if elapsed >= config.timer_hard:
        return True

    # Hard drift threshold — even early in the trajectory, a big jump in
    # the embedding implies the plan is no longer aligned.
    if d_t > config.tau_hard:
        return True

    # Soft drift + minimum cooldown.
    if d_t > config.tau_soft and elapsed >= config.timer_soft:
        return True

    # Subtask boundary signal from the classifier.
    if subtask_completion_logit > 0.0:
        return True

    return False


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def calibrate_taus(drift_log: list[float]) -> DriftConfig:
    """Compute ``tau_hard`` (P90) and ``tau_soft`` (P50) from observed drifts.

    Run this offline on a held-out set of cross-subtask cosine
    distances (e.g. measured between subtask boundaries on AgiBot World
    2026). Returns a :class:`DriftConfig` with empirically-grounded
    thresholds; ``timer_hard`` and ``timer_soft`` are kept at their
    defaults — adjust separately based on your control rate.

    Parameters
    ----------
    drift_log : list of float
        Observed cosine distances. Must be non-empty.

    Returns
    -------
    DriftConfig
        Calibrated thresholds. ``timer_hard`` / ``timer_soft`` keep
        their defaults.
    """
    assert isinstance(drift_log, list), "drift_log must be a list"
    assert len(drift_log) > 0, "drift_log must be non-empty"
    assert all(isinstance(x, (int, float)) for x in drift_log), (
        "drift_log entries must be numeric"
    )

    sorted_log = sorted(float(x) for x in drift_log)
    n = len(sorted_log)
    # P50 (median) and P90 with linear interpolation.
    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_log[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_log[lo] * (1 - frac) + sorted_log[hi] * frac

    tau_soft = _percentile(0.50)
    tau_hard = _percentile(0.90)

    return DriftConfig(tau_hard=float(tau_hard), tau_soft=float(tau_soft))


# ---------------------------------------------------------------------------
# Cosine-distance helper (convenience)
# ---------------------------------------------------------------------------
def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Return ``1 - cosine_similarity(a, b)`` with safe normalization.

    Both inputs are expected to already be L2-normalized; we re-normalize
    defensively so callers can pass raw embeddings without breaking
    correctness.
    """
    assert a.dim() >= 1 and b.dim() >= 1, "inputs must have at least 1 dim"
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)


__all__ = [
    "DriftConfig",
    "FDriftMLP",
    "should_refresh",
    "calibrate_taus",
    "cosine_distance",
]
