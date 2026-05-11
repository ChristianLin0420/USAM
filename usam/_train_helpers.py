# SPDX-License-Identifier: MIT
"""USAM training helpers: small Player stub + checkpointing utilities.

This module groups the **non-LDA** plumbing that ``usam.train`` needs but
that doesn't belong in any of the locked Wave-1+2 modules:

* :class:`SmokePlayer` — a tiny stand-in for the LDA-1B MM-DiT Player. It
  consumes the same input dict the real Player consumes (rgb/depth
  DINO features, proprio, action chunk, plan-cache) and emits the three
  velocity heads (``action``, ``image``, ``depth``) plus the auxiliary
  head inputs the unified loss expects. Used by the
  smoke-train integration test and the ``usam_350m_smoke`` config; the
  H200 burst replaces this with the real LDA-1B Player.
* :class:`USAMTrainModel` — wraps Tri-DINO + Conductor + SmokePlayer +
  drift / subtask heads + USAMUnifiedLoss into a single ``nn.Module``.
  The training loop calls ``model.training_step(batch, weights)`` once
  per step and gets back ``(total_loss, per_loss_dict)``.
* :class:`CheckpointManager` — keep last 3 checkpoints + best by val
  loss. Tags each checkpoint with the git SHA + a USAM run-id.
* :func:`detect_precision` — H200 capability sniffing for
  Transformer-Engine FP8 vs BF16 fallback.
* :func:`compute_ramped_weights` — the linear 0 → target ramp used for
  ``geom`` over the first 50_000 steps.

The Player stub is **deliberately small** (~5 M params with the smoke
config). It is *not* the production model — it is the minimum surface
area needed to exercise the training plumbing.
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from usam.conductor import (
    Conductor,
    FDriftMLP,
    MockConductorBackbone,
    PlanCache,
    SubtaskCompletionHead,
    apply_cache_dropout,
)
from usam.losses import LossWeights, USAMUnifiedLoss

__all__ = [
    "SmokePlayer",
    "USAMTrainModel",
    "CheckpointManager",
    "RunMetadata",
    "detect_precision",
    "compute_ramped_weights",
    "build_run_id",
    "git_sha",
    "load_checkpoint",
]


# ---------------------------------------------------------------------------
# Public helpers for inference-time checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path: "Path | str") -> Dict[str, Any]:
    """Load a USAM training checkpoint into a plain dict.

    Counterpart to :meth:`CheckpointManager._save`. The on-disk format is
    described in :class:`CheckpointManager`'s docstring; we just call
    ``torch.load(map_location="cpu")`` and surface the dict.

    Parameters
    ----------
    path : Path | str
        Absolute path to a ``.pt`` file produced by
        :class:`CheckpointManager`.

    Returns
    -------
    dict
        Keys: ``state_dict``, ``optimizer``, ``scheduler``, ``step``,
        ``run`` (a serialized :class:`RunMetadata`),
        ``best_val_loss``, ``val_loss``, ``timestamp``.

    Notes
    -----
    The inference path consumes ``state_dict`` and ``run["git_sha"]``;
    the rest is for resume / debugging.
    """
    p = Path(path)
    assert p.exists(), f"checkpoint does not exist: {p}"
    payload = torch.load(str(p), map_location="cpu")
    assert isinstance(payload, dict), (
        f"expected dict checkpoint, got {type(payload).__name__}"
    )
    return payload


# ---------------------------------------------------------------------------
# Run / precision metadata
# ---------------------------------------------------------------------------
@dataclass
class RunMetadata:
    """Metadata stamped onto every checkpoint.

    Parameters
    ----------
    run_id : str
        ``<timestamp>-<short-uuid>``.
    git_sha : str
        ``git rev-parse --short HEAD`` at training start; ``"unknown"``
        if the repo is dirty / not a git checkout.
    config_path : str
        YAML path used for this run.
    started_at : float
        ``time.time()`` at run start.
    """

    run_id: str
    git_sha: str
    config_path: str
    started_at: float


def git_sha(repo_root: Optional[Path] = None) -> str:
    """Return ``git rev-parse --short HEAD`` or ``"unknown"`` on failure."""
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def build_run_id() -> str:
    """``<YYYYmmdd-HHMMSS>-<short-uuid>``."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    short = uuid.uuid4().hex[:8]
    return f"{ts}-{short}"


@dataclass
class PrecisionPlan:
    """Result of :func:`detect_precision`.

    Parameters
    ----------
    weights_dtype : torch.dtype
        Weight storage dtype. ``torch.bfloat16`` when CUDA is available,
        ``torch.float32`` on CPU.
    use_te_fp8 : bool
        Whether Transformer-Engine FP8 should be enabled. Only ``True``
        on Hopper-class GPUs (capability ``(9, 0)``).
    device_type : str
        ``"cuda"`` / ``"cpu"``.
    note : str
        Human-readable explanation logged at startup.
    """

    weights_dtype: torch.dtype
    use_te_fp8: bool
    device_type: str
    note: str


def detect_precision(force_cpu: bool = False) -> PrecisionPlan:
    """Pick weights / activation precision for the current device.

    Hierarchy (matches plan §6.2):

    * H200 (CUDA capability ``(9, 0)``)  → BF16 weights + TE FP8 activations.
    * Other CUDA (A40 / A100)             → BF16 weights, no FP8.
    * CPU / no CUDA                       → FP32 weights, no FP8.

    ``transformer_engine`` is imported lazily — its absence on the
    ``qwen3vl`` env must not break CPU plumbing.
    """
    if force_cpu or not torch.cuda.is_available():
        return PrecisionPlan(
            weights_dtype=torch.float32,
            use_te_fp8=False,
            device_type="cpu",
            note="CPU fallback (force_cpu or CUDA unavailable)",
        )

    cap = torch.cuda.get_device_capability()
    is_h200 = cap == (9, 0)

    # FP8 also needs transformer-engine; if we're on H200 but TE isn't
    # installed, fall back to BF16 with a clear note.
    use_te = False
    note = f"GPU cap={cap} BF16 weights"
    if is_h200:
        try:
            import transformer_engine.pytorch as _te  # noqa: F401  pyright: ignore

            use_te = True
            note = f"H200 cap={cap} BF16 weights + TE FP8 activations"
        except (ImportError, ModuleNotFoundError):
            note = f"H200 cap={cap} but transformer_engine not installed; BF16 only"

    return PrecisionPlan(
        weights_dtype=torch.bfloat16,
        use_te_fp8=use_te,
        device_type="cuda",
        note=note,
    )


# ---------------------------------------------------------------------------
# Ramp schedule
# ---------------------------------------------------------------------------
def compute_ramped_weights(
    base: LossWeights,
    step: int,
    geom_target: float,
    ramp_steps: int = 50_000,
) -> LossWeights:
    """Linearly ramp ``geom`` weight from 0 → target.

    Schedule (per the plan §4.3):

    * ``step <= 0``                        → ``geom = 0.0``
    * ``0 < step < ramp_steps``           → ``geom = target * step / ramp_steps``
    * ``step >= ramp_steps``              → ``geom = target`` (clamped)

    The other five weights (``action``, ``rgb``, ``depth``, ``drift``,
    ``subtask``) come from ``base`` unchanged.

    Parameters
    ----------
    base : LossWeights
        The fixed weights from the YAML config. Its ``geom`` entry is
        *replaced*; everything else is copied.
    step : int
        Current global step counter.
    geom_target : float
        Final ``geom`` weight after the ramp.
    ramp_steps : int, optional
        Length of the ramp. Default ``50_000``.
    """
    assert isinstance(base, LossWeights)
    assert isinstance(step, int)
    assert ramp_steps > 0, "ramp_steps must be positive"
    assert geom_target >= 0.0

    if step <= 0:
        geom = 0.0
    elif step >= ramp_steps:
        geom = geom_target
    else:
        frac = float(step) / float(ramp_steps)
        geom = geom_target * frac

    return LossWeights(
        action=base.action,
        rgb=base.rgb,
        depth=base.depth,
        geom=geom,
        drift=base.drift,
        subtask=base.subtask,
    )


# ---------------------------------------------------------------------------
# Smoke Player — minimal stand-in for the real LDA-1B MM-DiT
# ---------------------------------------------------------------------------
class SmokePlayer(nn.Module):
    """A tiny Player that emits the four MM-DiT velocity predictions.

    Architecture (parameter-thin on purpose):

    * One linear stem per modality (rgb / depth) reads
      ``[B, T, N, D]`` features and projects to ``hidden_size``.
    * A ``num_layers``-deep Transformer encoder (no cross-attention)
      reads the concatenated tokens; on every layer we also pull the
      cached plan ``K`` / ``V`` from :class:`PlanCache` and add them to
      the running residual stream (cheap cross-conditioning).
    * Three projection heads emit the three velocity tensors with shapes
      that match the ground-truth tensors the dataloader produces.

    The plan ``K`` / ``V`` cache is read-only here — we never write back
    to it. The training loop is responsible for refreshing the cache via
    the Conductor before each step.

    The model is **not** the production Player. It exists so the unified
    loss can produce real gradients on real tensors during smoke
    training. Its weight count is ~5 M with the smoke config, well
    within the 8×A40 budget at bs=4.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        rgb_dim: int = 768,
        depth_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        action_dim: int = 7,
        action_chunk: int = 8,
        n_keep_tokens: int = 65,
        proprio_dim: int = 50,
    ) -> None:
        super().__init__()
        assert hidden_size > 0
        assert num_layers > 0
        assert num_heads > 0
        assert action_dim > 0
        assert action_chunk > 0
        assert n_keep_tokens > 0
        assert proprio_dim > 0

        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.n_keep_tokens = n_keep_tokens
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        self.num_layers = num_layers

        # Per-modality stems.
        self.rgb_stem = nn.Linear(rgb_dim, hidden_size)
        self.depth_stem = nn.Linear(depth_dim, hidden_size)

        # Proprio modulation token.
        self.proprio_proj = nn.Linear(proprio_dim, hidden_size)

        # Action chunk embedding (noisy input at training time).
        self.action_in = nn.Linear(action_dim, hidden_size)

        # Trunk.
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Heads (shapes match what the dataloader provides as targets).
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.rgb_head = nn.Linear(hidden_size, rgb_dim)
        self.depth_head = nn.Linear(hidden_size, depth_dim)

    # ------------------------------------------------------------------
    # Plan-cache K/V projections (consumed by ``Conductor.refresh``)
    # ------------------------------------------------------------------
    def make_plan_kv_projs(self) -> Tuple[List[nn.Linear], List[nn.Linear]]:
        """Build the per-layer K / V projections used by ``PlanCache.refresh``.

        We keep these on the player (rather than inside Conductor) so
        :class:`PlanCache` can call them without being aware of the
        Player's internal hidden dim. One pair per Transformer layer.

        Returns
        -------
        (k_projs, v_projs) : tuple of two lists of nn.Linear
            Each ``num_layers`` long; both project
            ``[..., d_model] -> [..., hidden_size]``.
        """
        # Lazily allocated so the user can call this once and re-use.
        # nn.Module.__setattr__ already registers ModuleList children — no
        # need to call ``add_module`` again.
        if not hasattr(self, "plan_k_projs"):
            self.plan_k_projs = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)]
            )
            self.plan_v_projs = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)]
            )
        return list(self.plan_k_projs), list(self.plan_v_projs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb_dino_seq: Tensor,
        depth_dino_seq: Tensor,
        proprio: Tensor,
        action_noisy: Tensor,
        plan_cache: PlanCache,
    ) -> Dict[str, Tensor]:
        """Run the smoke Player forward.

        Parameters
        ----------
        rgb_dino_seq, depth_dino_seq : Tensor
            ``[B, T_total, N, D_<mod>]`` cached DINO features. ``T_total``
            covers history + future; we just average across it for the
            stub.
        proprio : Tensor
            ``[B, proprio_dim]``.
        action_noisy : Tensor
            ``[B, action_chunk, action_dim]`` noisy input action.
        plan_cache : PlanCache
            Refreshed cache. Layer 0 KV is added to every Player token
            as a cheap cross-conditioning shortcut. Real cross-attention
            is the production Player's job.

        Returns
        -------
        dict of Tensor
            ``predictions`` dict keyed by ``image``, ``action``, ``depth``,
            ``geom`` (a sub-dict).
        """
        assert rgb_dino_seq.dim() == 4, f"rgb_dino_seq must be [B,T,N,D], got {tuple(rgb_dino_seq.shape)}"
        assert depth_dino_seq.dim() == 4
        assert proprio.dim() == 2
        assert action_noisy.dim() == 3

        b = rgb_dino_seq.shape[0]
        # Average across the time + token dims for the stem (cheap pooling).
        rgb_pool = rgb_dino_seq.mean(dim=(1, 2))  # [B, D_rgb]
        depth_pool = depth_dino_seq.mean(dim=(1, 2))

        # Stems → [B, hidden]
        rgb_tok = self.rgb_stem(rgb_pool).unsqueeze(1)  # [B, 1, H]
        depth_tok = self.depth_stem(depth_pool).unsqueeze(1)
        proprio_tok = self.proprio_proj(proprio).unsqueeze(1)
        action_tok = self.action_in(action_noisy)  # [B, chunk, H]

        x = torch.cat([rgb_tok, depth_tok, proprio_tok, action_tok], dim=1)
        # [B, S, H] with S = 3 + chunk

        # Add layer-0 plan K/V mean as cheap cross-conditioning.
        if plan_cache.is_valid():
            k0, v0 = plan_cache.get(0, branch="image")
            # Mean across the plan-tokens dim, broadcast over our seq.
            cond = (k0.float() + v0.float()).mean(dim=1, keepdim=True)  # [B, 1, H]
            x = x + cond.to(x.dtype)

        x = self.trunk(x)

        # Slice tokens back out for the heads.
        rgb_h = x[:, 0]  # [B, H]
        depth_h = x[:, 1]
        action_h = x[:, 3 : 3 + self.action_chunk]  # [B, chunk, H]

        # Heads — produce shapes that match the targets the dataloader emits.
        # rgb / depth targets are [B, T_total, N, D]; we tile back.
        T_total = rgb_dino_seq.shape[1]
        N = rgb_dino_seq.shape[2]
        rgb_pred = self.rgb_head(rgb_h).unsqueeze(1).unsqueeze(1).expand(b, T_total, N, -1)
        depth_pred = self.depth_head(depth_h).unsqueeze(1).unsqueeze(1).expand(b, T_total, N, -1)
        action_pred = self.action_head(action_h)  # [B, chunk, action_dim]

        return {
            "action": action_pred,
            "image": rgb_pred,
            "depth": depth_pred,
            "geom": {"depth_dino_pred": depth_pred, "rgb_dino_pred": rgb_pred},
        }


# ---------------------------------------------------------------------------
# USAM training model (composes Conductor + Player + drift + subtask + losses)
# ---------------------------------------------------------------------------
@dataclass
class _USAMTrainConfig:
    """Compact knobs the training loop reads off the YAML.

    Keeping this *outside* :class:`USAMTrainModel` lets the YAML and the
    constructor keyword args stay in one well-typed place. The training
    entry point in :mod:`usam.train` translates the OmegaConf config
    into one of these.
    """

    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    action_dim: int = 7
    action_chunk: int = 8
    rgb_dim: int = 768
    depth_dim: int = 768
    proprio_dim: int = 50
    n_keep_tokens: int = 65
    n_plan_tokens: int = 32
    e_proj_dim: int = 64
    backbone_hidden: int = 64
    backbone_seq_len: int = 64
    cache_dropout_p: float = 0.5
    cache_dropout_window: int = 60


class USAMTrainModel(nn.Module):
    """Full USAM training-time module: Conductor + SmokePlayer + heads + losses.

    The constructor takes a single :class:`_USAMTrainConfig`. Use
    :meth:`from_dict` for OmegaConf / YAML interop.

    Forward contract
    ----------------
    The training loop calls :meth:`training_step(batch, weights, step)`
    where ``batch`` is the dict produced by
    :class:`usam.dataloader.usam_lerobot.USAMLeRobotDataset.__getitem__`
    after collation. The method:

    1. Runs the Conductor on the head keyframe RGB.
    2. Refreshes the plan cache (and pushes a snapshot to its history).
    3. **Calls** :func:`apply_cache_dropout` to maybe substitute a stale
       cache for this step.
    4. Runs the SmokePlayer to produce the three velocity predictions
       plus the geom sub-dict.
    5. Builds drift + subtask predictions.
    6. Aggregates with :class:`USAMUnifiedLoss` and returns
       ``(total_loss, per_loss_dict)``.

    The Conductor uses :class:`MockConductorBackbone` by default; the
    real Qwen3-VL-4B is wired in by the H200 inference engineer in Wave 4.
    """

    def __init__(self, cfg: _USAMTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Conductor with mock backbone (Wave-4 swaps for real Qwen3-VL).
        backbone = MockConductorBackbone(
            hidden_size=cfg.backbone_hidden,
            seq_len=cfg.backbone_seq_len,
        )
        self.conductor = Conductor(
            qwen_ckpt="",
            n_plan_tokens=cfg.n_plan_tokens,
            player_d_model=cfg.hidden_size,
            e_proj_dim=cfg.e_proj_dim,
            backbone_override=backbone,
            backbone_hidden=cfg.backbone_hidden,
            backbone_seq_len=cfg.backbone_seq_len,
        )

        # Player.
        self.player = SmokePlayer(
            hidden_size=cfg.hidden_size,
            rgb_dim=cfg.rgb_dim,
            depth_dim=cfg.depth_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            action_dim=cfg.action_dim,
            action_chunk=cfg.action_chunk,
            n_keep_tokens=cfg.n_keep_tokens,
            proprio_dim=cfg.proprio_dim,
        )
        self.player.make_plan_kv_projs()  # eagerly allocate

        # Drift MLP — takes RGB-DINO [CLS] + projected committed embedding.
        self.drift_mlp = FDriftMLP(
            rgb_dino_dim=cfg.rgb_dim,
            e_dim=cfg.e_proj_dim,
            hidden=64,
        )

        # Subtask classifier.
        self.subtask_head = SubtaskCompletionHead(
            e_dim=cfg.e_proj_dim,
            obs_dim=cfg.rgb_dim,
            proprio_dim=cfg.proprio_dim,
            window=4,  # smoke fixture has 4 history frames
            hidden=64,
        )

        # Unified loss aggregator.
        # `dim` for GeomConsistencyLoss must match the modality embed_dim.
        self.loss_fn = USAMUnifiedLoss(
            weights=LossWeights(),  # weights are passed per-step
            geom_kwargs={"dim": cfg.rgb_dim, "hidden": 32, "tau": 1.0},
        )

        # Plan cache (1 layer of plan tokens — match SmokePlayer's depth).
        self.plan_cache = PlanCache(
            n_layers=cfg.num_layers,
            d_model=cfg.hidden_size,
            n_plan=cfg.n_plan_tokens,
            dtype=torch.float32,  # CPU plumbing test runs in fp32
            history_size=8,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "USAMTrainModel":
        """Build from a flat dict (OmegaConf-friendly)."""
        cfg = _USAMTrainConfig(**{k: v for k, v in raw.items() if k in _USAMTrainConfig.__dataclass_fields__})
        return cls(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _refresh_plan_cache(self, head_keyframe: Tensor, t: int) -> None:
        """Conductor pass + cache refresh. Called before the Player runs.

        Cache-dropout is applied **immediately after** this refresh — see
        the docstring of :mod:`usam.train` for the contract.
        """
        # Run the (frozen) backbone end-to-end. The mock backbone takes
        # raw pixels of shape [B, 3, H, W]; we tile the [B, D] head
        # feature into a deterministic small image so the smoke test is
        # reproducible. Production Qwen3-VL will receive real images.
        b, d = head_keyframe.shape
        # Map the D-dim feature to a 3-channel 8x8 pixel-style tensor.
        # The exact transform doesn't matter for the smoke test; we
        # just need a deterministic, differentiable function. We use a
        # padded reshape that is shape-stable across D.
        H = 8
        W = 8
        target = 3 * H * W  # = 192
        if d >= target:
            flat = head_keyframe[:, :target]
        else:
            flat = torch.cat(
                [head_keyframe, head_keyframe.new_zeros(b, target - d)], dim=-1
            )
        pixel_proxy = flat.reshape(b, 3, H, W)
        out = self.conductor.encode(pixel_proxy)
        k_projs, v_projs = self.player.make_plan_kv_projs()
        self.plan_cache.refresh(
            p_hat=out.P_hat,
            e=out.e,
            k_projs_image=k_projs,
            v_projs_image=v_projs,
            k_projs_action=None,
            v_projs_action=None,
            t=int(t),
        )

    def training_step(
        self,
        batch: Mapping[str, Any],
        weights: LossWeights,
        step: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """One training step. Returns ``(total_loss, per_loss_dict)``."""
        assert isinstance(weights, LossWeights)
        # Sync the loss aggregator's weights with the ramped values for
        # this step; per-component weights are cheap to mutate.
        self.loss_fn.weights = weights

        # ------------------------------------------------------------------
        # Pull tensors out of the batch dict. The dataloader emits a
        # mixture of Tensors, ints, strs and bools. We only consume the
        # tensor fields here; the rest is for the Conductor's NLP path,
        # which the smoke run skips.
        # ------------------------------------------------------------------
        rgb = batch["rgb_dino_seq"]  # [B, T, N, D]
        depth = batch.get("depth_dino_seq", torch.zeros_like(rgb))
        proprio = batch["proprio"]
        action_chunk = batch["action_chunk"]
        head_keyframe = batch.get("head_keyframe_rgb_dino")
        if head_keyframe is None:
            # Fall back to mean of head-camera RGB; smoke fixture always has it.
            head_keyframe = rgb.mean(dim=(1, 2))
        elif head_keyframe.dim() == 3:
            # Drop the token dim if present — keep [CLS] only.
            head_keyframe = head_keyframe[:, 0]
        elif head_keyframe.dim() == 4:
            head_keyframe = head_keyframe.mean(dim=(1, 2))

        # Convert to the player's compute dtype.
        compute_dtype = next(self.player.parameters()).dtype
        rgb = rgb.to(compute_dtype)
        depth = depth.to(compute_dtype)
        proprio = proprio.to(compute_dtype)
        action_chunk = action_chunk[..., : self.cfg.action_dim].to(compute_dtype)
        head_keyframe = head_keyframe.to(compute_dtype)

        # ------------------------------------------------------------------
        # 1. Conductor refresh (the cache-dropout call lives below).
        # 2. apply_cache_dropout(cache, t) — see line ~ in usam/train.py
        #    docstring. This MUST happen after every refresh.
        # ------------------------------------------------------------------
        self._refresh_plan_cache(head_keyframe, t=step)
        active_cache = apply_cache_dropout(
            self.plan_cache,
            t=step,
            p=self.cfg.cache_dropout_p,
            window=self.cfg.cache_dropout_window,
        )

        # ------------------------------------------------------------------
        # 3. Player forward — emits the three velocity heads.
        # The dataloader does not provide ground-truth velocity targets —
        # the production scheduler does. For the smoke run we use the
        # *input* features as targets for the modality heads, and the
        # action chunk itself as the action target (rectified-flow at
        # noise level 0).
        # ------------------------------------------------------------------
        # Build a noisy action proxy: small random perturbation of the GT.
        noise = 0.1 * torch.randn_like(action_chunk)
        action_noisy = action_chunk + noise

        preds = self.player(
            rgb_dino_seq=rgb,
            depth_dino_seq=depth,
            proprio=proprio,
            action_noisy=action_noisy,
            plan_cache=active_cache,
        )

        # Drift + subtask predictions.
        rgb_cls = rgb[:, 0, 0]  # [B, D]
        e_committed = active_cache.committed_emb  # [B, e_proj_dim]
        drift_pred = self.drift_mlp(rgb_cls, e_committed)
        # Drift target: a fresh Conductor pass would give us this in
        # production; for smoke we regress toward the committed embedding
        # itself (the trivial target — useful only for plumbing).
        drift_target = e_committed.detach()

        # Subtask classifier needs a window of [obs_cls, proprio]. We
        # synthesize a length-4 window by replicating the current frame
        # — this matches our smoke history_frames.
        win_obs = rgb_cls.unsqueeze(1).expand(-1, 4, -1).contiguous()
        win_prop = proprio.unsqueeze(1).expand(-1, 4, -1).contiguous()
        subtask_logit = self.subtask_head(e_committed, win_obs, win_prop).squeeze(-1)
        subtask_label = batch.get("subtask_label")
        if subtask_label is None:
            subtask_label = torch.zeros_like(subtask_logit)
        else:
            subtask_label = subtask_label.to(subtask_logit.dtype)
            if subtask_label.dim() > 1:
                subtask_label = subtask_label.view(-1)

        preds["drift"] = drift_pred
        preds["subtask"] = subtask_logit

        targets: Dict[str, Tensor] = {
            "action": action_chunk,
            "image": rgb,
            "depth": depth,
            "drift": drift_target,
            "subtask": subtask_label,
        }
        masks: Dict[str, Optional[Tensor]] = {}

        return self.loss_fn(preds, targets, masks)


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Save every ``every_steps`` steps; keep last ``keep_last`` + best.

    "Best" is tracked by lowest validation loss seen so far. The manager
    writes to ``<output_dir>/checkpoints/`` and also touches
    ``<output_dir>/checkpoints/latest_step.txt`` after every save so
    external watchers can poll without parsing filenames.

    Each checkpoint is a single ``.pt`` file containing:

    * ``state_dict``    — the model's state dict.
    * ``optimizer``     — the optimizer's state dict.
    * ``scheduler``     — LR scheduler's state dict (or ``None``).
    * ``step``          — global step at save time.
    * ``run``           — :class:`RunMetadata` as a dict.
    * ``best_val_loss`` — current best validation loss.
    * ``timestamp``     — ``time.time()``.
    """

    def __init__(
        self,
        output_dir: Path,
        run: RunMetadata,
        every_steps: int = 5_000,
        keep_last: int = 3,
    ) -> None:
        assert isinstance(output_dir, Path)
        assert every_steps > 0, "every_steps must be positive"
        assert keep_last >= 0
        self.output_dir = output_dir
        self.run = run
        self.every_steps = int(every_steps)
        self.keep_last = int(keep_last)
        self.ckpt_dir = output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss: Optional[float] = None
        self.saved_steps: List[int] = []

    def maybe_save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        val_loss: Optional[float] = None,
    ) -> Optional[Path]:
        """Save if ``step % every_steps == 0``. Always-save best by val.

        Returns the checkpoint path on save, else ``None``.
        """
        path: Optional[Path] = None
        if step > 0 and step % self.every_steps == 0:
            path = self._save(step, model, optimizer, scheduler, val_loss, tag=None)
            self.saved_steps.append(step)
            self._evict()

        # Best by val: independent of the step modulus.
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self._save(step, model, optimizer, scheduler, val_loss, tag="best")

        return path

    def _save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        val_loss: Optional[float],
        tag: Optional[str],
    ) -> Path:
        if tag == "best":
            fname = f"checkpoint_best.pt"
        else:
            fname = f"checkpoint_step{step:08d}.pt"
        path = self.ckpt_dir / fname
        payload: Dict[str, Any] = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": int(step),
            "run": dataclasses.asdict(self.run),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "timestamp": time.time(),
        }
        torch.save(payload, str(path))
        # Side-channel marker for external watchers.
        (self.ckpt_dir / "latest_step.txt").write_text(f"{step}\n{path.name}\n")
        return path

    def _evict(self) -> None:
        """Drop oldest non-best checkpoints beyond ``keep_last``."""
        # Re-scan disk in case the user resumed; rely on filenames.
        steps: List[Tuple[int, Path]] = []
        for p in self.ckpt_dir.glob("checkpoint_step*.pt"):
            try:
                step = int(p.stem.replace("checkpoint_step", ""))
                steps.append((step, p))
            except ValueError:
                continue
        steps.sort(key=lambda x: x[0])
        excess = len(steps) - self.keep_last
        for i in range(excess):
            steps[i][1].unlink(missing_ok=True)
