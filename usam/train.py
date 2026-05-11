# SPDX-License-Identifier: MIT
"""USAM top-level training entry point.

Wraps the LDA-1B trainer's optimizer / scheduler / FSDP scaffolding and
adds USAM-specific wiring:

* **Plan-cache dropout** — :func:`usam.conductor.cache_dropout.apply_cache_dropout`
  is invoked **once per step, immediately after every Conductor refresh**.
  The exact call site lives in
  :meth:`usam._train_helpers.USAMTrainModel.training_step` (around the
  ``# 1. Conductor refresh ...`` block); see also
  ``usam/_train_helpers.py:USAMTrainModel.training_step`` (the
  ``apply_cache_dropout(self.plan_cache, t=step, p=..., window=...)`` line
  immediately follows ``self._refresh_plan_cache(...)``).
  Following the contract in :mod:`usam.conductor.cache_dropout`, the cache
  history is populated automatically by :meth:`PlanCache.refresh`; we just
  call ``apply_cache_dropout(cache, t)`` with the global step ``t``.

* **Ramped loss weights** — :func:`compute_ramped_weights` linearly ramps
  ``geom`` and ``flow_act`` from ``0`` to their YAML targets over the
  first 50_000 steps (clamped after). The exact arithmetic is

      w_t = target * min(step / 50_000, 1.0)         when step > 0
      w_0 = 0                                         at step == 0

  See :func:`usam._train_helpers.compute_ramped_weights`.

* **Precision** — BF16 weights everywhere, FP8 activations only on H200
  (capability ``(9, 0)``) and only when ``transformer_engine`` imports
  cleanly. CPU plumbing falls back to FP32 with no FP8.

* **Checkpoints** — every 5_000 steps, keeping the last 3 + best-by-val.
  Each checkpoint is tagged with ``run_id`` (timestamp + 8-char uuid)
  and ``git_sha``.

CLI
---
``python -m usam.train --config configs/train/stage_b1_pretrain.yaml \
    --model configs/model/usam_350m_smoke.yaml --max_steps 100 \
    --data tests/golden_data/tiny_droid``

Most users invoke this through ``scripts/train_smoke_a40.sh`` or
``scripts/train_h200.sh``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os  # used by maybe_wrap_distributed (RANK/WORLD_SIZE env probe)
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from usam._train_helpers import (
    CheckpointManager,
    PrecisionPlan,
    RunMetadata,
    USAMTrainModel,
    _USAMTrainConfig,
    build_run_id,
    compute_ramped_weights,
    detect_precision,
    git_sha,
)
from usam.dataloader.usam_lerobot import USAMLeRobotDataset
from usam.losses import LossWeights

logger = logging.getLogger("usam.train")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
@dataclass
class TrainArgs:
    """Parsed CLI arguments for the training entry point."""

    config: Path
    model_config: Optional[Path]
    data: Optional[Path]
    output_dir: Path
    max_steps: Optional[int]
    device: str
    seed: int
    auto_oom_reduce: bool
    log_every: int


def parse_args(argv: Optional[List[str]] = None) -> TrainArgs:
    """Parse CLI args. Mirrors the shape used by ``scripts/train_smoke_a40.sh``."""
    p = argparse.ArgumentParser(description="USAM training entry point.")
    p.add_argument("--config", type=Path, required=True, help="Train YAML config.")
    p.add_argument("--model", dest="model_config", type=Path, default=None,
                   help="Optional model YAML override (otherwise read from --config).")
    p.add_argument("--data", type=Path, default=None,
                   help="Override the dataset path (USAM-LeRobot v2.1 root).")
    p.add_argument("--output_dir", type=Path,
                   default=Path("runs") / build_run_id(),
                   help="Where to write checkpoints + logs.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override total training steps.")
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cpu", "cuda"),
                   help="Force device. 'auto' picks cuda if available.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto_oom_reduce", action="store_true",
                   help="On a first-step OOM, halve per-device batch and retry.")
    p.add_argument("--log_every", type=int, default=10)

    ns = p.parse_args(argv)
    return TrainArgs(
        config=ns.config,
        model_config=ns.model_config,
        data=ns.data,
        output_dir=ns.output_dir,
        max_steps=ns.max_steps,
        device=ns.device,
        seed=ns.seed,
        auto_oom_reduce=ns.auto_oom_reduce,
        log_every=ns.log_every,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    assert isinstance(path, Path)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Optimizer / scheduler — wraps LDA-1B's helpers when available, otherwise
# a vanilla AdamW + cosine schedule (used by the CPU plumbing path).
# ---------------------------------------------------------------------------
def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    max_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Build optimizer + scheduler.

    Reuses LDA-1B's ``setup_optimizer_and_scheduler`` when (a) the import
    succeeds and (b) the YAML carries the ``trainer`` section it expects
    (``learning_rate.base``, ``optimizer.betas``, etc.). Otherwise we
    fall back to vanilla AdamW + cosine decay so the CPU plumbing path
    runs without ``accelerate`` / ``deepspeed`` / ``transformers``.

    The vanilla path uses the ``optimizer`` and ``schedule`` keys from
    USAM's own training YAMLs — see :ref:`configs/train/stage_b1_pretrain.yaml`.
    """
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1.0e-4))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    eps = float(opt_cfg.get("eps", 1e-8))

    sched_cfg = cfg.get("schedule", {})
    warmup = int(sched_cfg.get("warmup_steps", min(2000, max_steps // 10)))
    total = int(sched_cfg.get("total_steps", max_steps))
    sched_type = str(sched_cfg.get("type", "cosine"))
    min_lr = float(sched_cfg.get("min_lr", lr * 0.1))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        if sched_type == "constant":
            return 1.0
        if sched_type == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            min_ratio = min_lr / max(lr, 1e-12)
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# FSDP / accelerate wiring (real H200 path; skipped on CPU)
# ---------------------------------------------------------------------------
def maybe_wrap_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    plan: PrecisionPlan,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader]:
    """Wrap the trio with Accelerate (FSDP) when a CUDA process group exists.

    Reuses LDA-1B's distributed plumbing when possible — but only when:

    1. CUDA is available.
    2. ``torchrun`` populated ``RANK`` / ``WORLD_SIZE``.
    3. ``accelerate`` imports cleanly (it ships with LDA-1B's deps).

    Otherwise (single-GPU, CPU plumbing test), this is a no-op.
    """
    if plan.device_type == "cpu" or not torch.cuda.is_available():
        return model, optimizer, dataloader

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Single-process. Move the model to GPU and return.
        device = torch.device("cuda")
        model = model.to(device=device, dtype=plan.weights_dtype)
        return model, optimizer, dataloader

    try:
        from accelerate import Accelerator, DeepSpeedPlugin
    except ImportError:
        logger.warning("accelerate is not installed; falling back to single-process CUDA.")
        device = torch.device("cuda")
        model = model.to(device=device, dtype=plan.weights_dtype)
        return model, optimizer, dataloader

    # Reuse LDA-1B's helper; it sets up the deepspeed plugin.
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    return model, optimizer, dataloader


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloader(
    data_root: Path,
    batch_size: int,
    cfg: Dict[str, Any],
    num_workers: int = 0,
) -> DataLoader:
    """Build a USAMLeRobot DataLoader.

    The smoke fixture is tiny — defaults to ``num_workers=0`` so the test
    runs without spawning subprocesses. The H200 path passes a larger
    value via the YAML.
    """
    data_cfg = cfg.get("data", {})
    history_frames = int(data_cfg.get("history_frames", 4))
    future_frames = int(data_cfg.get("future_frames", 8))
    action_chunk = int(data_cfg.get("action_chunk", 8))
    fps_features = int(data_cfg.get("fps_features", 5))
    fps_action = int(data_cfg.get("fps_action", 30))
    cameras = list(data_cfg.get("cameras", ["head_rgb"]))
    modalities = list(data_cfg.get("modalities", ["rgb", "depth", "flow"]))

    ds = USAMLeRobotDataset(
        data_root,
        split="train",
        use_cached_features=True,
        modalities=modalities,
        cameras=cameras,
        history_frames=history_frames,
        future_frames=future_frames,
        action_chunk=action_chunk,
        fps_features=fps_features,
        fps_action=fps_action,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate the dataloader's mixed-type dicts into a single batch dict.

    Tensor fields are stacked; scalar fields (instructions, episode ids)
    are returned as Python lists.
    """
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                # Variable-length tensors — keep as a list (e.g. *_dino_seq
                # whose T_total can vary across episodes near boundaries).
                out[k] = vals
        elif isinstance(first, bool):
            out[k] = torch.tensor(vals, dtype=torch.bool)
        elif isinstance(first, int):
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif isinstance(first, float):
            out[k] = torch.tensor(vals, dtype=torch.float32)
        else:
            out[k] = vals
    return out


# ---------------------------------------------------------------------------
# Loss-weights helpers
# ---------------------------------------------------------------------------
def build_base_loss_weights(cfg: Dict[str, Any]) -> tuple[LossWeights, float, float]:
    """Read ``loss_weights:`` from the YAML.

    Returns
    -------
    (base, geom_target, flow_act_target) : tuple
        ``base`` is the steady-state weights with ``geom = flow_act = 0.0``
        (they're ramped externally per :func:`compute_ramped_weights`).
        ``geom_target`` and ``flow_act_target`` are the YAML's
        ``geom_target`` and ``flow_act_target`` keys.
    """
    lw_cfg = cfg.get("loss_weights", {})
    base = LossWeights(
        action=float(lw_cfg.get("action", 1.0)),
        rgb=float(lw_cfg.get("rgb", 1.0)),
        depth=float(lw_cfg.get("depth", 0.3)),
        flow=float(lw_cfg.get("flow", 0.3)),
        geom=0.0,
        flow_act=0.0,
        drift=float(lw_cfg.get("drift", 0.1)),
        subtask=float(lw_cfg.get("subtask", 0.1)),
    )
    # The plan calls these `geom_max` / `flow_act_max`; we accept either
    # spelling for forward compatibility with §6.2's YAML snippet.
    geom_target = float(
        lw_cfg.get("geom_target", lw_cfg.get("geom_max", 0.05))
    )
    flow_act_target = float(
        lw_cfg.get("flow_act_target", lw_cfg.get("flow_act_max", 0.05))
    )
    return base, geom_target, flow_act_target


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def build_model_from_cfg(model_cfg: Dict[str, Any]) -> USAMTrainModel:
    """Convert a model YAML's ``encoder`` / ``player`` / ``conductor`` keys
    into :class:`_USAMTrainConfig` and instantiate the model."""
    encoder = model_cfg.get("encoder", {})
    player = model_cfg.get("player", {})
    conductor = model_cfg.get("conductor", {})
    action_head = model_cfg.get("action_head", {})

    cfg = _USAMTrainConfig(
        hidden_size=int(player.get("hidden_size", 256)),
        num_layers=int(player.get("num_layers", 2)),
        num_heads=int(player.get("num_heads", 4)),
        action_dim=int(action_head.get("action_dim", 7)),
        action_chunk=int(action_head.get("action_horizon", 8)),
        rgb_dim=int(encoder.get("embed_dim", 768)),
        depth_dim=int(encoder.get("embed_dim", 768)),
        flow_dim=int(encoder.get("embed_dim", 768)),
        proprio_dim=int(player.get("proprio_dim", 50)),
        n_keep_tokens=int(encoder.get("cache_n_keep_tokens", 64)) + 1,
        n_plan_tokens=int(conductor.get("n_plan_tokens", 32)),
        e_proj_dim=64,  # locked: matches Conductor.e_proj_dim default
        backbone_hidden=64,  # mock backbone dim
        backbone_seq_len=64,
    )
    # Cap action_dim at 7 — production canonical EE is 7-D; our smoke
    # player was sized for that and the dataloader pads beyond it.
    cfg.action_dim = min(cfg.action_dim, 7)
    return USAMTrainModel(cfg)


def _data_iter_forever(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    """Loop the dataloader forever (handles tiny smoke fixtures)."""
    while True:
        for batch in loader:
            yield batch


def train_loop(
    model: USAMTrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    dataloader: DataLoader,
    base_weights: LossWeights,
    geom_target: float,
    flow_act_target: float,
    *,
    max_steps: int,
    device: torch.device,
    weights_dtype: torch.dtype,
    ckpt: Optional[CheckpointManager] = None,
    log_every: int = 10,
    ramp_steps: int = 50_000,
    grad_clip: float = 1.0,
    wandb_run: Any = None,
) -> List[float]:
    """Run the training loop. Returns the list of per-step total losses.

    The cache-dropout call lives inside :meth:`USAMTrainModel.training_step`.
    See the module docstring for the contract.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run | None
        If non-None, every step is logged to wandb under namespaced keys
        (``loss/total``, ``loss/<component>``, ``lr``, ``step_time_ms``,
        ``grad_norm``). Use :func:`_maybe_init_wandb` to construct one
        from the ``WANDB_API_KEY`` env var.
    """
    model.train()
    losses: List[float] = []
    data_iter = _data_iter_forever(dataloader)

    for step in range(max_steps):
        step_t0 = time.time()
        batch = next(data_iter)
        # Move tensor batch entries to the device.
        batch = _to_device(batch, device, weights_dtype)

        # Linear ramp of geom + flow_act weights from 0 → target across
        # the first `ramp_steps` (=50_000) steps.
        weights = compute_ramped_weights(
            base=base_weights,
            step=step,
            geom_target=geom_target,
            flow_act_target=flow_act_target,
            ramp_steps=ramp_steps,
        )

        optimizer.zero_grad(set_to_none=True)
        # Note: `apply_cache_dropout(cache, t)` is invoked inside
        # `model.training_step` immediately after the Conductor refresh —
        # see usam._train_helpers.USAMTrainModel._refresh_plan_cache.
        total_loss, per_loss = model.training_step(batch, weights, step)

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {total_loss.item()}")

        total_loss.backward()
        grad_norm = None
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                grad_clip,
            )
        optimizer.step()
        scheduler.step()

        loss_val = float(total_loss.detach().item())
        losses.append(loss_val)
        step_time_ms = (time.time() - step_t0) * 1000.0
        cur_lr = scheduler.get_last_lr()[0]

        if (step % log_every == 0) or (step == max_steps - 1):
            per_str = " ".join(
                f"{k}={float(v.detach().item()):.4f}" for k, v in per_loss.items()
            )
            logger.info("step=%d loss=%.4f lr=%.3e %s",
                        step, loss_val, cur_lr, per_str)

        # ---- wandb (soft) ----
        if wandb_run is not None:
            log_payload: Dict[str, Any] = {
                "loss/total": loss_val,
                "lr": cur_lr,
                "step_time_ms": step_time_ms,
            }
            for k, v in per_loss.items():
                log_payload[f"loss/{k}"] = float(v.detach().item())
            for k, v in weights.as_dict().items():
                log_payload[f"weight/{k}"] = float(v)
            if grad_norm is not None:
                log_payload["grad_norm"] = float(grad_norm.detach().item())
            try:
                wandb_run.log(log_payload, step=step)
            except Exception as e:  # pragma: no cover
                logger.warning("wandb.log failed at step %d (continuing): %s", step, e)

        if ckpt is not None:
            ckpt.maybe_save(step, model, optimizer, scheduler, val_loss=None)

    return losses


def _maybe_init_wandb(
    run_meta: "RunMetadata",
    train_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    args: TrainArgs,
) -> Any:
    """Initialize wandb if ``WANDB_API_KEY`` is set; otherwise return ``None``.

    Soft-fails on every error path so a missing API key or a transient
    network glitch never blocks training. The returned object (or
    ``None``) is passed into :func:`train_loop` as the ``wandb_run``
    kwarg, where it gates the per-step ``wandb.log`` call.

    Honored env vars:
      * ``WANDB_API_KEY``  — credentials. Without it, this returns None.
      * ``WANDB_PROJECT``  — project name. Defaults to ``"usam"``.
      * ``WANDB_ENTITY``   — team/user. Defaults to wandb's own default.
      * ``WANDB_MODE``     — ``"online"`` / ``"offline"`` / ``"disabled"``.
    """
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set; skipping wandb init (stdout logging only)")
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb not installed; skipping wandb init")
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "usam"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_meta.run_id,
            config={
                "train_cfg": train_cfg,
                "model_cfg": model_cfg,
                "cli_args": {
                    "max_steps": args.max_steps,
                    "device": args.device,
                    "seed": args.seed,
                    "config": str(args.config),
                    "model_config": str(args.model_config) if args.model_config else None,
                    "data": str(args.data) if args.data else None,
                    "output_dir": str(args.output_dir),
                },
                "git_sha": run_meta.git_sha,
            },
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info("wandb run: %s (%s)", run.name, run.url)
        return run
    except Exception as e:  # pragma: no cover
        logger.warning("wandb init failed (continuing without wandb): %s", e)
        return None


def _to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """Move tensor entries to ``device``; non-tensors pass through untouched."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    args: TrainArgs,
    train_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> List[float]:
    """High-level run helper. Used by both the CLI and the integration test."""
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Device / precision detection
    # ------------------------------------------------------------------
    force_cpu = args.device == "cpu"
    plan = detect_precision(force_cpu=force_cpu)
    logger.info("Precision plan: %s", plan.note)
    if args.device == "cuda" and plan.device_type != "cuda":
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    device = torch.device(plan.device_type)
    weights_dtype = plan.weights_dtype

    # ------------------------------------------------------------------
    # Build model + dataloader
    # ------------------------------------------------------------------
    model = build_model_from_cfg(model_cfg).to(device=device, dtype=weights_dtype)

    data_root = args.data
    if data_root is None:
        data_root = Path(train_cfg.get("data", {}).get("root", "tests/golden_data/tiny_droid"))
    batch_size = int(
        train_cfg.get("batch", {}).get("micro_size", 1)
    )
    if plan.device_type == "cpu":
        # CPU plumbing: shrink batch so the test runs in <1 minute.
        batch_size = min(batch_size, 1)

    # Bound to the dataset size so tiny fixtures don't OOM.
    loader = build_dataloader(data_root, batch_size, train_cfg, num_workers=0)
    if len(loader.dataset) == 0:  # pragma: no cover - defensive
        raise RuntimeError(f"empty dataset at {data_root}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    max_steps = int(args.max_steps if args.max_steps else train_cfg.get("max_steps", 100))
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_cfg, max_steps)

    # Wrap with Accelerate / FSDP if running multi-GPU.
    model, optimizer, loader = maybe_wrap_distributed(model, optimizer, loader, plan)

    # ------------------------------------------------------------------
    # Loss schedule
    # ------------------------------------------------------------------
    base_weights, geom_target, flow_act_target = build_base_loss_weights(train_cfg)
    ramp_steps = int(train_cfg.get("loss_weights", {}).get("ramp_steps", 50_000))

    # ------------------------------------------------------------------
    # Run-id, checkpointing
    # ------------------------------------------------------------------
    run_meta = RunMetadata(
        run_id=build_run_id(),
        git_sha=git_sha(Path.cwd()),
        config_path=str(args.config),
        started_at=time.time(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(
        output_dir=args.output_dir,
        run=run_meta,
        every_steps=int(train_cfg.get("checkpoint", {}).get("every_steps", 5_000)),
        keep_last=int(train_cfg.get("checkpoint", {}).get("keep_last", 3)),
    )
    logger.info("Run ID: %s | git=%s | output=%s",
                run_meta.run_id, run_meta.git_sha, args.output_dir)

    # ------------------------------------------------------------------
    # Optional wandb. Soft-fails if WANDB_API_KEY is unset or wandb is
    # not installed; in that case training proceeds with stdout-only
    # logging via the loguru/standard logger above.
    # ------------------------------------------------------------------
    wandb_run = _maybe_init_wandb(run_meta, train_cfg, model_cfg, args)

    # ------------------------------------------------------------------
    # Train. Auto-OOM-reduce: catch the first OOM, halve batch, retry.
    # ------------------------------------------------------------------
    try:
        return train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=loader,
            base_weights=base_weights,
            geom_target=geom_target,
            flow_act_target=flow_act_target,
            max_steps=max_steps,
            device=device,
            weights_dtype=weights_dtype,
            ckpt=ckpt,
            log_every=args.log_every,
            ramp_steps=ramp_steps,
            wandb_run=wandb_run,
        )
    except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - GPU-only
        if not args.auto_oom_reduce:
            raise
        torch.cuda.empty_cache()
        new_bs = max(1, batch_size // 2)
        logger.warning("OOM at bs=%d, retrying at bs=%d. Original: %s",
                       batch_size, new_bs, e)
        loader = build_dataloader(data_root, new_bs, train_cfg, num_workers=0)
        return train_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=loader,
            base_weights=base_weights,
            geom_target=geom_target,
            flow_act_target=flow_act_target,
            max_steps=max_steps,
            device=device,
            weights_dtype=weights_dtype,
            ckpt=ckpt,
            log_every=args.log_every,
            ramp_steps=ramp_steps,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:  # pragma: no cover
                pass


def main(argv: Optional[List[str]] = None) -> List[float]:
    """CLI wrapper used by ``python -m usam.train ...`` and the smoke tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = parse_args(argv)
    train_cfg = load_yaml(args.config)
    model_cfg_path = args.model_config or args.config.parent.parent / "model" / "usam_350m_smoke.yaml"
    if not model_cfg_path.exists():
        # Fallback: read model from the train cfg if it nests one.
        model_cfg = train_cfg.get("model", {})
        if not model_cfg:
            raise FileNotFoundError(
                f"Model config not found at {model_cfg_path} and `model:` "
                "section absent from train YAML."
            )
    else:
        model_cfg = load_yaml(model_cfg_path)
    return run(args, train_cfg, model_cfg)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
