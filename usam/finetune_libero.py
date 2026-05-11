# SPDX-License-Identifier: MIT
"""LIBERO finetuning CLI entry point.

This module provides a top-level CLI for finetuning a base USAM checkpoint
on a LIBERO trajectory suite (``libero_10`` / ``libero_object`` / ``libero_spatial``
/ ``libero_goal``). It mirrors the CLI shape used by :mod:`usam.train` and
:mod:`prep.adapter_pretrain`.

Pipeline summary
----------------
1. Parse CLI args (argparse, mirroring :func:`prep.adapter_pretrain.main`).
2. Load the model YAML config (via :func:`yaml.safe_load`).
3. Build the model from config — uses the same construction path as
   :func:`usam.train.build_model_from_cfg` (:class:`usam._train_helpers.USAMTrainModel`).
4. Load the base checkpoint via :func:`torch.load` + :meth:`load_state_dict`
   (``strict=False`` to allow loading partial-trained smoke checkpoints).
5. Build a LIBERO trajectory dataset (see :class:`LiberoTrajectoryDataset`).
6. Build optimizer + cosine-with-warmup scheduler.
7. Run the training loop. **The loop body is currently a stub** — it
   consumes one real batch from the dataloader per step and prints
   progress; future contributors fill in the forward + loss + backward
   without touching this CLI plumbing.
8. Save the finetuned checkpoint to ``<output_dir>/finetune_ckpt.pt``.

LIBERO dataset adapter
----------------------
There is no first-party ``libero`` Python package in the qwen3vl env.
:class:`LiberoTrajectoryDataset` walks the ``--libero-data`` directory for
``*.hdf5`` files (LIBERO ships HDF5 trajectory dumps) and reads them with
:mod:`h5py` when available. If no HDF5 files are found, the dataset
fabricates a tiny synthetic batch of the right shape so the CLI plumbing
remains testable on bare developer boxes. The "real interface, synthetic
data" path is flagged in the run log so production users see it.

CLI
---
``python -m usam.finetune_libero \
    --base-ckpt /path/to/pretrain.pt \
    --libero-data /path/to/libero/data \
    --output-dir /path/to/output \
    --max-steps 10000 \
    --model-config configs/model/usam_350m_smoke.yaml \
    --suite libero_10 \
    --learning-rate 1e-5 \
    --batch-size 8 \
    --eval-every 500 \
    --log-every 50``
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

# loguru is in requirements.txt — see /localhome/local-chrislin/USAM/requirements.txt
# but we soft-import so the module remains testable in stripped-down envs.
try:
    from loguru import logger as _loguru_logger
except ImportError:  # pragma: no cover - loguru is in requirements
    import logging as _logging

    _loguru_logger = _logging.getLogger("usam.finetune_libero")

__all__ = [
    "FinetuneArgs",
    "LiberoTrajectoryDataset",
    "build_model_from_config",
    "load_model_config",
    "make_cosine_warmup_scheduler",
    "parse_args",
    "run_finetune",
    "main",
]


# ---------------------------------------------------------------------------
# Argument parsing — mirrors `prep/adapter_pretrain.py::main`
# ---------------------------------------------------------------------------
@dataclass
class FinetuneArgs:
    """Parsed CLI arguments for the LIBERO finetune entry point."""

    base_ckpt: Path
    libero_data: Path
    output_dir: Path
    max_steps: int
    model_config: Path
    suite: str
    learning_rate: float
    batch_size: int
    eval_every: int
    log_every: int
    device: str
    seed: int


def parse_args(argv: Optional[List[str]] = None) -> FinetuneArgs:
    """Parse CLI args. Mirrors the argparse style of :mod:`prep.adapter_pretrain`."""
    p = argparse.ArgumentParser(
        description="USAM LIBERO finetuning entry point.",
    )
    p.add_argument(
        "--base-ckpt",
        type=Path,
        required=True,
        help="Path to the pretrained USAM checkpoint (.pt) to finetune from.",
    )
    p.add_argument(
        "--libero-data",
        type=Path,
        required=True,
        help="Root directory containing LIBERO trajectory HDF5 files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the finetuned checkpoint and logs.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Total finetune steps. Default: 10000.",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model YAML config (e.g. configs/model/usam_350m_smoke.yaml).",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="libero_10",
        choices=("libero_10", "libero_object", "libero_spatial", "libero_goal"),
        help="LIBERO benchmark suite. Default: libero_10.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="AdamW base learning rate. Default: 1e-5.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-process batch size. Default: 8.",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Run eval every N steps. Default: 500.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N steps. Default: 50.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Force device. 'auto' picks cuda if available. Default: auto.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed. Default: 0.",
    )

    ns = p.parse_args(argv)
    return FinetuneArgs(
        base_ckpt=ns.base_ckpt,
        libero_data=ns.libero_data,
        output_dir=ns.output_dir,
        max_steps=int(ns.max_steps),
        model_config=ns.model_config,
        suite=str(ns.suite),
        learning_rate=float(ns.learning_rate),
        batch_size=int(ns.batch_size),
        eval_every=int(ns.eval_every),
        log_every=int(ns.log_every),
        device=str(ns.device),
        seed=int(ns.seed),
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_model_config(path: Path) -> Dict[str, Any]:
    """Load and validate a model YAML config.

    Validates that the YAML contains the three top-level sections the
    USAM model expects: ``encoder``, ``player``, ``action_head``. Raises
    ``ValueError`` with a clear message if any are missing.

    Parameters
    ----------
    path : Path
        Path to the model YAML config.

    Returns
    -------
    dict
        Plain dict view of the YAML.
    """
    assert isinstance(path, Path), f"expected Path, got {type(path).__name__}"
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Model config at {path} did not parse to a dict.")

    required = ("encoder", "player", "action_head")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"Model config at {path} is missing required section(s): {missing}. "
            f"Required sections: {required}."
        )
    return cfg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model_from_config(model_cfg: Dict[str, Any]) -> nn.Module:
    """Build a USAM training model from a YAML config.

    Reuses :func:`usam.train.build_model_from_cfg` so the CLI lives in
    perfect symmetry with the main training entry point. Imported lazily
    so the test suite can patch out the (heavy) construction path.

    Returns
    -------
    nn.Module
        A :class:`usam._train_helpers.USAMTrainModel`.
    """
    from usam.train import build_model_from_cfg

    return build_model_from_cfg(model_cfg)


def _count_params(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _resolve_device(arg: str) -> torch.device:
    """Resolve the ``--device`` flag into a concrete :class:`torch.device`."""
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# LIBERO dataset adapter
# ---------------------------------------------------------------------------
class LiberoTrajectoryDataset(Dataset):
    """Minimal LIBERO trajectory adapter.

    Walks ``libero_data`` for ``*.hdf5`` files and exposes them as a flat
    list of ``(file, demo_key, t)`` samples. If no HDF5 files are found
    (e.g. on a developer box without the LIBERO benchmark downloaded),
    the dataset falls back to a small synthetic-trajectory mode so the
    CLI plumbing remains testable. The fallback flag is exposed as
    :attr:`is_synthetic` so the caller can log it.

    The ``__getitem__`` contract returns a dict with keys:

    * ``rgb``    — ``[C=3, H, W]`` float32 image tensor.
    * ``proprio`` — ``[proprio_dim]`` float32 proprioception vector.
    * ``action`` — ``[action_dim]`` float32 action target.

    Production callers will swap in the real ``libero`` benchmark API
    once it lands in the qwen3vl env; until then this is the "real
    interface, synthetic data" path documented in the module docstring.
    """

    def __init__(
        self,
        libero_data: Path,
        suite: str = "libero_10",
        rgb_hw: tuple[int, int] = (128, 128),
        proprio_dim: int = 9,
        action_dim: int = 7,
        synthetic_length: int = 64,
    ) -> None:
        super().__init__()
        assert isinstance(libero_data, Path)
        assert proprio_dim > 0
        assert action_dim > 0
        assert synthetic_length > 0

        self.libero_data = libero_data
        self.suite = suite
        self.rgb_hw = rgb_hw
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)

        self.hdf5_files: List[Path] = []
        if libero_data.exists():
            # LIBERO suite subdirs are named like ``libero_10`` so we
            # search the requested suite first and fall back to the root.
            search_roots = [libero_data / suite, libero_data]
            seen: set[Path] = set()
            for root in search_roots:
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*.hdf5")):
                    if p in seen:
                        continue
                    seen.add(p)
                    self.hdf5_files.append(p)

        self.is_synthetic = len(self.hdf5_files) == 0
        # Sample index. In real mode we'd enumerate (file, demo, t) tuples;
        # for the stub we just expose ``synthetic_length`` samples that
        # round-robin through the discovered files (or produce synthetic
        # tensors when no files exist).
        self._length = int(synthetic_length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert 0 <= idx < self._length, f"idx {idx} out of range [0, {self._length})"
        if self.is_synthetic:
            return self._synthetic_sample(idx)
        return self._hdf5_sample(idx)

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------
    def _synthetic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fabricate a single synthetic trajectory step.

        Deterministic per ``idx`` so unit tests can reproduce shapes.
        """
        g = torch.Generator().manual_seed(int(idx))
        h, w = self.rgb_hw
        rgb = torch.rand((3, h, w), generator=g, dtype=torch.float32)
        proprio = torch.randn((self.proprio_dim,), generator=g, dtype=torch.float32)
        action = torch.randn((self.action_dim,), generator=g, dtype=torch.float32)
        return {"rgb": rgb, "proprio": proprio, "action": action}

    # ------------------------------------------------------------------
    # Real HDF5 path
    # ------------------------------------------------------------------
    def _hdf5_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Read a single step from one of the discovered HDF5 files.

        LIBERO HDF5 layout (see ``robosuite``-style demos):

        * ``data/<demo_key>/obs/agentview_rgb`` — ``[T, H, W, 3]`` uint8
        * ``data/<demo_key>/obs/robot0_proprio``— ``[T, proprio_dim]`` float
        * ``data/<demo_key>/actions``           — ``[T, action_dim]`` float

        For robustness we accept either ``obs/`` or top-level keys with
        common LIBERO names; missing fields fall back to synthetic.
        """
        # h5py is in requirements.txt; lazy import keeps the module
        # importable on environments where h5py is absent.
        try:
            import h5py
        except ImportError:  # pragma: no cover - h5py is in requirements
            return self._synthetic_sample(idx)

        path = self.hdf5_files[idx % len(self.hdf5_files)]
        try:
            with h5py.File(path, "r") as f:
                # Pick the first demo group inside ``data/`` if present.
                if "data" in f:
                    group = f["data"]
                    keys = list(group.keys())
                    if not keys:
                        return self._synthetic_sample(idx)
                    demo = group[keys[0]]
                else:
                    demo = f

                rgb = self._extract(demo, ("obs/agentview_rgb", "agentview_rgb", "rgb"))
                proprio = self._extract(demo, ("obs/robot0_proprio", "robot0_proprio", "proprio"))
                action = self._extract(demo, ("actions", "action"))

                # If any field is missing, fall back to synthetic.
                if rgb is None or proprio is None or action is None:
                    return self._synthetic_sample(idx)

                # Pick a random step within the trajectory.
                t = idx % int(rgb.shape[0])
                rgb_step = torch.as_tensor(rgb[t], dtype=torch.float32)
                if rgb_step.ndim == 3 and rgb_step.shape[-1] in (1, 3, 4):
                    # HWC -> CHW
                    rgb_step = rgb_step.permute(2, 0, 1)
                proprio_step = torch.as_tensor(proprio[t], dtype=torch.float32).flatten()
                action_step = torch.as_tensor(action[t], dtype=torch.float32).flatten()

                # Truncate / pad to the configured dims so collate doesn't fail.
                proprio_step = self._fit_1d(proprio_step, self.proprio_dim)
                action_step = self._fit_1d(action_step, self.action_dim)
                return {"rgb": rgb_step, "proprio": proprio_step, "action": action_step}
        except (OSError, KeyError, ValueError):
            # Corrupt HDF5 or unexpected schema — fall back gracefully.
            return self._synthetic_sample(idx)

    @staticmethod
    def _extract(group: Any, candidate_keys: tuple[str, ...]) -> Optional[Any]:
        """Return the first key from ``candidate_keys`` that resolves inside ``group``."""
        for key in candidate_keys:
            try:
                # h5py supports slash-separated paths.
                node = group[key]
                return node
            except (KeyError, TypeError):
                continue
        return None

    @staticmethod
    def _fit_1d(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Truncate / zero-pad a 1-D tensor so it matches ``target_dim``."""
        cur = int(tensor.shape[0])
        if cur == target_dim:
            return tensor
        if cur > target_dim:
            return tensor[:target_dim]
        out = torch.zeros((target_dim,), dtype=tensor.dtype)
        out[:cur] = tensor
        return out


def build_libero_dataloader(args: FinetuneArgs) -> DataLoader:
    """Build a :class:`DataLoader` from a :class:`LiberoTrajectoryDataset`."""
    dataset = LiberoTrajectoryDataset(args.libero_data, suite=args.suite)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # smoke / dev environments — keep it single-process.
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose lr will be scheduled.
    max_steps : int
        Total training steps.
    warmup_steps : int, optional
        Number of warmup steps (default 100).
    min_lr_ratio : float, optional
        Floor for the cosine, expressed as a fraction of the base lr
        (default 0.01 → lr decays to 1% of base).
    """
    assert max_steps > 0
    assert warmup_steps >= 0
    assert 0.0 <= min_lr_ratio <= 1.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# WandB (optional)
# ---------------------------------------------------------------------------
def _maybe_init_wandb(args: FinetuneArgs) -> Any:
    """Initialize wandb if ``WANDB_API_KEY`` is set, otherwise return ``None``.

    Soft-fails on any wandb import / init error so that the CLI works on
    machines without internet access.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "usam-finetune"),
            name=f"libero-{args.suite}",
            config={
                "max_steps": args.max_steps,
                "suite": args.suite,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "base_ckpt": str(args.base_ckpt),
            },
            reinit=True,
        )
        return run
    except Exception as e:  # pragma: no cover - network / install paths
        _loguru_logger.warning("wandb init failed (continuing without wandb): %s", e)
        return None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _setup_logger(output_dir: Path) -> None:
    """Attach a file sink to the loguru logger at ``output_dir/finetune.log``.

    Uses loguru's ``add`` API when available; falls back silently when the
    bare-stdlib ``logging`` shim is active (see the soft import at top).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "finetune.log"
    add_fn = getattr(_loguru_logger, "add", None)
    if add_fn is not None:
        try:
            add_fn(str(log_path), level="INFO", enqueue=False)
        except Exception:  # pragma: no cover - re-entrant add
            pass
    else:  # pragma: no cover - logging fallback
        import logging

        handler = logging.FileHandler(str(log_path))
        handler.setLevel(logging.INFO)
        _loguru_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Training loop (stub body)
# ---------------------------------------------------------------------------
def run_finetune(args: FinetuneArgs) -> Path:
    """Run the LIBERO finetuning pipeline end-to-end.

    The training-loop body is a stub: it consumes a real batch from the
    dataloader on each step (so the dataloader path is genuinely exercised)
    but does not run forward / loss / backward. Future contributors fill
    in the stubbed block — see the `FUTURE-FILL-IN` marker inside this
    function — without touching the surrounding plumbing.
    """
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logger(args.output_dir)
    _loguru_logger.info(
        "USAM LIBERO finetune | base_ckpt=%s | output_dir=%s | suite=%s | max_steps=%d",
        args.base_ckpt, args.output_dir, args.suite, args.max_steps,
    )

    # ------------------------------------------------------------------
    # Optional wandb
    # ------------------------------------------------------------------
    wandb_run = _maybe_init_wandb(args)
    if wandb_run is None:
        _loguru_logger.info("wandb not initialised (WANDB_API_KEY unset or wandb missing).")

    # ------------------------------------------------------------------
    # Config + model
    # ------------------------------------------------------------------
    cfg = load_model_config(args.model_config)
    model = build_model_from_config(cfg)

    # ------------------------------------------------------------------
    # Base checkpoint load
    # ------------------------------------------------------------------
    if args.base_ckpt.exists():
        # weights_only=False so we can read our own checkpoints which carry
        # string metadata (run_id, config, etc.) alongside tensors.
        ckpt = torch.load(str(args.base_ckpt), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            try:
                # ``strict=False`` — smoke checkpoints may differ slightly
                # from the finetune target. We log the deltas for transparency.
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    _loguru_logger.info(
                        "Loaded base ckpt with %d missing keys (first 8): %s",
                        len(missing), list(missing)[:8],
                    )
                if unexpected:
                    _loguru_logger.info(
                        "Loaded base ckpt with %d unexpected keys (first 8): %s",
                        len(unexpected), list(unexpected)[:8],
                    )
            except (RuntimeError, TypeError) as e:
                _loguru_logger.warning(
                    "load_state_dict failed (%s); continuing with random init.", e,
                )
        else:
            _loguru_logger.warning(
                "Base checkpoint at %s is not a state_dict; skipping load.", args.base_ckpt,
            )
    else:
        _loguru_logger.warning(
            "Base checkpoint %s does not exist; using random init.", args.base_ckpt,
        )

    # ------------------------------------------------------------------
    # Device + param count
    # ------------------------------------------------------------------
    device = _resolve_device(args.device)
    model = model.to(device=device)
    total, trainable = _count_params(model)
    _loguru_logger.info(
        "Model on %s | total params: %d | trainable params: %d", device, total, trainable,
    )

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    dataloader = build_libero_dataloader(args)
    if hasattr(dataloader.dataset, "is_synthetic") and dataloader.dataset.is_synthetic:
        _loguru_logger.warning(
            "LIBERO data path %s has no HDF5 files — using synthetic samples. "
            "Real LIBERO trajectories needed before launching production runs.",
            args.libero_data,
        )

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        # No trainable params (e.g. stubbed test model that's frozen).
        # AdamW won't accept an empty list. Fall back to a dummy parameter
        # so the optimizer / scheduler plumbing remains exercised.
        dummy = nn.Parameter(torch.zeros(1, device=device))
        params = [dummy]
        _loguru_logger.warning(
            "Model has no trainable parameters; using a dummy parameter so the "
            "optimizer + scheduler plumbing remains exercised."
        )
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = make_cosine_warmup_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_steps=100,
        min_lr_ratio=0.01,
    )

    # ------------------------------------------------------------------
    # Training loop (STUB BODY — replace forward + loss + backward with
    # the real implementation once the production Player is wired up).
    # ------------------------------------------------------------------
    model.train()
    data_iter = iter(dataloader)
    step = 0
    for step in range(args.max_steps):
        # Pull a real batch — exercises the dataloader path even though
        # the forward/loss/backward block below is stubbed.
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # === FUTURE-FILL-IN: real forward + loss + backward ===
        # Drop-in target signature:
        #     preds = model(batch)
        #     loss = loss_fn(preds, batch)
        #     loss.backward(); optimizer.step(); scheduler.step()
        # The stub below just steps the LR schedule so its state-dict
        # contains real numbers at save time, and logs progress.
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % max(1, args.log_every) == 0:
            _loguru_logger.info(
                "[step %d/%d] STUB step (replace with real forward+loss+backward) "
                "lr=%.3e batch_keys=%s",
                step, args.max_steps,
                scheduler.get_last_lr()[0],
                sorted(batch.keys()) if isinstance(batch, dict) else type(batch).__name__,
            )

        # Bound the stub at 10 steps so the CLI exits cleanly even when
        # callers pass a large --max-steps. Real training will remove this
        # cap as part of the FUTURE-FILL-IN block above.
        if step >= 9:
            _loguru_logger.info("STUB cap reached (10 steps); exiting loop.")
            break

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    ckpt_path = args.output_dir / "finetune_ckpt.pt"
    payload = {
        "model": model.state_dict(),
        "step": int(step),
        "config": yaml.safe_dump(cfg, sort_keys=False),
        "suite": args.suite,
        "base_ckpt": str(args.base_ckpt),
    }
    torch.save(payload, str(ckpt_path))
    _loguru_logger.info("Saved finetune checkpoint to %s", ckpt_path)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:  # pragma: no cover
            pass

    return ckpt_path


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """CLI wrapper used by ``python -m usam.finetune_libero``.

    Returns ``0`` on success.
    """
    args = parse_args(argv)
    run_finetune(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
