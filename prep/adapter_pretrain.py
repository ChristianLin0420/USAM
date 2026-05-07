# SPDX-License-Identifier: MIT
"""Phase A.5 — Tri-DINO adapter pretraining.

Goal
----
Warm-start the new ``depth_patch`` and ``flow_patch`` Conv2d weights and the
Q/K/V LoRA paths so that the depth and flow encoders produce features
roughly aligned with the RGB-DINO latent space. This is run **once**
before the Phase B pretraining; the resulting checkpoint is loaded by the
training entry point and frozen further or fine-tuned at low LR.

Pipeline summary
----------------
1.  Load the Tri-DINO encoder from a config.
2.  Stream RGB + depth + flow frames from a small subset (≈ 5 M frames) of
    the converted USAM-LeRobot Tier-1 corpus.
3.  Compute three losses, all with the **frozen RGB encoder as teacher**:
      * **Patch-level InfoNCE** between RGB and the depth/flow tokens
        (excluding [CLS] + register tokens).
      * **CLS MSE** between projected depth-CLS / flow-CLS and the
        RGB-CLS.
4.  Backprop only into the depth/flow patch_embed + LoRA params (the
    backbone stays frozen by virtue of :class:`TriDINOTower`'s freeze
    policy).
5.  Save ``checkpoints/tri_dino_adapter.pt`` with the trainable state dict.

This module is intentionally self-contained: it does **not** import the
data-engineer's dataloader (which doesn't exist yet). It instead consumes
a generic ``Iterable`` of ``(rgb, depth, flow)`` tensor triples produced
by a callable the caller passes in — at integration time the wrapper
script wires up the real loader.

The script can also be invoked as ``python -m prep.adapter_pretrain
--config configs/train/adapter_pretrain.yaml`` once that config and the
real loader land. Until then a ``--dry-run`` flag exercises a single
random-input step to verify wiring.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from usam.encoders.tri_dino import TriDINOTower, TriDinoConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AdapterPretrainConfig:
    """Hyper-parameters and IO paths for adapter pretraining."""

    encoder: TriDinoConfig = field(default_factory=TriDinoConfig)
    out_path: Path = Path("checkpoints/tri_dino_adapter.pt")
    batch_size: int = 32
    max_steps: int = 50_000
    lr: float = 5e-4
    weight_decay: float = 0.0
    info_nce_temp: float = 0.07
    cls_mse_weight: float = 1.0
    info_nce_weight: float = 1.0
    log_every: int = 50
    save_every: int = 5_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(yaml_path: str | Path) -> AdapterPretrainConfig:
    """Read a YAML config into an :class:`AdapterPretrainConfig` instance."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    enc_raw = raw.pop("encoder", {})
    enc = TriDinoConfig(**enc_raw)
    return AdapterPretrainConfig(encoder=enc, **raw)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def patch_info_nce(
    student_tokens: Tensor,
    teacher_tokens: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric patch-level InfoNCE.

    Each token at the same ``(b, n)`` position is the positive pair; all
    other tokens (across batch and patch index) are negatives.

    Parameters
    ----------
    student_tokens, teacher_tokens : Tensor
        Shape ``[B, N, D]``. The teacher is detached internally.
    temperature : float
        Softmax temperature.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    assert student_tokens.shape == teacher_tokens.shape, (
        f"shape mismatch: student {student_tokens.shape} vs teacher {teacher_tokens.shape}"
    )
    b, n, d = student_tokens.shape
    s = F.normalize(student_tokens.reshape(b * n, d), dim=-1)
    t = F.normalize(teacher_tokens.reshape(b * n, d), dim=-1).detach()
    logits = (s @ t.t()) / temperature
    labels = torch.arange(b * n, device=s.device)
    loss_st = F.cross_entropy(logits, labels)
    loss_ts = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_st + loss_ts)


def cls_mse(student_cls: Tensor, teacher_cls: Tensor) -> Tensor:
    """L2 between the student CLS and the (detached) teacher CLS."""
    assert student_cls.shape == teacher_cls.shape
    return F.mse_loss(student_cls, teacher_cls.detach())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def _trainable_params(model: TriDINOTower) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_step(
    encoder: TriDINOTower,
    rgb: Tensor,
    depth: Tensor,
    flow: Tensor,
    cfg: AdapterPretrainConfig,
) -> dict[str, Tensor]:
    """Single step. Returns dict of named loss components (all scalars)."""
    rgb_out = encoder(rgb, "rgb")
    depth_out = encoder(depth, "depth")
    flow_out = encoder(flow, "flow")

    n_reg = encoder.num_register_tokens
    rgb_cls = rgb_out[:, 0]
    depth_cls = depth_out[:, 0]
    flow_cls = flow_out[:, 0]
    rgb_patches = rgb_out[:, 1 + n_reg :]
    depth_patches = depth_out[:, 1 + n_reg :]
    flow_patches = flow_out[:, 1 + n_reg :]

    l_nce_d = patch_info_nce(depth_patches, rgb_patches, cfg.info_nce_temp)
    l_nce_f = patch_info_nce(flow_patches, rgb_patches, cfg.info_nce_temp)
    l_mse_d = cls_mse(depth_cls, rgb_cls)
    l_mse_f = cls_mse(flow_cls, rgb_cls)

    total = (
        cfg.info_nce_weight * (l_nce_d + l_nce_f)
        + cfg.cls_mse_weight * (l_mse_d + l_mse_f)
    )
    return {
        "loss": total,
        "info_nce_depth": l_nce_d.detach(),
        "info_nce_flow": l_nce_f.detach(),
        "cls_mse_depth": l_mse_d.detach(),
        "cls_mse_flow": l_mse_f.detach(),
    }


def run(
    cfg: AdapterPretrainConfig,
    batch_iter: Iterable[tuple[Tensor, Tensor, Tensor]],
) -> Path:
    """Run adapter pretraining. ``batch_iter`` yields (rgb, depth, flow).

    Parameters
    ----------
    cfg : AdapterPretrainConfig
        Hyper-parameters.
    batch_iter : Iterable[tuple[Tensor, Tensor, Tensor]]
        Each element is ``(rgb [B,3,H,W], depth [B,1,H,W], flow [B,2,H,W])``
        already on the correct device.

    Returns
    -------
    Path
        Where the final checkpoint was written.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    encoder = TriDINOTower(cfg.encoder).to(device)
    encoder.train()

    optimizer = torch.optim.AdamW(
        _trainable_params(encoder),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    iterator: Iterator[tuple[Tensor, Tensor, Tensor]] = iter(batch_iter)
    while step < cfg.max_steps:
        try:
            rgb, depth, flow = next(iterator)
        except StopIteration:
            iterator = iter(batch_iter)
            continue

        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)

        losses = train_step(encoder, rgb, depth, flow, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        if step % cfg.log_every == 0:
            log_str = " ".join(
                f"{k}={v.item():.4f}" for k, v in losses.items()
            )
            logger.info("[step %d] %s", step, log_str)

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            _save(encoder, cfg.out_path.with_name(cfg.out_path.stem + f"_step{step}.pt"))

        step += 1

    out = _save(encoder, cfg.out_path)
    logger.info("Saved final checkpoint to %s", out)
    return out


def _save(encoder: TriDINOTower, path: Path) -> Path:
    """Persist only the trainable parameters (patch_embed + LoRA)."""
    sd = encoder.state_dict()
    trainable_names = {n for n, p in encoder.named_parameters() if p.requires_grad}
    state = {n: sd[n].detach().cpu() for n in trainable_names if n in sd}
    torch.save({"state_dict": state}, path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dry_run(cfg: AdapterPretrainConfig) -> None:
    """Run a single step with random tensors to verify wiring."""
    cfg.max_steps = 1
    cfg.save_every = 0
    img = cfg.encoder.image_size
    bs = max(1, cfg.batch_size)

    def _gen() -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        while True:
            yield (
                torch.randn(bs, 3, img, img),
                torch.randn(bs, 1, img, img),
                torch.randn(bs, 2, img, img),
            )

    run(cfg, _gen())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--dry-run", action="store_true", help="Run one random step and exit")
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else AdapterPretrainConfig()
    if args.dry_run:
        _dry_run(cfg)
        return
    raise NotImplementedError(
        "Real-data training requires the data-engineer's loader. Pass a "
        "batch iterator to `run(...)` from your driver script, or call "
        "with --dry-run to exercise the wiring."
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
