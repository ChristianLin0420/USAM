# SPDX-License-Identifier: MIT
"""Tri-modal DINO Tower.

One DINOv3 backbone, three input adapters (RGB / depth / flow), three
modality-aware LoRA paths inside the attention blocks.

* The **RGB patch_embed** is the original DINOv3 conv2d ``[D, 3, P, P]``.
* The **depth patch_embed** is initialized from
  ``mean(rgb_patch.weight, dim=1, keepdim=True)`` and accepts a single
  channel.
* The **flow patch_embed** is initialized from ``rgb_patch.weight[:, :2]``
  and accepts two channels.
* All three patch embeddings remain trainable.
* The backbone (everything except the patch embeddings and LoRA params) is
  frozen.

The encoder targets DINOv3 ViT (HuggingFace ``transformers`` style) but
also accepts any backbone that exposes ``embeddings.patch_embeddings.projection``
(an ``nn.Conv2d``) and a forward returning a ``last_hidden_state`` tensor of
shape ``[B, N_tokens, D]`` where the prefix tokens are the [CLS] token plus
optional register tokens.

For unit testing without DINOv3 weights, see :class:`MiniDinoBackbone` —
a tiny randomly-initialised stand-in that mirrors the relevant interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

import torch
from torch import Tensor, nn

from usam.adapters.lora import LoRALinear, apply_lora


Modality = Literal["rgb", "depth", "flow"]
_MODALITIES: tuple[str, ...] = ("rgb", "depth", "flow")
_CHANNELS: dict[str, int] = {"rgb": 3, "depth": 1, "flow": 2}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TriDinoConfig:
    """Configuration for :class:`TriDINOTower`.

    Parameters
    ----------
    dinov3_ckpt : str
        Local path to a HuggingFace-style DINOv3 checkpoint directory, or
        the empty string ``""`` to indicate a random stand-in (mainly used
        in tests).
    dinov3_arch : str
        Architecture tag, e.g. ``"vit_b_14"`` or ``"vit_l_14"``. Used only
        for logging when a real backbone is loaded; the actual config comes
        from the checkpoint itself.
    image_size : int
        Square input resolution. Patch grid is ``image_size // patch_size``.
    patch_size : int
        Patch size of the underlying ViT (14 for DINOv3).
    embed_dim : int
        Hidden dimension of the ViT (768 for B/14, 1024 for L/14).
    num_register_tokens : int
        Number of register tokens DINOv3 prepends after [CLS].
    lora_rank : int
        Rank of the LoRA adapters wrapping Q/K/V. ``0`` disables LoRA.
    lora_target_names : tuple[str, ...]
        Module-name suffixes to wrap. Defaults match HF DINOv3 attention
        modules.
    freeze_rgb_patch_embed : bool
        Keep the RGB patch_embed in eval and ``requires_grad=False``. By
        default we keep it trainable to allow distribution shift handling.
    """

    dinov3_ckpt: str = ""
    dinov3_arch: str = "vit_b_14"
    # 27 * 14 = 378; preserves the canonical 27x27=729 patch grid that the
    # cache + plan reference. The plan colloquially says "384²"; YAMLs
    # already override to 378. Default matches the binding contract.
    image_size: int = 378
    patch_size: int = 14
    embed_dim: int = 768
    num_register_tokens: int = 4
    lora_rank: int = 8
    lora_target_names: tuple[str, ...] = ("query", "key", "value")
    freeze_rgb_patch_embed: bool = False
    # Optional override for tests / alternate backbones.
    backbone_override: nn.Module | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Tiny stand-in backbone for tests
# ---------------------------------------------------------------------------
class _MiniAttention(nn.Module):
    """Minimal multi-head self-attention with named query/key/value linears.

    The submodule names ``query``, ``key``, ``value`` are intentionally chosen
    so that :func:`apply_lora` (with default ``target_module_names``) wraps
    them.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(out)


class _MiniBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _MiniAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MiniPatchEmbeddings(nn.Module):
    """Wraps a Conv2d under the attribute name ``projection`` to match HF DINOv3."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class _MiniEmbeddings(nn.Module):
    """Embeddings module exposing ``patch_embeddings.projection`` and prefix tokens."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        num_register_tokens: int,
        num_patches: int,
    ) -> None:
        super().__init__()
        self.patch_embeddings = _MiniPatchEmbeddings(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 1 + num_register_tokens + num_patches, embed_dim)
        )
        self.num_register_tokens = num_register_tokens

    def forward(self, pixel_values: Tensor) -> Tensor:
        b = pixel_values.shape[0]
        x = self.patch_embeddings(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        prefix = [self.cls_token.expand(b, -1, -1)]
        if self.register_tokens is not None:
            prefix.append(self.register_tokens.expand(b, -1, -1))
        x = torch.cat(prefix + [x], dim=1)
        return x + self.position_embeddings


@dataclass
class _MiniBackboneConfig:
    image_size: int
    patch_size: int
    hidden_size: int
    num_register_tokens: int


class MiniDinoBackbone(nn.Module):
    """A tiny randomly-initialized DINOv3-shaped backbone used by tests.

    It exposes the surface area :class:`TriDINOTower` relies on:

    * ``embeddings.patch_embeddings.projection`` — Conv2d patch embed.
    * ``encoder.layer[i].attention.{query,key,value}`` — Linear projections
      that LoRA can wrap.
    * ``forward(pixel_values=...)`` returning an object with
      ``last_hidden_state`` of shape ``[B, 1+R+N, D]``.

    The output is decoupled from any pretraining; this class is only for
    unit tests and should never be used at training time.
    """

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        hidden_size: int = 768,
        num_register_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        self.config = _MiniBackboneConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
        )
        grid = image_size // patch_size
        self.embeddings = _MiniEmbeddings(
            in_channels=3,
            embed_dim=hidden_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            num_patches=grid * grid,
        )

        class _Encoder(nn.Module):
            def __init__(self, dim: int, num_layers: int, num_heads: int) -> None:
                super().__init__()
                self.layer = nn.ModuleList(
                    [_MiniBlock(dim, num_heads=num_heads) for _ in range(num_layers)]
                )

            def forward(self, x: Tensor) -> Tensor:
                for blk in self.layer:
                    x = blk(x)
                return x

        self.encoder = _Encoder(hidden_size, num_layers=num_layers, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: Tensor) -> Mapping[str, Tensor]:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.layernorm(x)
        return {"last_hidden_state": x}


# ---------------------------------------------------------------------------
# The Tri-DINO Tower
# ---------------------------------------------------------------------------
class TriDINOTower(nn.Module):
    """Tri-modal DINOv3 encoder with shared backbone + per-modality adapters.

    Parameters
    ----------
    config : TriDinoConfig
        See :class:`TriDinoConfig`.

    Notes on training freeze policy
    -------------------------------
    * Backbone parameters (everything inside ``backbone`` other than
      patch_embeddings and LoRA paths) → ``requires_grad=False``.
    * RGB / depth / flow patch_embed Conv2d weights → trainable, unless
      ``config.freeze_rgb_patch_embed`` is True (then RGB stays frozen).
    * LoRA A/B parameters → trainable.

    Forward contracts
    -----------------
    ``forward(x, modality)`` returns ``[B, N_tokens, D]`` where
    ``N_tokens = 1 + num_register_tokens + (image_size / patch_size)**2``
    and ``D = embed_dim``.

    ``extract_features(x, modality, n_keep_tokens=64)`` returns a fp16
    tensor of shape ``[B, n_keep_tokens + 1, D]`` containing the [CLS]
    token followed by ``n_keep_tokens`` patch tokens (register tokens are
    dropped). The +1 is the [CLS] / pooled prefix token.
    """

    def __init__(self, config: TriDinoConfig) -> None:
        super().__init__()
        self.config = config

        backbone = self._build_backbone(config)
        self.backbone = backbone

        # Discover the original RGB patch_embed conv. We expect the standard
        # HF DINOv3 path; the mini backbone uses the same path.
        emb_module = self.backbone.embeddings
        assert hasattr(emb_module, "patch_embeddings") and hasattr(
            emb_module.patch_embeddings, "projection"
        ), "Backbone is missing embeddings.patch_embeddings.projection (Conv2d)."

        self.rgb_patch: nn.Conv2d = emb_module.patch_embeddings.projection
        embed_dim = self.rgb_patch.out_channels
        patch_size = self.rgb_patch.kernel_size[0]

        # Build depth/flow patch embeddings with surgical weight init.
        self.depth_patch = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flow_patch = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        with torch.no_grad():
            # depth: average over input channels -> [D, 1, P, P]
            self.depth_patch.weight.copy_(
                self.rgb_patch.weight.mean(dim=1, keepdim=True)
            )
            if self.rgb_patch.bias is not None and self.depth_patch.bias is not None:
                self.depth_patch.bias.copy_(self.rgb_patch.bias)
            # flow: take first two input channels -> [D, 2, P, P]
            self.flow_patch.weight.copy_(self.rgb_patch.weight[:, :2].clone())
            if self.rgb_patch.bias is not None and self.flow_patch.bias is not None:
                self.flow_patch.bias.copy_(self.rgb_patch.bias)

        # Freeze backbone first; then re-enable patch_embed + LoRA params.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Patch embedding adapters are trainable.
        if not config.freeze_rgb_patch_embed:
            for p in self.rgb_patch.parameters():
                p.requires_grad_(True)
        for p in self.depth_patch.parameters():
            p.requires_grad_(True)
        for p in self.flow_patch.parameters():
            p.requires_grad_(True)

        # Apply LoRA to every Q/K/V linear in the backbone.
        if config.lora_rank > 0:
            self.lora_modules = apply_lora(
                self.backbone,
                r=config.lora_rank,
                target_module_names=tuple(config.lora_target_names),
                modality_ids=list(_MODALITIES),
            )
        else:
            self.lora_modules = {m: [] for m in _MODALITIES}

        # Cache shape metadata that consumers (config writers, tests) want.
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = config.image_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        grid = self.image_size // self.patch_size
        self.num_patches = grid * grid
        self.num_tokens = 1 + self.num_register_tokens + self.num_patches

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_backbone(config: TriDinoConfig) -> nn.Module:
        if config.backbone_override is not None:
            return config.backbone_override
        if not config.dinov3_ckpt:
            # Default to the mini stand-in. Used only when no checkpoint is
            # supplied (e.g. unit tests via the override path).
            return MiniDinoBackbone(
                image_size=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.embed_dim,
                num_register_tokens=config.num_register_tokens,
            )
        # Real DINOv3 path. We avoid importing transformers at module top
        # level so the unit tests can run without network access.
        from transformers import AutoModel  # type: ignore

        backbone = AutoModel.from_pretrained(config.dinov3_ckpt)
        return backbone

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _patch_embed(self, x: Tensor, modality: str) -> Tensor:
        """Per-modality patch embed, reshaped to ``[B, N_patches, D]``."""
        if modality == "rgb":
            feats = self.rgb_patch(x)
        elif modality == "depth":
            feats = self.depth_patch(x)
        elif modality == "flow":
            feats = self.flow_patch(x)
        else:
            raise ValueError(f"unknown modality {modality!r}")
        return feats.flatten(2).transpose(1, 2).contiguous()

    def _attach_modality_to_lora(self, modality: str) -> None:
        """Re-route ``LoRALinear.forward`` to use the requested modality.

        We monkey-patch each :class:`LoRALinear` instance's ``forward`` to
        bind the modality id. This keeps the underlying backbone's call
        sites unchanged (they still call ``linear(x)``) while letting LoRA
        pick the correct path. A previous binding is overwritten on each
        call.
        """
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module._usam_active_modality = modality  # type: ignore[attr-defined]

    @staticmethod
    def _patched_lora_forward(self: LoRALinear, x: Tensor) -> Tensor:  # noqa: D401
        modality = getattr(self, "_usam_active_modality", None)
        return LoRALinear.forward(self, x, modality_id=modality)

    def _ensure_lora_routing(self) -> None:
        """Install the modality-routing forward on each LoRALinear once."""
        if getattr(self, "_lora_routed", False):
            return
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.forward = self._patched_lora_forward.__get__(  # type: ignore[method-assign]
                    module, LoRALinear
                )
        self._lora_routed = True

    def _run_backbone(self, prefix_tokens: Tensor, patch_tokens: Tensor) -> Tensor:
        """Run the frozen backbone on already-embedded tokens.

        We replace the backbone's patch_embed step (which expects raw
        pixels) by feeding the [CLS] + register + patch token sequence
        directly into the encoder + final layernorm. This lets us use a
        per-modality patch embedding while keeping the rest of the
        backbone bit-identical to the upstream model.
        """
        emb = self.backbone.embeddings
        b = patch_tokens.shape[0]
        cls = emb.cls_token.expand(b, -1, -1)
        prefix = [cls]
        regs = getattr(emb, "register_tokens", None)
        if regs is not None:
            prefix.append(regs.expand(b, -1, -1))
        x = torch.cat(prefix + [patch_tokens], dim=1)
        # Add absolute position embeddings if present.
        pe = getattr(emb, "position_embeddings", None)
        if isinstance(pe, nn.Parameter):
            assert pe.shape[1] == x.shape[1], (
                f"position_embeddings length {pe.shape[1]} != token count {x.shape[1]}"
            )
            x = x + pe
        # Encoder.
        encoder = self.backbone.encoder
        out = encoder(x)
        if isinstance(out, Tensor):
            x = out
        elif isinstance(out, dict):
            x = out.get("last_hidden_state", out.get("hidden_states", out.get("hidden_state", None)))
            assert x is not None, f"encoder dict did not contain hidden states: keys={list(out.keys())}"
        elif isinstance(out, tuple):
            x = out[0]
        else:
            # transformers ModelOutput etc.; fall back to attribute access.
            x = getattr(out, "last_hidden_state", None) or getattr(out, "hidden_states", None)
            assert isinstance(x, Tensor), f"unexpected encoder output type {type(out)}"
        ln = getattr(self.backbone, "layernorm", None) or getattr(self.backbone, "norm", None)
        if isinstance(ln, nn.Module):
            x = ln(x)
        return x

    def forward(self, x: Tensor, modality: Modality) -> Tensor:
        """Run the encoder for one modality.

        Parameters
        ----------
        x : Tensor
            Pixel-style input. RGB ``[B, 3, H, W]``, depth ``[B, 1, H, W]``,
            flow ``[B, 2, H, W]``.
        modality : {"rgb", "depth", "flow"}
            Which adapter + LoRA path to use.

        Returns
        -------
        Tensor
            ``[B, num_tokens, embed_dim]``.
        """
        assert x.dim() == 4, f"input must be [B, C, H, W], got {tuple(x.shape)}"
        expected_c = _CHANNELS[modality]
        assert x.shape[1] == expected_c, (
            f"channel mismatch for modality={modality}: expected {expected_c}, got {x.shape[1]}"
        )

        self._ensure_lora_routing()
        self._attach_modality_to_lora(modality)

        patch_tokens = self._patch_embed(x, modality)
        # The [CLS] / register prefix is added inside _run_backbone.
        return self._run_backbone(prefix_tokens=None, patch_tokens=patch_tokens)

    @torch.no_grad()
    def extract_features(
        self,
        x: Tensor,
        modality: Modality,
        n_keep_tokens: int = 64,
    ) -> Tensor:
        """Cache-extraction helper: returns fp16 ``[B, n_keep_tokens+1, D]``.

        The output is ``[CLS] | first n_keep_tokens patch tokens``. Register
        tokens are dropped. The output is ``torch.float16`` so it can be
        written directly to the safetensors feature cache.
        """
        assert n_keep_tokens > 0, "n_keep_tokens must be positive"
        out = self.forward(x, modality)
        # Layout: [CLS] | register_tokens | patch_tokens
        cls = out[:, :1]
        patch_start = 1 + self.num_register_tokens
        patches = out[:, patch_start : patch_start + n_keep_tokens]
        cached = torch.cat([cls, patches], dim=1)
        return cached.to(torch.float16)

    # ------------------------------------------------------------------
    # Parameter-group helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only the parameters with ``requires_grad=True``."""
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        """Yield LoRA A/B parameters across all modality paths."""
        seen: set[int] = set()
        for mod_list in self.lora_modules.values():
            for wrapper in mod_list:
                for p in wrapper.lora_parameters():
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    yield p


__all__ = ["TriDINOTower", "TriDinoConfig", "MiniDinoBackbone"]
