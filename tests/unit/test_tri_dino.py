# SPDX-License-Identifier: MIT
"""Unit tests for the Tri-DINO Tower.

These tests use a tiny randomly-initialized stand-in
(:class:`MiniDinoBackbone`) as the DINOv3 backbone so they run without
network access or large checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn

from usam.adapters.lora import LoRALinear
from usam.encoders.tri_dino import (
    MiniDinoBackbone,
    TriDINOTower,
    TriDinoConfig,
)


def _make_tower(
    embed_dim: int = 768,
    image_size: int = 448,
    patch_size: int = 16,
    num_register_tokens: int = 4,
    lora_rank: int = 8,
) -> TriDINOTower:
    backbone = MiniDinoBackbone(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=embed_dim,
        num_register_tokens=num_register_tokens,
        num_layers=2,
        num_heads=4,
    )
    cfg = TriDinoConfig(
        dinov3_arch="vit_b_16",
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_register_tokens=num_register_tokens,
        lora_rank=lora_rank,
        lora_target_names=("query", "key", "value"),
        backbone_override=backbone,
    )
    return TriDINOTower(cfg)


def test_forward_shape_all_modalities() -> None:
    """ViT-B/16 at 448² → 784 patches + 1 [CLS] + 4 register = 789 tokens."""
    tower = _make_tower()
    img_size = tower.image_size
    bs = 2
    grid = img_size // tower.patch_size  # 28
    expected_tokens = 1 + tower.num_register_tokens + grid * grid
    assert expected_tokens == 1 + 4 + 784

    rgb = torch.randn(bs, 3, img_size, img_size)
    depth = torch.randn(bs, 1, img_size, img_size)
    flow = torch.randn(bs, 2, img_size, img_size)

    out_rgb = tower(rgb, "rgb")
    out_depth = tower(depth, "depth")
    out_flow = tower(flow, "flow")

    assert out_rgb.shape == (bs, expected_tokens, tower.embed_dim)
    assert out_depth.shape == (bs, expected_tokens, tower.embed_dim)
    assert out_flow.shape == (bs, expected_tokens, tower.embed_dim)


def test_extract_features_dtype_and_shape() -> None:
    """`extract_features` returns fp16 [B, n_keep+1, D]."""
    tower = _make_tower()
    bs = 3
    n_keep = 64
    rgb = torch.randn(bs, 3, tower.image_size, tower.image_size)
    feats = tower.extract_features(rgb, "rgb", n_keep_tokens=n_keep)
    assert feats.shape == (bs, n_keep + 1, tower.embed_dim)
    assert feats.dtype == torch.float16


def test_reencode_bit_exactness() -> None:
    """Two forwards on identical input produce identical output (eval mode)."""
    tower = _make_tower()
    tower.eval()
    rgb = torch.randn(1, 3, tower.image_size, tower.image_size)
    with torch.no_grad():
        a = tower(rgb, "rgb")
        b = tower(rgb, "rgb")
    assert torch.equal(a, b)


def test_freeze_policy() -> None:
    """Backbone (non-patch_embed, non-LoRA) frozen; adapters + LoRA trainable."""
    tower = _make_tower()
    # Patch_embed weights -> trainable
    assert tower.depth_patch.weight.requires_grad
    assert tower.flow_patch.weight.requires_grad
    assert tower.rgb_patch.weight.requires_grad

    # All LoRA params -> trainable
    n_lora_params = 0
    for p in tower.lora_parameters():
        assert p.requires_grad
        n_lora_params += 1
    assert n_lora_params > 0, "expected at least one LoRA parameter group"

    # Encoder weights inside the backbone -> frozen
    for blk in tower.backbone.encoder.layer:
        # base linears inside LoRALinear are frozen
        for module in blk.attention.modules():
            if isinstance(module, LoRALinear):
                for p in module.base.parameters():
                    assert not p.requires_grad, (
                        "LoRA base linear must be frozen"
                    )
        # Non-attention encoder weights (mlp, layernorm) frozen
        for p in blk.mlp.parameters():
            assert not p.requires_grad
        for p in blk.norm1.parameters():
            assert not p.requires_grad
        for p in blk.norm2.parameters():
            assert not p.requires_grad


def test_depth_patch_init_from_rgb_mean() -> None:
    """depth_patch.weight == mean(rgb_patch.weight, dim=1, keepdim=True)."""
    tower = _make_tower()
    expected = tower.rgb_patch.weight.detach().mean(dim=1, keepdim=True)
    assert tower.depth_patch.weight.shape == expected.shape
    assert torch.equal(tower.depth_patch.weight.detach(), expected)


def test_flow_patch_init_from_rgb_first_two_channels() -> None:
    """flow_patch.weight == rgb_patch.weight[:, :2]."""
    tower = _make_tower()
    expected = tower.rgb_patch.weight.detach()[:, :2]
    assert tower.flow_patch.weight.shape == expected.shape
    assert torch.equal(tower.flow_patch.weight.detach(), expected)


def test_lora_routing_changes_output_when_b_nonzero() -> None:
    """Setting one modality's LoRA-B nonzero must change *that* modality's output only."""
    tower = _make_tower()
    tower.eval()
    img_size = tower.image_size
    rgb = torch.randn(1, 3, img_size, img_size)
    flow = torch.randn(1, 2, img_size, img_size)

    with torch.no_grad():
        baseline_rgb = tower(rgb, "rgb")
        baseline_flow = tower(flow, "flow")

    # Inflate the flow LoRA-B for the very first wrapper.
    first_wrapper = tower.lora_modules["flow"][0]
    with torch.no_grad():
        first_wrapper.lora_B["flow"].add_(0.5)

    with torch.no_grad():
        new_rgb = tower(rgb, "rgb")
        new_flow = tower(flow, "flow")

    # rgb path uses the rgb LoRA paths -> unchanged
    assert torch.allclose(baseline_rgb, new_rgb), "rgb output should be invariant"
    # flow path uses the flow LoRA paths -> changed
    assert not torch.allclose(baseline_flow, new_flow), "flow output should differ"


def test_tridino_with_dinov3_layout_backbone() -> None:
    """Verify TriDINOTower works with the real DINOv3 ViT layout
    (patch_embeddings IS Conv2d, layer at top level, q_proj/k_proj/v_proj,
    no absolute position_embeddings)."""
    embed_dim = 64
    image_size = 32
    patch_size = 16
    num_register_tokens = 2
    grid = image_size // patch_size  # 2
    num_patches = grid * grid  # 4
    seq_len = 1 + num_register_tokens + num_patches  # 7

    # Build a DINOv3-shaped mock: patch_embeddings is Conv2d directly,
    # layer is a ModuleList of attention blocks with q_proj/k_proj/v_proj,
    # no position_embeddings (RoPE-style).
    class _DinoV3Embeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Conv2d DIRECTLY under patch_embeddings, not nested .projection
            self.patch_embeddings = nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim)
            )
            # NO position_embeddings — RoPE style

    class _DinoV3Attention(nn.Module):
        """Attention with q_proj/k_proj/v_proj naming (DINOv3 convention)."""

        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.o_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class _DinoV3Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attention = _DinoV3Attention()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.attention(self.norm1(x))

    # `dataclass` field defaults can't capture closures, so bind to locals
    # via a captured-default closure trick on the class itself.
    _image_size = image_size
    _patch_size = patch_size
    _embed_dim = embed_dim
    _num_register_tokens = num_register_tokens

    class _Cfg:
        def __init__(self) -> None:
            self.image_size = _image_size
            self.patch_size = _patch_size
            self.hidden_size = _embed_dim
            self.num_register_tokens = _num_register_tokens

    class _DinoV3Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            self.embeddings = _DinoV3Embeddings()
            # `layer` AT TOP LEVEL (not under .encoder), matching DINOv3 ViT
            self.layer = nn.ModuleList([_DinoV3Block() for _ in range(2)])
            self.norm = nn.LayerNorm(embed_dim)
            # No layernorm — DINOv3 uses `norm`

    cfg = TriDinoConfig(
        dinov3_arch="vit_b_16",
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_register_tokens=num_register_tokens,
        lora_rank=4,
        lora_target_names=("q_proj", "k_proj", "v_proj"),
        backbone_override=_DinoV3Backbone(),
    )
    tower = TriDINOTower(cfg)
    rgb = torch.randn(2, 3, image_size, image_size)
    out = tower(rgb, "rgb")
    assert out.shape == (2, seq_len, embed_dim), out.shape
