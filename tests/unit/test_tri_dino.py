# SPDX-License-Identifier: MIT
"""Unit tests for the Tri-DINO Tower.

These tests use a tiny randomly-initialized stand-in
(:class:`MiniDinoBackbone`) as the DINOv3 backbone so they run without
network access or large checkpoints.
"""
from __future__ import annotations

import torch

from usam.adapters.lora import LoRALinear
from usam.encoders.tri_dino import (
    MiniDinoBackbone,
    TriDINOTower,
    TriDinoConfig,
)


def _make_tower(
    embed_dim: int = 768,
    # 27×14 = 378 — keeps the patch grid an exact integer for ViT-B/14.
    # The plan calls this "384²" colloquially, but the math works out to
    # a 27×27 = 729 patch grid, which requires image_size 378 with
    # patch_size 14.
    image_size: int = 378,
    patch_size: int = 14,
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
        dinov3_arch="vit_b_14",
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
    """ViT-B/14 at 384² → 729 patches + 1 [CLS] + 4 register = 734 tokens."""
    tower = _make_tower()
    img_size = tower.image_size
    bs = 2
    grid = img_size // tower.patch_size  # 27
    expected_tokens = 1 + tower.num_register_tokens + grid * grid
    assert expected_tokens == 1 + 4 + 729

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
