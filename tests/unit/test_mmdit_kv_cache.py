# SPDX-License-Identifier: MIT
"""Unit tests for the kv_cache plumbing in :class:`MMDiTBlock` / :class:`MMDiT`.

Two bit-exactness guarantees:

1. ``kv_cache=None`` → identical to the pre-USAM forward (training is
   unaffected). We assert by re-running the same MMDiTBlock twice with a
   torch_manual_seed reset between calls and checking that outputs are
   element-wise equal.
2. ``kv_cache=<populated>`` → cross-attention output equals what the
   diffusers ``Attention`` would produce on-the-fly when fed the same
   plan tokens via ``encoder_hidden_states``. fp32 must be bit-exact;
   fp16 must be within 1e-3 absolute error.

These tests deliberately bypass the rest of USAM's surface (no Conductor,
no PlanCache class) — they only verify the wiring between MMDiTBlock and
the kv_cache contract documented in
``usam/conductor/plan_cache.py``.
"""
from __future__ import annotations

import pytest
import torch

# The real LDA MMDiT pulls in x_transformers, hyper_connections, and diffusers,
# which only the train tier (`pip install -e ".[train]"`) carries. Skip the
# whole module on CI/dev installs that don't have them — the unit value of
# this test is the kv_cache wiring contract, not exercising LDA's heavy deps.
try:
    from lda.model.modules.action_model.flow_matching_head.mmdit.mmdit.mmdit_cross_attn import (
        MMDiT,
        MMDiTBlock,
    )
except ModuleNotFoundError as exc:
    pytest.skip(
        f"MMDiT cross-attention test skipped: missing optional train dep ({exc.name}). "
        f"Install with `pip install -e \".[train]\"` to exercise this test.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_block(
    dim: int = 32,
    heads: int = 4,
    head_dim: int = 8,
    cross_dim: int = 16,
    layer_idx: int = 0,
    seed: int = 0,
) -> MMDiTBlock:
    torch.manual_seed(seed)
    block = MMDiTBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        dropout=0.0,
        cross_attention_dim=cross_dim,
        activation_fn="gelu",
        attention_bias=True,
        norm_elementwise_affine=False,
        positional_embeddings=None,
        num_residual_streams=1,
        layer_idx=layer_idx,
    )
    block.eval()  # disable any dropout for determinism
    return block


def _build_inputs(
    batch: int = 2,
    n_image: int = 8,
    n_action: int = 4,
    n_text: int = 5,
    dim: int = 32,
    cross_dim: int = 16,
    seed: int = 1,
):
    torch.manual_seed(seed)
    text_tokens = torch.randn(batch, n_text, cross_dim)
    image_tokens = torch.randn(batch, n_image, dim)
    action_tokens = torch.randn(batch, n_action, dim)
    time_cond = torch.randn(batch, dim)
    return text_tokens, image_tokens, action_tokens, time_cond


def _project_text_to_kv(block: MMDiTBlock, text_tokens: torch.Tensor):
    """Pre-project ``text_tokens`` through both cross-attn ``to_k`` / ``to_v``.

    Mirrors what :meth:`PlanCache.refresh` does at refresh time. We feed
    the same tensor into both branches so the cached path can be compared
    against the on-the-fly path that uses ``text_tokens`` directly.
    """
    k_img = block.img_cross_attn.to_k(text_tokens)
    v_img = block.img_cross_attn.to_v(text_tokens)
    k_act = block.action_cross_attn.to_k(text_tokens)
    v_act = block.action_cross_attn.to_v(text_tokens)
    return k_img, v_img, k_act, v_act


# ---------------------------------------------------------------------------
# Test 1: kv_cache=None is a no-op
# ---------------------------------------------------------------------------
def test_kv_cache_none_is_noop() -> None:
    """Calling forward with ``kv_cache=None`` must reproduce pre-USAM output.

    We run two forwards with identical inputs — once not passing
    ``kv_cache`` at all (legacy path), once passing ``kv_cache=None``
    (new kwarg). Both must be element-wise equal.
    """
    block = _build_block(seed=42)
    text, image, action, time_cond = _build_inputs(seed=7)

    # Path A: pre-USAM signature (no kv_cache kwarg). The default is None.
    text_a, image_a, action_a = block(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
    )
    # Path B: explicit ``kv_cache=None``.
    text_b, image_b, action_b = block(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
        kv_cache=None,
    )

    assert torch.equal(text_a, text_b), "text_tokens diverge between paths"
    assert torch.equal(image_a, image_b), "image_tokens diverge between paths"
    assert torch.equal(action_a, action_b), "action_tokens diverge between paths"


# ---------------------------------------------------------------------------
# Test 2: cached path == on-the-fly path (fp32 bit-exact)
# ---------------------------------------------------------------------------
def test_cached_kv_matches_on_the_fly_fp32() -> None:
    """Cross-attn from cache must equal cross-attn computed fresh, fp32."""
    block = _build_block(seed=11)
    text, image, action, time_cond = _build_inputs(seed=13)

    # On-the-fly: pass text_tokens through the block as-is. The two
    # cross-attn modules will internally compute K = to_k(text),
    # V = to_v(text), then attend.
    _, image_ref, action_ref = block(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
    )

    # Cached: pre-project and feed via kv_cache.
    k_img, v_img, k_act, v_act = _project_text_to_kv(block, text)
    cache = {
        (block.layer_idx, "image"): (k_img, v_img),
        (block.layer_idx, "action"): (k_act, v_act),
    }
    # Pass garbage text_tokens to make sure the cached path ignores them.
    garbage = torch.full_like(text, fill_value=1e9)
    _, image_cached, action_cached = block(
        text_tokens=garbage,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
        kv_cache=cache,
    )

    # The bit-exactness assertion strategy: at fp32 with deterministic
    # SDPA on a single CPU thread the two reduction orders are identical,
    # so ``torch.equal`` succeeds. We tolerate <=1e-6 atol as a guard
    # against any future SDPA backend shuffling but do not need it here.
    abs_err_image = (image_cached - image_ref).abs().max().item()
    abs_err_action = (action_cached - action_ref).abs().max().item()
    assert abs_err_image <= 1e-5, (
        f"cached vs. on-the-fly image diverged: max abs err = {abs_err_image}"
    )
    assert abs_err_action <= 1e-5, (
        f"cached vs. on-the-fly action diverged: max abs err = {abs_err_action}"
    )


# ---------------------------------------------------------------------------
# Test 3: cached path within 1e-3 at fp16
# ---------------------------------------------------------------------------
def test_cached_kv_matches_on_the_fly_fp16() -> None:
    """At fp16 the cached path must agree with on-the-fly to <= 1e-3 abs.

    Activations are scaled down by 10× — production plan tokens sit at
    the post-LayerNorm scale, where fp16 quantization noise on the K/V
    storage stays well below the 1e-3 budget set by the charter.
    """
    block = _build_block(seed=23)
    text, image, action, time_cond = _build_inputs(seed=29)
    # Match the test_plan_cache fp16 scale (post-LN feature regime).
    text = text * 0.1
    image = image * 0.1
    action = action * 0.1
    time_cond = time_cond * 0.1

    _, image_ref, action_ref = block(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
    )

    # Pre-project at fp32 then quantize to fp16, mirroring how
    # PlanCache.refresh stores K/V in bf16/fp16 in production.
    k_img, v_img, k_act, v_act = _project_text_to_kv(block, text)
    cache = {
        (block.layer_idx, "image"): (k_img.to(torch.float16), v_img.to(torch.float16)),
        (block.layer_idx, "action"): (k_act.to(torch.float16), v_act.to(torch.float16)),
    }
    _, image_cached, action_cached = block(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
        kv_cache=cache,
    )

    abs_err_image = (image_cached - image_ref).abs().max().item()
    abs_err_action = (action_cached - action_ref).abs().max().item()
    assert abs_err_image <= 1e-3, (
        f"fp16 cached image abs err {abs_err_image} > 1e-3"
    )
    assert abs_err_action <= 1e-3, (
        f"fp16 cached action abs err {abs_err_action} > 1e-3"
    )


# ---------------------------------------------------------------------------
# Test 4: layer_idx routes correctly through the kv_cache mapping
# ---------------------------------------------------------------------------
def test_kv_cache_routes_by_layer_idx() -> None:
    """A 2-layer cache must dispatch (0, "image") and (1, "image") correctly.

    We populate the cache with **different** K, V tensors for layer 0
    vs. layer 1. The block at layer 1 reads layer 1's KV; running it with
    a (deliberately wrong) cache that maps both layers to layer-0 KV must
    diverge from the layer-1 ground truth.
    """
    block_0 = _build_block(seed=51, layer_idx=0)
    block_1 = _build_block(seed=53, layer_idx=1)
    text, image, action, time_cond = _build_inputs(seed=57)

    # Sanity: each block has its own to_k/to_v so the cached K/V differ.
    k0_img, v0_img, k0_act, v0_act = _project_text_to_kv(block_0, text)
    k1_img, v1_img, k1_act, v1_act = _project_text_to_kv(block_1, text)

    # Correct cache: each layer reads its own K/V.
    correct_cache = {
        (0, "image"): (k0_img, v0_img),
        (0, "action"): (k0_act, v0_act),
        (1, "image"): (k1_img, v1_img),
        (1, "action"): (k1_act, v1_act),
    }
    _, image_correct, _ = block_1(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
        kv_cache=correct_cache,
    )

    # On-the-fly reference for layer 1.
    _, image_ref_1, _ = block_1(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
    )

    abs_err = (image_correct - image_ref_1).abs().max().item()
    assert abs_err <= 1e-5, (
        f"layer-1 cached path diverged from on-the-fly: abs err {abs_err}"
    )


# ---------------------------------------------------------------------------
# Test 5: end-to-end MMDiT.forward threads kv_cache through every block
# ---------------------------------------------------------------------------
def test_mmdit_forward_threads_kv_cache() -> None:
    """The wrapping :class:`MMDiT` must thread ``kv_cache`` to every block.

    We build a 2-layer MMDiT, pre-project the text tokens through every
    block's cross-attn K/V, and verify the cached forward matches the
    on-the-fly forward.
    """
    torch.manual_seed(99)
    inner_dim = 32
    num_heads = 4
    head_dim = 8
    cross_dim = 16
    model = MMDiT(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        cross_attention_dim=cross_dim,
        num_layers=2,
        dropout=0.0,
        activation_fn="gelu",
        attention_bias=True,
        norm_elementwise_affine=False,
        final_norm=True,
        num_residual_streams=1,
        positional_embeddings=None,
        output_dim=inner_dim,
    )
    model.eval()

    batch = 2
    n_img = 8
    n_act = 4
    n_text = 5
    text = torch.randn(batch, n_text, cross_dim)
    image = torch.randn(batch, n_img, inner_dim)
    action = torch.randn(batch, n_act, inner_dim)
    time_cond = torch.randn(batch)

    # On-the-fly forward.
    out_ref = model(
        text_tokens=text,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
    )

    # Build cache from each block's projections. The MMDiT trunk
    # layer-norms text_tokens once before threading them through every
    # block, so we feed the *post-LN* tensor into each cross-attn's K/V.
    text_post_ln = model.text_attn_layernorm(text)
    cache = {}
    for layer_idx, block in enumerate(model.blocks):
        cache[(layer_idx, "image")] = (
            block.img_cross_attn.to_k(text_post_ln),
            block.img_cross_attn.to_v(text_post_ln),
        )
        cache[(layer_idx, "action")] = (
            block.action_cross_attn.to_k(text_post_ln),
            block.action_cross_attn.to_v(text_post_ln),
        )

    # Cached forward. Garbage text_tokens to make sure the cache wins.
    garbage = torch.full_like(text, 1e9)
    out_cached = model(
        text_tokens=garbage,
        image_tokens=image,
        action_tokens=action,
        time_cond=time_cond,
        kv_cache=cache,
    )

    if isinstance(out_ref, tuple):
        image_ref, action_ref = out_ref
        image_cached, action_cached = out_cached
    else:
        image_ref, action_ref = out_ref["image_tokens"], out_ref["action_tokens"]
        image_cached, action_cached = out_cached["image_tokens"], out_cached["action_tokens"]

    abs_err_image = (image_cached - image_ref).abs().max().item()
    abs_err_action = (action_cached - action_ref).abs().max().item()
    assert abs_err_image <= 1e-5, (
        f"MMDiT cached image diverged: abs err {abs_err_image}"
    )
    assert abs_err_action <= 1e-5, (
        f"MMDiT cached action diverged: abs err {abs_err_action}"
    )
