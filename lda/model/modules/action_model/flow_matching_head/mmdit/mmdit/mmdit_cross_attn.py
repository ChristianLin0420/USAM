from __future__ import annotations
from typing import Any, Mapping, Tuple, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    RMSNorm
)

from hyper_connections import (
    HyperConnections,
    Residual
)
from diffusers.models.attention import Attention, FeedForward
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

from lda.model.modules.action_model.flow_matching_head.cdit import TimestepEncoder
from lda.model.modules.action_model.flow_matching_head.mmdit.mmdit.mmdit_self_attn import JointAttention
from lda.model.modules.action_model.flow_matching_head.mmdit.mmdit.rope_3d import Rotary3D, Rotary1D
# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


# class

def _kv_from_cache(
    kv_cache: Any,
    layer_idx: int,
    branch: str,
) -> Tuple[Tensor, Tensor]:
    """Look up ``(K, V)`` for ``layer_idx`` / ``branch`` from a kv_cache.

    Accepts two shapes:

    * A mapping keyed by ``(layer_idx, branch)`` (e.g. a plain ``dict``).
      The cache contract documented in
      ``usam/conductor/plan_cache.py`` calls this the canonical layout.
    * Any object exposing ``get(layer_idx, branch="image"|"action")``
      (e.g. :class:`usam.conductor.plan_cache.PlanCache`).
    """
    assert branch in ("image", "action"), f"unknown branch {branch!r}"
    key = (int(layer_idx), branch)
    if isinstance(kv_cache, Mapping):
        assert key in kv_cache, f"kv_cache missing {key!r}"
        k, v = kv_cache[key]
        return k, v
    # PlanCache-like duck-type fallback.
    return kv_cache.get(int(layer_idx), branch=branch)


def _cached_cross_attention(
    attn: "Attention",
    hidden_states: Tensor,
    cached_k: Tensor,
    cached_v: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Run cross-attention reusing pre-projected K, V from the PlanCache.

    Mirrors :class:`diffusers.models.attention_processor.AttnProcessor2_0`
    but skips the ``to_k`` / ``to_v`` projections — the cache already holds
    their outputs, courtesy of :meth:`usam.conductor.plan_cache.PlanCache.refresh`.

    Parameters
    ----------
    attn : diffusers.models.attention.Attention
        The cross-attention module. We borrow ``to_q`` / ``to_out`` /
        ``heads`` from it; ``to_k`` / ``to_v`` are deliberately unused.
    hidden_states : Tensor
        Query stream, ``[B, S_q, D]``.
    cached_k, cached_v : Tensor
        Pre-projected K, V from the cache. Shape
        ``[B, S_plan, heads * head_dim]`` (i.e. after the linear and
        before any head split). Dtype is whatever the cache stores
        (typically bf16); we cast to the query dtype before SDPA.
    attention_mask : Optional[Tensor]
        Same convention as diffusers: additive mask broadcast to the
        ``[B, heads, S_q, S_plan]`` score tensor.

    Returns
    -------
    Tensor
        ``[B, S_q, D]``. Bit-exact match to the on-the-fly path at fp32
        when ``cached_k = attn.to_k(P_hat)`` and
        ``cached_v = attn.to_v(P_hat)``.
    """
    assert hidden_states.dim() == 3, (
        f"hidden_states must be [B,S,D], got {tuple(hidden_states.shape)}"
    )
    assert cached_k.dim() == 3 and cached_v.dim() == 3, (
        f"cached_k/v must be [B,S_plan,D], got {tuple(cached_k.shape)} / "
        f"{tuple(cached_v.shape)}"
    )

    b, s_q, _ = hidden_states.shape
    heads = attn.heads
    inner_dim = cached_k.shape[-1]
    head_dim = inner_dim // heads

    # Cast the cache to the query dtype so SDPA stays in one precision.
    cached_k = cached_k.to(hidden_states.dtype)
    cached_v = cached_v.to(hidden_states.dtype)

    q = attn.to_q(hidden_states)

    # [B, S, H*D_head] -> [B, H, S, D_head]
    q = q.view(b, s_q, heads, head_dim).transpose(1, 2)
    k = cached_k.view(b, cached_k.shape[1], heads, head_dim).transpose(1, 2)
    v = cached_v.view(b, cached_v.shape[1], heads, head_dim).transpose(1, 2)

    # Mirror AttnProcessor2_0: optional Q/K norm (only present when the
    # parent ``Attention`` was built with ``qk_norm`` enabled). The
    # current MMDiTBlock does not enable it, so this branch is dormant
    # in production — but keeping it makes the cached path future-proof.
    if getattr(attn, "norm_q", None) is not None:
        q = attn.norm_q(q)
    if getattr(attn, "norm_k", None) is not None:
        k = attn.norm_k(k)

    if attention_mask is not None:
        # Diffusers normalizes the mask in `prepare_attention_mask`; we
        # accept whatever the caller gives us and rely on broadcasting.
        attention_mask = attention_mask.to(hidden_states.dtype)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    # [B, H, S, D_head] -> [B, S, H*D_head]
    out = out.transpose(1, 2).reshape(b, s_q, inner_dim)
    out = out.to(q.dtype)

    # Standard diffusers Attention output projection: to_out is a Sequential
    # (Linear, Dropout). We follow the public pattern.
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    # Match diffusers' AttnProcessor2_0 trailing rescale (default 1.0).
    rescale = getattr(attn, "rescale_output_factor", 1.0)
    if rescale != 1.0:
        out = out / rescale
    return out


class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,

        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        layer_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        # USAM extension: each block records its own index so the
        # PlanCache can be looked up via ``kv_cache[(layer_idx, branch)]``
        # in :meth:`forward`. Default 0 keeps single-block tests / older
        # call sites that don't pass an index a no-op.
        self.layer_idx = int(layer_idx)

        # residual functions / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.image_cross_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim)

        self.action_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.action_cross_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.action_ff_residual_fn = residual_klass(num_residual_streams, dim = dim)

        # pos embedding
        self.positional_embeddings = positional_embeddings
        if positional_embeddings == "sinusoidal":
            self.image_pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
            self.action_pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        elif positional_embeddings == "rope":
            self.image_pos_embed = Rotary3D(dim=dim)
            self.action_pos_embed = Rotary1D(dim=dim)
        else:
            self.image_pos_embed = None
            self.action_pos_embed = None
        # handle optional time conditioning

        dim_gammas = (
            *((dim,) * 4),
            *((dim,) * 4),
        )

        dim_betas = (
            *((dim,) * 2),
            *((dim,) * 2),
        )

        self.cond_dims = (*dim_gammas, *dim_betas)

        to_cond_linear = nn.Linear(dim, sum(self.cond_dims))

        self.to_cond = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            to_cond_linear
        )

        nn.init.zeros_(to_cond_linear.weight)
        nn.init.zeros_(to_cond_linear.bias)
        nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms

        self.image_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        self.image_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        # self.text_attn_layernorm = nn.LayerNorm(cross_attention_dim, elementwise_affine = False)

        self.image_ff_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_ff_layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        # attention and feedforward

        self.img_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.action_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # joint self attention
        self.patch_shape = kwargs.get("patch_shape", None)
        self.glob_len = kwargs.get("glob_len", 0)
        self.obs_timesteps = kwargs.get("obs_timesteps", 1)
        self.num_register_tokens = kwargs.get("num_register_tokens", 0)
        self.joint_attn = JointAttention(
            dim_inputs = (dim, dim),
            dim_head = attention_head_dim, 
            heads = num_attention_heads,
            flash = flash_attn,
            patch_shape = self.patch_shape,
            glob_len = self.glob_len,
            obs_timesteps = self.obs_timesteps,
            num_register_tokens=self.num_register_tokens,
        )

        self.image_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,)
        self.action_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        action_tokens,
        text_mask = None,
        time_cond = None,
        kv_cache: Optional[Mapping[Tuple[int, str], Tuple[Tensor, Tensor]]] = None,
    ):
        """Run one MM-DiT block.

        USAM extension: if ``kv_cache`` is provided, the cross-attention
        K / V projections are skipped — we read pre-projected tensors out
        of ``kv_cache[(self.layer_idx, "image")]`` for the image branch
        and ``kv_cache[(self.layer_idx, "action")]`` for the action
        branch. ``kv_cache`` may also be an object exposing
        ``get(layer_idx, branch="image"|"action") -> (K, V)`` (e.g.
        :class:`usam.conductor.plan_cache.PlanCache`).

        When ``kv_cache is None`` the path is bit-exact identical to the
        pre-USAM behaviour (training is unaffected).
        """

        (
            image_pre_attn_gamma,
            image_post_attn_gamma,
            image_pre_ff_gamma,
            image_post_ff_gamma,
            action_pre_attn_gamma,
            action_post_attn_gamma,
            action_pre_ff_gamma,
            action_post_ff_gamma,
            image_pre_attn_beta,
            image_pre_ff_beta,
            action_pre_attn_beta,
            action_pre_ff_beta,
        ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)
        action_tokens, add_action_residual = self.action_attn_residual_fn(action_tokens)

        image_tokens = self.image_attn_layernorm(image_tokens)
        action_tokens = self.action_attn_layernorm(action_tokens)

        image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta
        action_tokens = action_tokens * action_pre_attn_gamma + action_pre_attn_beta

        if self.positional_embeddings == "sinusoidal":
            image_tokens = self.image_pos_embed(image_tokens)
            action_tokens = self.action_pos_embed(action_tokens)

            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
            )
        elif self.positional_embeddings == "rope":
            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
                image_rope_3d_embedding = self.image_pos_embed,
                action_rope_embedding = self.action_pos_embed,
            )
        else:
            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
            )
        # add attention residual
        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # 2) cross attention
        image_tokens, add_image_residual = self.image_cross_attn_residual_fn(image_tokens)
        action_tokens, add_action_residual = self.action_cross_attn_residual_fn(action_tokens)

        image_tokens = self.image_cross_attn_layernorm(image_tokens)
        action_tokens = self.action_cross_attn_layernorm(action_tokens)
        # text_tokens = self.text_attn_layernorm(text_tokens)
        # USAM: when a Plan-KV-Cache is present, skip the K/V projections
        # and read pre-projected tensors out of the cache. The kv_cache=None
        # branch is a bit-exact no-op vs. the pre-USAM forward.
        if kv_cache is not None:
            k_img, v_img = _kv_from_cache(kv_cache, self.layer_idx, "image")
            image_tokens = _cached_cross_attention(
                self.img_cross_attn,
                image_tokens,
                cached_k=k_img,
                cached_v=v_img,
                attention_mask=text_mask,
            )
            k_act, v_act = _kv_from_cache(kv_cache, self.layer_idx, "action")
            action_tokens = _cached_cross_attention(
                self.action_cross_attn,
                action_tokens,
                cached_k=k_act,
                cached_v=v_act,
                attention_mask=text_mask,
            )
        else:
            image_tokens = self.img_cross_attn(
                image_tokens,
                encoder_hidden_states=text_tokens,
                attention_mask=text_mask,
            )
            action_tokens = self.action_cross_attn(
                action_tokens,
                encoder_hidden_states=text_tokens,
                attention_mask=text_mask,
            )

        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)

        # condition attention output
        image_tokens = image_tokens * image_post_attn_gamma
        action_tokens = action_tokens * action_post_attn_gamma
        
        # handle feedforward adaptive layernorm
        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        action_tokens, add_action_residual = self.action_ff_residual_fn(action_tokens)
        action_tokens = self.action_ff_layernorm(action_tokens)

        image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta
        action_tokens = action_tokens * action_pre_ff_gamma + action_pre_ff_beta

        # images feedforward

        image_tokens = self.image_ff(image_tokens)
        action_tokens = self.action_ff(action_tokens)
        # images condition feedforward output

        image_tokens = image_tokens * image_post_ff_gamma
        action_tokens = action_tokens * action_post_ff_gamma
        # images feedforward residual

        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # return

        return text_tokens, image_tokens, action_tokens

# mm dit transformer - simply many blocks

class MMDiT(ModelMixin, ConfigMixin):
    @register_to_config 
    def __init__(
        self,
        *,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        final_norm = True,
        num_residual_streams = 1,
        **kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)

        # only norm once for text tokens
        self.text_attn_layernorm = nn.LayerNorm(cross_attention_dim, elementwise_affine = False)
        self.blocks = ModuleList([])

        for layer_idx in range(num_layers):
            block = MMDiTBlock(
                dim = self.inner_dim,
                num_attention_heads = num_attention_heads,
                attention_head_dim = attention_head_dim,
                cross_attention_dim= cross_attention_dim,
                num_residual_streams = num_residual_streams,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                positional_embeddings=positional_embeddings,
                num_positional_embeddings=self.config.max_num_positional_embeddings,
                final_dropout=final_dropout,
                layer_idx=layer_idx,
                **kwargs
            )

            self.blocks.append(block)

        self.norm = RMSNorm(self.inner_dim) if final_norm else nn.Identity()
        self.action_norm = RMSNorm(self.inner_dim) if final_norm else nn.Identity()

        # Output blocks
        self.action_proj_out = nn.Linear(self.inner_dim, self.config.output_dim)
        self.image_proj_out = nn.Linear(self.inner_dim, self.config.output_dim)

        # USAM extensions ----------------------------------------------------
        # Optional proprio modulation source for AdaLN-Zero (Edit 1).
        # Gated by `enable_proprio_cond`; the projection input dim is
        # provided via `proprio_dim` and defaults to 50 (the USAM-LeRobot
        # padded state vector). When disabled the attribute is None and the
        # forward path is identical to the original LDA-1B model.
        self.enable_proprio_cond = bool(kwargs.get("enable_proprio_cond", False))
        proprio_dim = int(kwargs.get("proprio_dim", 50))
        self.proprio_proj = (
            nn.Linear(proprio_dim, self.inner_dim) if self.enable_proprio_cond else None
        )

        # Optional depth flow-matching head (Edit 2). Gated by
        # `enable_depth_head`. The smoke config disables it.
        self.enable_depth_head = bool(kwargs.get("enable_depth_head", False))
        self.depth_proj_out = (
            nn.Linear(self.inner_dim, self.config.output_dim) if self.enable_depth_head else None
        )
        # --------------------------------------------------------------------

        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        *,
        image_tokens,
        action_tokens,
        text_tokens,
        register_tokens = None,
        text_mask = None,
        time_cond = None,
        task_embedding = None,
        proprio: Optional[Tensor] = None,
        kv_cache: Optional[Mapping[Tuple[int, str], Tuple[Tensor, Tensor]]] = None,
    ):

        if register_tokens is not None:
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')
        image_tokens = self.expand_streams(image_tokens)
        action_tokens = self.expand_streams(action_tokens)

        text_tokens = self.text_attn_layernorm(text_tokens)
        # cond embedding
        if time_cond is not None:
            time_cond = self.timestep_encoder(time_cond)
            if task_embedding is not None:
                time_cond += task_embedding
            if proprio is not None and self.proprio_proj is not None:
                # USAM: third modulation source (proprio embedding).
                time_cond = time_cond + self.proprio_proj(proprio)

        for ind, block in enumerate(self.blocks):

            text_tokens, image_tokens, action_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                action_tokens = action_tokens,
                text_mask = text_mask,
                kv_cache = kv_cache,
            )
        if register_tokens is not None:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        image_tokens = self.reduce_streams(image_tokens)
        action_tokens = self.reduce_streams(action_tokens)

        image_tokens = self.norm(image_tokens)
        action_tokens = self.action_norm(action_tokens)
        
        # proj to output dim
        action_pred = self.action_proj_out(action_tokens)
        image_pred = self.image_proj_out(image_tokens)

        # USAM: optional depth head shares the image-branch features.
        if self.depth_proj_out is not None:
            outputs: dict[str, Tensor] = {
                "image_tokens": image_pred,
                "action_tokens": action_pred,
                "depth_tokens": self.depth_proj_out(image_tokens),
            }
            return outputs

        return image_pred, action_pred

def test_mmdit():
    device = "cpu"
    batch_size = 2
    
    # Dimensions
    dim_text = 384
    inner_dim = 512  # 8 heads * 64 dim
    num_img_tokens = 64
    num_action_tokens = 10
    num_text_tokens = 16
    num_register_tokens = 4
    # Model (with time conditioning and register tokens)
    model = MMDiT(
        num_attention_heads=8,
        attention_head_dim=64,
        cross_attention_dim=dim_text,
        num_layers=2,
        dropout=0.1,
        activation_fn="gelu",
        attention_bias=True,
        norm_elementwise_affine=False,
        final_norm=True,
        num_residual_streams=1,
    ).to(device)
    
    # Inputs
    text_tokens = torch.randn(batch_size, num_text_tokens, dim_text).to(device)
    image_tokens = torch.randn(batch_size, num_img_tokens, inner_dim).to(device)
    action_tokens = torch.randn(batch_size, num_action_tokens, inner_dim).to(device)
    register_tokens = torch.randn(batch_size, num_register_tokens, inner_dim).to(device)

    time_cond = torch.randn(batch_size, ).to(device)  

    
    # Forward pass
    image_out, action_out = model(
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        action_tokens=action_tokens,
        time_cond=time_cond,  
        register_tokens=register_tokens,
        text_mask=None,
    )
    
    print("✅ Success!")
    print(f"Image output: {image_out.shape}")   # [2, 64, 512]
    print(f"Action output: {action_out.shape}") # [2, 10, 512]
    
    # Sanity check
    assert not torch.isnan(image_out).any()
    assert not torch.isnan(action_out).any()

if __name__ == "__main__":
    test_mmdit()