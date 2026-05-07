# SPDX-License-Identifier: MIT
"""Unit tests for `usam.adapters.lora`."""
from __future__ import annotations

import torch
from torch import nn

from usam.adapters.lora import LoRALinear, apply_lora


def _zero_b_keeps_base_equal() -> None:
    """At init (B=0) the wrapper must reproduce the base output bit-exactly."""
    base = nn.Linear(16, 24, bias=True)
    base_w = base.weight.detach().clone()
    base_b = base.bias.detach().clone()
    wrapper = LoRALinear(base, r=8, modality_ids=["rgb", "depth", "flow"])

    x = torch.randn(2, 5, 16)
    out_base = nn.functional.linear(x, base_w, base_b)
    for mid in ("rgb", "depth", "flow"):
        out_w = wrapper(x, modality_id=mid)
        assert torch.allclose(out_base, out_w, atol=0, rtol=0), (
            f"LoRA at init must equal base for modality={mid}"
        )
    # Also without any modality:
    out_no_mod = wrapper(x, modality_id=None)
    assert torch.allclose(out_base, out_no_mod, atol=0, rtol=0)


def test_zero_lora_equals_base() -> None:
    _zero_b_keeps_base_equal()


def test_nonzero_lora_changes_output() -> None:
    base = nn.Linear(16, 24, bias=True)
    wrapper = LoRALinear(base, r=4, modality_ids=["rgb", "depth"])

    # Inflate one B matrix; output should now diverge from base.
    with torch.no_grad():
        wrapper.lora_B["depth"].fill_(0.5)

    x = torch.randn(3, 16)
    out_base = base(x)
    out_depth = wrapper(x, modality_id="depth")
    out_rgb = wrapper(x, modality_id="rgb")

    # depth has nonzero LoRA, rgb still does not
    assert not torch.allclose(out_base, out_depth)
    assert torch.allclose(out_base, out_rgb)


def test_grad_flows_only_to_lora_params() -> None:
    base = nn.Linear(8, 8, bias=True)
    wrapper = LoRALinear(base, r=4, modality_ids=["rgb", "depth"])

    # Base must be frozen.
    for p in wrapper.base.parameters():
        assert p.requires_grad is False, "base params must be frozen"

    # Make B nonzero so the gradient is actually nontrivial.
    with torch.no_grad():
        wrapper.lora_B["rgb"].fill_(0.1)

    x = torch.randn(2, 8, requires_grad=False)
    y = wrapper(x, modality_id="rgb").sum()
    y.backward()

    assert wrapper.base.weight.grad is None
    assert wrapper.base.bias.grad is None
    # LoRA params for the active modality must have grads.
    assert wrapper.lora_A["rgb"].grad is not None
    assert wrapper.lora_B["rgb"].grad is not None
    # LoRA params for the *inactive* modality should NOT receive grads
    # because they were never touched in the forward path.
    assert wrapper.lora_A["depth"].grad is None
    assert wrapper.lora_B["depth"].grad is None


def test_apply_lora_walks_qkv() -> None:
    """`apply_lora` should wrap every named Linear matching the suffix."""

    class TinyAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.key = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)
            self.proj = nn.Linear(8, 8)  # NOT wrapped

        def forward(self, x):  # pragma: no cover - shape only
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            return self.proj(q + k + v)

    class TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = TinyAttn()

    class TinyEnc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.ModuleList([TinyBlock() for _ in range(3)])

    enc = TinyEnc()
    wrappers = apply_lora(
        enc, r=4, target_module_names=("query", "key", "value"), modality_ids=["rgb", "depth"]
    )

    # Each modality should have one wrapper per (block × q/k/v) = 3 * 3 = 9.
    for mid in ("rgb", "depth"):
        assert len(wrappers[mid]) == 9

    # Sanity: q/k/v have been replaced by LoRALinear; proj is untouched.
    blk = enc.layer[0].attention
    assert isinstance(blk.query, LoRALinear)
    assert isinstance(blk.key, LoRALinear)
    assert isinstance(blk.value, LoRALinear)
    assert isinstance(blk.proj, nn.Linear) and not isinstance(blk.proj, LoRALinear)
