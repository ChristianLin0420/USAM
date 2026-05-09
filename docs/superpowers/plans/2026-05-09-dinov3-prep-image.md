# DINOv3-ViT-L/16 Prep Image + 8-GPU stage_4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-09-dinov3-prep-image-design.md`](../specs/2026-05-09-dinov3-prep-image-design.md) (commit `d4914d9`).

**Goal:** Cut USAM over to `facebook/dinov3-vitl16-pretrain-lvd1689m` (patch-16) end-to-end, parallelize stage_4 DINO caching across 8 GPUs, restyle all three Dockerfiles in the unified format with the gated weights baked in, and gate everything behind an A40 smoke that runs the full DROID-chunk-0 pipeline before shipping the SIF to Slurm.

**Architecture:** Five sequential waves. Wave A is a configs-and-code patch-16 cutover with no Dockerfile work (so existing unit tests stay green). Wave B adds an `mp.spawn` wrapper around the existing single-process `encode_chunk()` for 8-GPU sharding without changing its episode-level idempotency. Wave C restyles all three Dockerfiles in the Geo-Flow VLA format and bakes the gated weights via `--build-arg HF_TOKEN` (T1/T2 required, T0 optional). Wave D is a manual smoke gate the user runs on the local 8xA40 box (build → image self-test → 8-GPU TriDINO load → end-to-end DROID chunk 0 → pytest gate). Wave E ships the SIF to Slurm and writes the reproduction runbook from the *actually verified* commands.

**Tech Stack:** PyTorch 2.6, HuggingFace `transformers==4.57.0`, `torch.multiprocessing.spawn`, Docker BuildKit (`--secret`), Singularity, Slurm, pytest.

**Stage-naming correction (vs. spec):** The spec references a single `stage_3_flow_depth` for convenience; the actual repo splits flow and depth into `prep/stage_2b_compute_flow.py` and `prep/stage_2c_compute_depth.py`, with `stage_3_canonical.py` doing action canonicalization. This plan uses the real names. A small follow-up commit (Task E5) corrects the spec wording.

---

## File Map

**Modified:**
- `usam/encoders/tri_dino.py` — defaults bump to patch-16, image_size 448, vit_b_16; assertion added.
- `prep/stage_4_dino_cache.py` — `target_hw=(448,448)`, `embed_dim` parameterized, new `encode_chunk_multigpu()` wrapper, real CLI entry.
- `configs/model/usam_1_4b.yaml` — DINOv3 ViT-L/16 HF id, `image_size: 448`, `patch_size: 16`.
- `configs/model/usam_350m_smoke.yaml` — DINOv3 ViT-B/16 HF id, `image_size: 448`, `patch_size: 16`.
- `configs/train/adapter_pretrain.yaml` — same as smoke.
- `tests/unit/test_tri_dino.py` — expectations bumped 27→28, 378→448, 14→16, 729→784, ViT-B/14→ViT-B/16.
- `docker/Dockerfile.local_a40` — restyled, optional bake.
- `docker/Dockerfile.prep_a100` — restyled, required bake.
- `docker/Dockerfile.train_h200` — restyled (NGC base preserved), required bake.
- `docker/README.md` — build-arg / BuildKit-secret instructions, link to HOWTO.
- `slurm/env.sh` — drop `HUGGINGFACE_TOKEN` requirement note for compute jobs.

**Created:**
- `tests/integration/test_dino_cache_sharding.py` — sharding correctness, mocked encoder, runs always.
- `tests/integration/test_dino_cache_real_weights.py` — real-weights smoke, gated by `USAM_DINOV3_CKPT`.
- `docs/HOWTO_PREP_DINOV3.md` — operator runbook, written from verified commands at the end of Wave E.

---

# Wave A — Configs & code patch-16 cutover

Goal: every existing test still passes after Wave A. No Dockerfile work yet.

## Task A1: Bump `TriDinoConfig` defaults to patch-16 / image_size 448

**Files:**
- Modify: `usam/encoders/tri_dino.py:77-83`

- [ ] **Step 1: Read the current defaults**

The current dataclass (lines 77-83):
```python
    dinov3_ckpt: str = ""
    dinov3_arch: str = "vit_b_14"
    # 27 * 14 = 378; preserves the canonical 27x27=729 patch grid that the
    # cache + plan reference. The plan colloquially says "384²"; YAMLs
    # already override to 378. Default matches the binding contract.
    image_size: int = 378
    patch_size: int = 14
    embed_dim: int = 768
```

- [ ] **Step 2: Replace with patch-16 defaults**

Replace those lines with:
```python
    dinov3_ckpt: str = ""
    dinov3_arch: str = "vit_b_16"
    # 28 * 16 = 448. ViT-B/16 + 448x448 → 28x28 = 784 patches. Token count
    # is 1 (CLS) + num_register_tokens + 784. The cache slices the first
    # n_keep_tokens patches downstream (see `extract_features`).
    image_size: int = 448
    patch_size: int = 16
    embed_dim: int = 768
```

- [ ] **Step 3: Update `MiniDinoBackbone` defaults**

Same file, line 211:
```python
    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        hidden_size: int = 768,
```
becomes:
```python
    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 16,
        hidden_size: int = 768,
```

- [ ] **Step 4: Add divisibility assertion in `TriDINOTower.__init__`**

After the existing assertion (line 297-300), and after `patch_size = self.rgb_patch.kernel_size[0]` is set (around line 303), add:
```python
        assert config.image_size % patch_size == 0, (
            f"image_size={config.image_size} must be divisible by patch_size={patch_size}"
        )
```

- [ ] **Step 5: Run existing tri-dino unit tests — they should fail (expected)**

Run: `pytest tests/unit/test_tri_dino.py -v`
Expected: FAIL — assertions still expect 27×27=729 patches; we'll fix them in Task A3.

- [ ] **Step 6: Stage but do not commit**

```bash
git add usam/encoders/tri_dino.py
```
Commit comes after Task A3 so all unit tests are green in one commit.

## Task A2: Update `test_tri_dino.py` expectations

**Files:**
- Modify: `tests/unit/test_tri_dino.py`

- [ ] **Step 1: Read the current `_make_tower` defaults (lines 20-49)**

Existing helper sets `image_size=378, patch_size=14, dinov3_arch="vit_b_14"`.

- [ ] **Step 2: Update `_make_tower` to patch-16**

Replace its body (keeping the function name and signature) with:
```python
def _make_tower(
    embed_dim: int = 768,
    # 28×16 = 448 — patch-16 grid 28×28 → 784 patches.
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
```

- [ ] **Step 3: Update `test_forward_shape_all_modalities`**

Replace the docstring + the hardcoded 729 expectation with:
```python
def test_forward_shape_all_modalities() -> None:
    """ViT-B/16 at 448² → 784 patches + 1 [CLS] + 4 register = 789 tokens."""
    tower = _make_tower()
    img_size = tower.image_size
    bs = 2
    grid = img_size // tower.patch_size  # 28
    expected_tokens = 1 + tower.num_register_tokens + grid * grid
    assert expected_tokens == 1 + 4 + 784
```

The rest of the function (rgb / depth / flow forward + shape checks) does not change.

- [ ] **Step 4: Search the file for any other hardcoded 27, 729, 378, or "vit_b_14"**

Run: `grep -nE '\b(27|729|378|vit_b_14)\b' tests/unit/test_tri_dino.py`
For each hit, update to the patch-16 equivalent (28, 784, 448, vit_b_16) so the docstring + asserts match.

- [ ] **Step 5: Run unit tests — they should pass**

Run: `pytest tests/unit/test_tri_dino.py -v`
Expected: PASS (all tests in this file).

- [ ] **Step 6: Run the broader unit suite to catch any other regressions**

Run: `pytest tests/unit/ -x -q`
Expected: PASS. If `test_dataloader.py` or `test_lora.py` reference the old shape, fix them inline using the same pattern.

- [ ] **Step 7: Commit Tasks A1 + A2 together**

```bash
git add usam/encoders/tri_dino.py tests/unit/test_tri_dino.py
# also add any other unit-test files updated in Step 6
git commit -m "feat(encoder): cut TriDinoConfig defaults over to patch-16 / 448²

DINOv3-ViT-{B,L}/16 produces a 28×28 = 784 patch grid at 448². The
existing extract_features() still just slices the first n_keep_tokens
patch tokens, so no pooling math changes. Adds a runtime assertion
that image_size is divisible by patch_size."
```

## Task A3: Update `DinoCacheConfig` and parameterize embed_dim

**Files:**
- Modify: `prep/stage_4_dino_cache.py:29-53,109-145,56-73`

- [ ] **Step 1: Update `DinoCacheConfig.target_hw`**

Replace lines 29-53 with:
```python
@dataclass
class DinoCacheConfig:
    """Hyperparameters for DINO feature caching.

    Attributes
    ----------
    target_hw : tuple[int, int]
        Inference resolution. ViT-B/16 (or ViT-L/16) at 448x448 yields a
        28x28 = 784 patch grid; the cache keeps the first ``n_keep_tokens``
        patches plus [CLS].
    n_keep_tokens : int
        Number of patch tokens kept per frame. Default 64; with [CLS] prepended
        the on-disk shard's per-frame token dimension is 65.
    embed_dim : int
        Hidden dim of the encoder. 768 for ViT-B/16, 1024 for ViT-L/16. Used
        only by the placeholder (encoder=None) path; the real path reads
        the dim off the encoder.
    batch_size : int
    cache_fps : int
        Target output fps. We stride raw frames so the cache is at this fps.
    fp16 : bool
        Always True at runtime; arg kept for parity with the other stages.
    """

    target_hw: tuple[int, int] = (448, 448)
    n_keep_tokens: int = 64
    embed_dim: int = 768
    batch_size: int = 16
    cache_fps: int = 5
    fp16: bool = True
```

- [ ] **Step 2: Update `_load_tri_dino` default arch**

Replace lines 56-73 (the `_load_tri_dino` function's body) keeping signature, except change the default:
```python
def _load_tri_dino(ckpt_path: Path, dinov3_arch: str = "vit_b_16"):
```
and the body's `cfg = TriDinoConfig(...)` line stays the same — the dataclass now has patch-16 defaults from Task A1.

- [ ] **Step 3: Parameterize the placeholder shape by `cfg.embed_dim`**

In `encode_chunk` (lines 138-141 currently):
```python
                if encoder is None:
                    feats = torch.zeros(
                        (len(idxs), cfg.n_keep_tokens + 1, 768), dtype=torch.float16
                    )
```
becomes:
```python
                if encoder is None:
                    feats = torch.zeros(
                        (len(idxs), cfg.n_keep_tokens + 1, cfg.embed_dim),
                        dtype=torch.float16,
                    )
```

- [ ] **Step 4: Run any tests that exercise stage_4 placeholder path**

Run: `grep -rln "stage_4_dino_cache\|encode_chunk\|DinoCacheConfig" tests/`
Run pytest on each file found:
`pytest <files-from-grep> -v`
Expected: PASS. If a test hardcodes 768 in an assert against the placeholder shape, update it to `cfg.embed_dim`.

- [ ] **Step 5: Commit**

```bash
git add prep/stage_4_dino_cache.py
# include any test files touched in Step 4
git commit -m "feat(prep): bump stage_4 to 448² and parameterize placeholder embed_dim

Matches patch-16 cutover in TriDinoConfig. Placeholder path (no encoder
provided) now shapes its zero tensor by cfg.embed_dim instead of hardcoding
768, so ViT-L/16 (1024) callers get the right shape too."
```

## Task A4: Update the three model/training configs

**Files:**
- Modify: `configs/model/usam_1_4b.yaml:14-22`
- Modify: `configs/model/usam_350m_smoke.yaml:11-16`
- Modify: `configs/train/adapter_pretrain.yaml:23-27`

- [ ] **Step 1: `usam_1_4b.yaml`** — replace lines 14-22 (the encoder block's HF/arch/image_size/patch_size keys) with:
```yaml
  # HuggingFace model id; weights are baked into the prep / train Docker
  # images so compute nodes never need an HF token at runtime.
  dinov3_ckpt: facebook/dinov3-vitl16-pretrain-lvd1689m
  dinov3_arch: vit_l_16
  # 28 × 16 = 448. ViT-L/16 patch grid 28×28 → 784 patches → 65-token cache.
  image_size: 448
  patch_size: 16
  embed_dim: 1024
```

- [ ] **Step 2: `usam_350m_smoke.yaml`** — replace lines 11-16 with:
```yaml
  dinov3_ckpt: facebook/dinov3-vitb16-pretrain-lvd1689m
  dinov3_arch: vit_b_16
  # 28 × 16 = 448. ViT-B/16 patch grid 28×28 → 784 patches → 65-token cache.
  image_size: 448
  patch_size: 16
  embed_dim: 768
```

- [ ] **Step 3: `configs/train/adapter_pretrain.yaml`** — replace lines 23-27 with:
```yaml
  dinov3_ckpt: facebook/dinov3-vitb16-pretrain-lvd1689m
  dinov3_arch: vit_b_16
  image_size: 448
  patch_size: 16
  embed_dim: 768
```

- [ ] **Step 4: Smoke-load each YAML**

Run:
```bash
python -c "
import yaml
for f in [
    'configs/model/usam_1_4b.yaml',
    'configs/model/usam_350m_smoke.yaml',
    'configs/train/adapter_pretrain.yaml',
]:
    cfg = yaml.safe_load(open(f))
    enc = cfg.get('encoder', cfg)
    assert enc['patch_size'] == 16, f
    assert enc['image_size'] == 448, f
    assert 'vitl16' in enc['dinov3_ckpt'] or 'vitb16' in enc['dinov3_ckpt'], f
    print(f, 'ok:', enc['dinov3_arch'], enc['embed_dim'])
"
```
Expected: 3 lines printed, all `ok:`.

- [ ] **Step 5: Commit**

```bash
git add configs/model/usam_1_4b.yaml configs/model/usam_350m_smoke.yaml configs/train/adapter_pretrain.yaml
git commit -m "feat(configs): cut model/training configs over to DINOv3-ViT-{B,L}/16

Replaces local /datasets/dinov3 paths with HF model ids (gated weights
are baked into prep/train Docker images, so no runtime HF call). Bumps
image_size 378→448 and patch_size 14→16 across all three configs."
```

## Task A5: Run the broader test suite to confirm no regressions

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/unit/ -x -q`
Expected: PASS.

- [ ] **Step 2: Run integration smoke-train CPU plumbing test (uses 350m_smoke config)**

Run: `pytest tests/integration/test_smoke_train.py::test_smoke_train_cpu_plumbing -v`
Expected: PASS. (This test runs without GPUs; it loads `usam_350m_smoke.yaml` and `tiny_droid` fixture.)
If FAIL: the smoke fixture's DINO cache shape may have been baked against the old 768 placeholder. Inspect the failure; if it's a shape mismatch, the fixture needs regeneration — see Task A6.

- [ ] **Step 3: Run integration pipeline test**

Run: `pytest tests/integration/test_pipeline_end_to_end.py -v`
Expected: PASS. If FAIL: same — likely a placeholder shape mismatch fixed in Task A6.

## Task A6 (conditional): Regenerate `tiny_*` golden fixtures

**Only do this task if Task A5 surfaced a fixture shape mismatch.**

**Files:**
- Inspect: `tests/fixtures/golden_data/` (or wherever `tiny_droid` lives — `tests/conftest.py` has the path)
- Possibly modify: a fixture-generation script if one exists, otherwise the fixtures themselves.

- [ ] **Step 1: Find the fixture generator**

Run: `grep -rn "tiny_droid\|fixture\|golden" tests/conftest.py prep/ 2>/dev/null | head`
Expected: a generator function or a `pytest fixture` that materializes `tiny_droid` on demand. If it materializes via stage_4's placeholder path, the placeholder is now `cfg.embed_dim`-shaped (Task A3) and should auto-regenerate to ViT-B/16 dim correctly.

- [ ] **Step 2: Delete cached fixtures and rerun the test**

```bash
rm -rf tests/fixtures/golden_data/tiny_droid_cache_/* 2>/dev/null  # adjust path to actual cache dir
pytest tests/integration/test_smoke_train.py::test_smoke_train_cpu_plumbing -v
```
Expected: fixture regenerates with new shape; test passes.

- [ ] **Step 3: Commit only if fixture binaries changed**

```bash
git status
# If only test files changed (no binary fixture diffs), nothing to commit.
# If binaries changed and they're tracked, commit them:
git add tests/fixtures/...
git commit -m "test: regenerate tiny_droid fixture for patch-16 shape"
```

---

# Wave B — 8-GPU stage_4 sharding

## Task B1: Sharding test (mocked encoder)

**Files:**
- Create: `tests/integration/test_dino_cache_sharding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_dino_cache_sharding.py`:
```python
# SPDX-License-Identifier: MIT
"""Sharding correctness for stage_4_dino_cache.encode_chunk_multigpu.

These tests do NOT need a real DINOv3 encoder — they exercise the
multi-process episode partitioning by passing dinov3_ckpt=None (which
takes stage_4's placeholder/zero-tensor path) and asserting:

* Each rank writes exactly one shard file named file-{rank:03d}.safetensors.
* The union of episode_idxs across all rank shards equals the input set.
* Episode partition is disjoint (no episode appears in two shards).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def _build_synthetic_chunk(chunk_dir: Path, num_episodes: int, num_frames: int = 8) -> None:
    """Materialize ``num_episodes`` ep_*/ subdirs with the on-disk layout
    that ``encode_chunk`` expects for one (camera, modality) combo."""
    for ep_idx in range(num_episodes):
        ep_dir = chunk_dir / f"ep_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # camera_head_rgb.npy: [T, H, W, 3] uint8
        np.save(ep_dir / "camera_head_rgb.npy",
                np.zeros((num_frames, 32, 32, 3), dtype=np.uint8))
        (ep_dir / "meta.json").write_text(json.dumps({"episode_index": ep_idx}))


def test_sharding_partitions_episodes_disjoint(tmp_path: Path) -> None:
    """world_size=4 across 10 episodes ⇒ 4 shard files, partition disjoint."""
    from prep.stage_4_dino_cache import encode_chunk_multigpu

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_synthetic_chunk(staged, num_episodes=10)

    written = encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=None,           # placeholder path — no GPU needed
        source_fps=30,
        world_size=4,
    )
    # Filter to the (cam, mod) combo we built.
    rgb_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(rgb_dir.glob("file-*.safetensors"))
    assert [s.name for s in shards] == [
        "file-000.safetensors", "file-001.safetensors",
        "file-002.safetensors", "file-003.safetensors",
    ]
    # Read the safetensors back; key names encode episode_index.
    from usam.dataloader.feature_cache import read_feature_shard

    seen: set[int] = set()
    for shard in shards:
        ep_to_tensor = read_feature_shard(shard)
        for ep_idx in ep_to_tensor.keys():
            assert ep_idx not in seen, f"duplicate ep_idx={ep_idx} across shards"
            seen.add(int(ep_idx))
    assert seen == set(range(10))


def test_sharding_with_world_size_larger_than_episodes(tmp_path: Path) -> None:
    """world_size=8 across 3 episodes ⇒ first 3 ranks write shards, others empty."""
    from prep.stage_4_dino_cache import encode_chunk_multigpu

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_synthetic_chunk(staged, num_episodes=3)

    encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=None,
        source_fps=30,
        world_size=8,
    )
    rgb_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(rgb_dir.glob("file-*.safetensors"))
    # At most 3 shards because there are 3 episodes; ranks 3..7 produce nothing.
    assert 1 <= len(shards) <= 3
    from usam.dataloader.feature_cache import read_feature_shard

    seen: set[int] = set()
    for shard in shards:
        seen |= {int(k) for k in read_feature_shard(shard).keys()}
    assert seen == {0, 1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_dino_cache_sharding.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode_chunk_multigpu'`. That's the failing-test signal we want — Task B2 implements it.

- [ ] **Step 3: Stage test (don't commit yet — Task B2 will commit them together)**

```bash
git add tests/integration/test_dino_cache_sharding.py
```

## Task B2: Implement `encode_chunk_multigpu` in stage_4

**Files:**
- Modify: `prep/stage_4_dino_cache.py` — append new function after `encode_chunk`.

- [ ] **Step 1: Add the multi-GPU wrapper**

Append to `prep/stage_4_dino_cache.py` (after the existing `_encode_modality` function, before `__all__`):

```python
# ---------------------------------------------------------------------------
# Multi-GPU sharding wrapper
# ---------------------------------------------------------------------------
def _shard_episodes_by_rank(
    staged_chunk_dir: Path, world_size: int, rank: int
) -> Path:
    """Return a transient view of ``staged_chunk_dir`` whose ``ep_*`` symlinks
    are filtered to the episodes owned by ``rank``.

    We build a per-rank scratch directory under ``staged_chunk_dir.parent /
    f"_shard_view_{rank}_of_{world_size}"`` and symlink only the episode dirs
    where ``ep_idx % world_size == rank``. This lets us reuse the existing
    single-process ``encode_chunk`` unmodified — it just sees fewer episodes.
    """
    import json as _json

    view_root = staged_chunk_dir.parent / f"_shard_view_{rank}_of_{world_size}"
    view_root.mkdir(parents=True, exist_ok=True)
    # Clear any stale symlinks from a previous run.
    for old in view_root.glob("ep_*"):
        if old.is_symlink():
            old.unlink()
        elif old.is_dir():
            import shutil
            shutil.rmtree(old)
    for ep_dir in sorted(staged_chunk_dir.glob("ep_*")):
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        ep_idx = int(_json.loads(meta_path.read_text())["episode_index"])
        if ep_idx % world_size != rank:
            continue
        (view_root / ep_dir.name).symlink_to(ep_dir.resolve(), target_is_directory=True)
    return view_root


def _encode_chunk_worker(
    rank: int,
    world_size: int,
    staged_chunk_dir: str,
    output_root: str,
    modalities: tuple[str, ...],
    cameras: tuple[str, ...],
    dinov3_ckpt: Optional[str],
    source_fps: int,
    config_kwargs: dict,
) -> None:
    """torch.multiprocessing.spawn entry point.

    Pinned to one GPU; loads its own DINOv3; processes only the episodes
    owned by ``rank``; writes ``file-{rank:03d}.safetensors`` per (cam, mod).
    """
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format=f"[rank {rank}] %(asctime)s %(message)s")
    log = _logging.getLogger(__name__)

    # Pin to our GPU. spawn launches us with all GPUs visible, so we have
    # to set_device explicitly. CUDA_VISIBLE_DEVICES is not honored after
    # the parent process has already initialized CUDA in some PyTorch builds.
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.set_device(rank)

    view_dir = _shard_episodes_by_rank(
        Path(staged_chunk_dir), world_size=world_size, rank=rank
    )

    # If our shard has no episodes, return immediately. encode_chunk would
    # also be a no-op but logging here makes the rank diagnostics clearer.
    if not list(view_dir.glob("ep_*")):
        log.info("no episodes assigned; exiting")
        return

    cfg = DinoCacheConfig(**config_kwargs)
    written = encode_chunk(
        staged_chunk_dir=view_dir,
        output_root=Path(output_root),
        modalities=modalities,
        cameras=cameras,
        dinov3_ckpt=Path(dinov3_ckpt) if dinov3_ckpt else None,
        source_fps=source_fps,
        config=cfg,
    )
    # encode_chunk hardcodes file-000.safetensors; rename our outputs to
    # file-{rank:03d}.safetensors so the 8 ranks don't clobber each other.
    for shard_path in written:
        new_name = shard_path.parent / f"file-{rank:03d}.safetensors"
        if shard_path != new_name:
            shard_path.replace(new_name)
    log.info("wrote %d shards", len(written))


def encode_chunk_multigpu(
    staged_chunk_dir: Path,
    output_root: Path,
    modalities: Iterable[str] = ("rgb", "depth", "flow"),
    cameras: Iterable[str] = ("head_rgb", "wrist_rgb"),
    dinov3_ckpt: Optional[Path] = None,
    source_fps: int = 30,
    world_size: int = 0,
    config: DinoCacheConfig | None = None,
) -> None:
    """Run :func:`encode_chunk` sharded across ``world_size`` GPUs via
    ``torch.multiprocessing.spawn``.

    * ``world_size=0`` (default) auto-detects ``torch.cuda.device_count()``
      and falls back to 1 if no CUDAs are visible.
    * Each rank handles episodes ``ep_idx % world_size == rank``.
    * Each rank writes ``file-{rank:03d}.safetensors`` per (cam, mod).

    We do NOT return the list of shards because spawn's children write to
    disk autonomously; callers should glob ``output_root/.../chunk-*/file-*.safetensors``.
    """
    import torch as _torch
    if world_size <= 0:
        world_size = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
    cfg = config or DinoCacheConfig()
    config_kwargs = dict(
        target_hw=cfg.target_hw,
        n_keep_tokens=cfg.n_keep_tokens,
        embed_dim=cfg.embed_dim,
        batch_size=cfg.batch_size,
        cache_fps=cfg.cache_fps,
        fp16=cfg.fp16,
    )

    if world_size == 1:
        # Single-process path: avoid mp.spawn so the placeholder smoke
        # tests (no CUDA) and CI runners stay simple.
        _encode_chunk_worker(
            rank=0,
            world_size=1,
            staged_chunk_dir=str(staged_chunk_dir),
            output_root=str(output_root),
            modalities=tuple(modalities),
            cameras=tuple(cameras),
            dinov3_ckpt=str(dinov3_ckpt) if dinov3_ckpt else None,
            source_fps=source_fps,
            config_kwargs=config_kwargs,
        )
        return

    import torch.multiprocessing as mp
    mp.spawn(
        _encode_chunk_worker,
        args=(
            world_size,
            str(staged_chunk_dir),
            str(output_root),
            tuple(modalities),
            tuple(cameras),
            str(dinov3_ckpt) if dinov3_ckpt else None,
            source_fps,
            config_kwargs,
        ),
        nprocs=world_size,
        join=True,
    )
```

Update the `__all__` line at the bottom:
```python
__all__ = ["DinoCacheConfig", "encode_chunk", "encode_chunk_multigpu"]
```

- [ ] **Step 2: Run the sharding tests — they should pass**

Run: `pytest tests/integration/test_dino_cache_sharding.py -v`
Expected: PASS — both test functions.

- [ ] **Step 3: Re-run the existing unit + integration tests for stage_4**

Run: `pytest tests/ -q -k "stage_4 or dino_cache or smoke_train_cpu"`
Expected: PASS. The single-process path through `encode_chunk` is unchanged; the multi-GPU path is additive.

- [ ] **Step 4: Commit B1 + B2 together**

```bash
git add prep/stage_4_dino_cache.py tests/integration/test_dino_cache_sharding.py
git commit -m "feat(prep): 8-GPU sharding wrapper for stage_4 DINO caching

encode_chunk_multigpu(world_size=N) spawns N workers via
torch.multiprocessing.spawn. Each worker pins to one GPU, sees only
the episodes where ep_idx % world_size == rank (via a transient
symlinked view), and writes file-{rank:03d}.safetensors per (cam, mod).
Reader code is unchanged: it already globs file-*.safetensors.

Includes integration tests that exercise the partitioning logic with
the placeholder (no-encoder) path so they run without GPUs."
```

## Task B3: Real-weights smoke test (skip-gated)

**Files:**
- Create: `tests/integration/test_dino_cache_real_weights.py`

- [ ] **Step 1: Write the test**

Create the file:
```python
# SPDX-License-Identifier: MIT
"""Real-DINOv3-weights smoke for stage_4_dino_cache.

Skipped unless ``USAM_DINOV3_CKPT`` is set in the environment, e.g.::

    USAM_DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m \
        pytest tests/integration/test_dino_cache_real_weights.py

Designed to run inside the prep Docker image where the gated weights
are baked at /opt/dinov3-cache; passing the HF model id resolves
locally because TRANSFORMERS_OFFLINE=1 is set in the image.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

DINOV3_CKPT = os.environ.get("USAM_DINOV3_CKPT")

pytestmark = pytest.mark.skipif(
    not DINOV3_CKPT,
    reason="USAM_DINOV3_CKPT not set; needs a real DINOv3 checkpoint to run",
)


def _build_two_episode_chunk(chunk_dir: Path, num_frames: int = 4) -> None:
    """Two episodes of ``num_frames`` random RGB frames each, 64×64."""
    rng = np.random.default_rng(0)
    for ep_idx in range(2):
        ep_dir = chunk_dir / f"ep_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "camera_head_rgb.npy",
                rng.integers(0, 256, (num_frames, 64, 64, 3), dtype=np.uint8))
        (ep_dir / "meta.json").write_text(json.dumps({"episode_index": ep_idx}))


def test_real_weights_produces_nonzero_shards(tmp_path: Path) -> None:
    """End-to-end: load real DINOv3, encode 2 episodes, assert non-zero output."""
    import torch
    from prep.stage_4_dino_cache import (
        DinoCacheConfig,
        encode_chunk_multigpu,
    )
    from usam.dataloader.feature_cache import read_feature_shard

    staged = tmp_path / "staged"
    output = tmp_path / "out"
    _build_two_episode_chunk(staged)

    world_size = max(1, min(2, torch.cuda.device_count()))
    cfg = DinoCacheConfig(
        target_hw=(448, 448),
        n_keep_tokens=64,
        embed_dim=1024 if "vitl16" in DINOV3_CKPT else 768,
        batch_size=2,
        cache_fps=5,
    )
    encode_chunk_multigpu(
        staged_chunk_dir=staged,
        output_root=output,
        modalities=("rgb",),
        cameras=("head_rgb",),
        dinov3_ckpt=Path(DINOV3_CKPT),
        source_fps=10,
        world_size=world_size,
        config=cfg,
    )
    chunk_dir = output / "head_rgb" / "rgb" / "chunk-000"
    shards = sorted(chunk_dir.glob("file-*.safetensors"))
    assert shards, "no shards written"

    saw_nonzero = False
    for shard in shards:
        ep_to_tensor = read_feature_shard(shard)
        for ep_idx, t in ep_to_tensor.items():
            # [T, n_keep+1, embed_dim]
            assert t.dim() == 3, t.shape
            assert t.shape[1] == cfg.n_keep_tokens + 1, t.shape
            assert t.shape[2] == cfg.embed_dim, t.shape
            assert t.dtype == torch.float16
            if t.abs().sum() > 0:
                saw_nonzero = True
    assert saw_nonzero, "all shards are zero — encoder forward did not run"
```

- [ ] **Step 2: Verify the test skips cleanly without the env var**

Run: `pytest tests/integration/test_dino_cache_real_weights.py -v`
Expected: 1 skipped (with reason `"USAM_DINOV3_CKPT not set..."`).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_dino_cache_real_weights.py
git commit -m "test(prep): real-weights smoke for stage_4 (env-gated, skipped by default)"
```

## Task B4: CLI entry point for stage_4 with --num-gpus

**Files:**
- Modify: `prep/stage_4_dino_cache.py` — add a `tyro` (or `argparse`) main block.

- [ ] **Step 1: Inspect how other prep stages expose CLI**

Run: `grep -n "if __name__\|tyro\|argparse" prep/stage_2b_compute_flow.py prep/stage_2c_compute_depth.py prep/stage_3_canonical.py prep/dispatch.py`
Identify the convention. The existing dispatcher likely calls `python -m prep.stage_X --source SRC --chunk N`. Match that convention.

- [ ] **Step 2: Add a `tyro`-based main block at the bottom of `prep/stage_4_dino_cache.py`**

Append (after `__all__`):

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tyro

    @dataclass
    class _Args:
        source: str
        """Source name (e.g. 'droid', 'bridge'). Used to build paths."""
        chunk: int
        """Chunk index."""
        staged_root: Path = Path(os.environ.get("USAM_SCRATCH", "/scratch/usam")) / "staged"
        """Root containing ``<source>/chunk-NNN/ep_*/`` directories."""
        output_root: Path = Path(os.environ.get("USAM_SCRATCH", "/scratch/usam")) / "dino_cache"
        """Root where shards are written."""
        dinov3_ckpt: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        """HF model id or local path for the DINOv3 checkpoint."""
        source_fps: int = 30
        """Source video FPS; cache stride is computed from this."""
        num_gpus: int = 0
        """0 = auto-detect torch.cuda.device_count()."""
        target_h: int = 448
        target_w: int = 448
        n_keep_tokens: int = 64
        embed_dim: int = 1024
        batch_size: int = 16
        cache_fps: int = 5

    args = tyro.cli(_Args)

    chunk_dir = args.staged_root / args.source / f"chunk-{args.chunk:03d}"
    cfg = DinoCacheConfig(
        target_hw=(args.target_h, args.target_w),
        n_keep_tokens=args.n_keep_tokens,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        cache_fps=args.cache_fps,
    )
    encode_chunk_multigpu(
        staged_chunk_dir=chunk_dir,
        output_root=args.output_root / args.source,
        dinov3_ckpt=Path(args.dinov3_ckpt),
        source_fps=args.source_fps,
        world_size=args.num_gpus,
        config=cfg,
    )
```

Add `import os` at the top of the file (next to the other stdlib imports).

- [ ] **Step 3: Smoke-run the CLI with --help**

Run: `python -m prep.stage_4_dino_cache --help`
Expected: tyro-formatted help block listing all the flags above. No tracebacks.

- [ ] **Step 4: Commit**

```bash
git add prep/stage_4_dino_cache.py
git commit -m "feat(prep): tyro CLI for stage_4_dino_cache with --num-gpus

Allows 'python -m prep.stage_4_dino_cache --source droid --chunk 0
--num-gpus 8' to drive the multi-GPU sharding path. num_gpus=0
auto-detects torch.cuda.device_count()."
```

---

# Wave C — Restyle Dockerfiles + bake DINOv3

Wave C produces three Dockerfiles. Order matters: T1 first (most-used / required-bake test), then T0 (optional bake), then T2 (NGC base). Each Dockerfile is independently testable with `docker build --check` (BuildKit syntax check) before any actual build runs.

## Task C1: Restyle `Dockerfile.prep_a100` (T1) with required DINOv3 bake

**Files:**
- Modify: `docker/Dockerfile.prep_a100` (full rewrite)

- [ ] **Step 1: Replace the file's contents**

Overwrite `docker/Dockerfile.prep_a100` with:

```dockerfile
# =============================================================================
# USAM Prep Docker Image (T1 — 8xA100 Slurm)
# Phase A pipeline: download → flow/depth → DINO caching → HF upload
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

LABEL maintainer="USAM Team"
LABEL description="USAM Phase A prep image: data pipeline + baked DINOv3-ViT-L/16"

# Build-time args (NOT persisted into final image env).
ARG HF_TOKEN
ARG DINOV3_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m

# =============================================================================
# Environment Variables
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Headless rendering (for any LIBERO/sim usage from prep).
ENV PYOPENGL_PLATFORM=egl \
    MUJOCO_GL=egl

WORKDIR /workspace

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl vim ca-certificates build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        ffmpeg \
        libegl1-mesa libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
        libglvnd0 libglvnd-dev \
        libhdf5-dev \
        rsync zstd \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Build Tools
# =============================================================================
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies (USAM base + prep tier)
# =============================================================================
COPY pyproject.toml /tmp/pyproject.toml
COPY requirements/base.txt /tmp/base.txt
COPY requirements/prep.txt /tmp/prep.txt
RUN pip install --no-cache-dir -r /tmp/base.txt \
    && pip install --no-cache-dir -r /tmp/prep.txt

# =============================================================================
# Project Source
# =============================================================================
COPY . /workspace/USAM/
WORKDIR /workspace/USAM
RUN pip install -e .

# =============================================================================
# Pre-download Gated Models (DINOv3-ViT-L/16) — REQUIRED on T1
# =============================================================================
ENV HF_HOME=/opt/dinov3-cache \
    HUGGINGFACE_HUB_CACHE=/opt/dinov3-cache/hub \
    TRANSFORMERS_CACHE=/opt/dinov3-cache/transformers
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -n "$HF_TOKEN" ]; then \
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; \
    elif [ -f /run/secrets/hf_token ]; then \
        export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)"; \
    else \
        echo "ERROR: T1 image requires HF_TOKEN. Pass it via:" >&2; \
        echo "  --build-arg HF_TOKEN=\$HF_TOKEN" >&2; \
        echo "  or DOCKER_BUILDKIT=1 ... --secret id=hf_token,env=HF_TOKEN" >&2; \
        exit 1; \
    fi \
    && mkdir -p "${HF_HOME}" \
    && huggingface-cli download "${DINOV3_MODEL}" \
        --local-dir "/opt/dinov3-cache/${DINOV3_MODEL}" \
        --local-dir-use-symlinks False \
    && python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('${DINOV3_MODEL}'); print(f'✓ DINOv3 baked: hidden={m.config.hidden_size} patch={m.config.patch_size}')" \
    && unset HUGGING_FACE_HUB_TOKEN

# Lock runtime to offline mode — Slurm compute nodes must not call HF.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Hub fast-transfer paths (used by prep/_hub.py + stage_6_upload.py on the
# login node, where OFFLINE=1 is overridden via env at runtime).
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1

# =============================================================================
# Necessary Directories
# =============================================================================
RUN mkdir -p \
    /workspace/USAM/data/staged \
    /workspace/USAM/data/dino_cache \
    /workspace/USAM/logs \
    /workspace/USAM/outputs

# =============================================================================
# Set PYTHONPATH
# =============================================================================
ENV PYTHONPATH=/workspace/USAM

# =============================================================================
# Verify Installation
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" && \
    python -c "import einops; print('✓ Einops OK')" && \
    python -c "import safetensors; print('✓ Safetensors OK')" && \
    python -c "from transformers import AutoModel; print('✓ Transformers OK')" && \
    python -c "from huggingface_hub import HfApi; print('✓ HF Hub OK')" && \
    python -c "import decord, cv2; print('✓ Video IO OK')" && \
    python -c "from usam.encoders.tri_dino import TriDINOTower, TriDinoConfig; print('✓ TriDINOTower OK')" && \
    python -c "from prep import stage_4_dino_cache; print('✓ prep.stage_4_dino_cache OK')" && \
    python -c "from prep import stage_2b_compute_flow, stage_2c_compute_depth; print('✓ prep.stage_2b/2c OK')"

# =============================================================================
# Expose Ports
# =============================================================================
EXPOSE 8080

# =============================================================================
# Default Command
# =============================================================================
CMD ["/bin/bash"]
```

- [ ] **Step 2: Lint the Dockerfile syntax (no actual build yet)**

Run: `docker buildx build --check -f docker/Dockerfile.prep_a100 .`
Expected: no errors. If `buildx` not installed, skip — Step 3 of Task C5 will catch real syntax issues.

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile.prep_a100
git commit -m "feat(docker): restyle T1 prep image + bake DINOv3-ViT-L/16

Adopts the unified Dockerfile style (banner, sections, verify block,
EXPOSE 8080, EGL env, mkdir tree). Switches base from
nvidia/cuda:12.4.1 to pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel.
Adds REQUIRED build-time bake of facebook/dinov3-vitl16-pretrain-lvd1689m
under /opt/dinov3-cache + offline-mode runtime env so Slurm compute
nodes never need an HF token or internet."
```

## Task C2: Restyle `Dockerfile.local_a40` (T0) with optional bake

**Files:**
- Modify: `docker/Dockerfile.local_a40` (full rewrite)

- [ ] **Step 1: Replace the file's contents**

Overwrite `docker/Dockerfile.local_a40` with:

```dockerfile
# =============================================================================
# USAM Local Dev Docker Image (T0 — 8xA40)
# Code dev, unit + integration tests, smoke train, inference smoke ≤10 Hz
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

LABEL maintainer="USAM Team"
LABEL description="USAM local 8xA40 image: tests + smoke train + inference smoke"

# Build-time args (optional on T0 — bake is skipped if HF_TOKEN is empty).
ARG HF_TOKEN
ARG DINOV3_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m

# =============================================================================
# Environment Variables
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

ENV PYOPENGL_PLATFORM=egl \
    MUJOCO_GL=egl

WORKDIR /workspace

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl vim ca-certificates build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        ffmpeg \
        libegl1-mesa libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
        libglvnd0 libglvnd-dev \
        libhdf5-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Build Tools
# =============================================================================
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies (USAM base + train tier)
# =============================================================================
COPY pyproject.toml /tmp/pyproject.toml
COPY requirements/base.txt /tmp/base.txt
COPY requirements/train.txt /tmp/train.txt
RUN pip install --no-cache-dir -r /tmp/base.txt \
    && pip install --no-cache-dir -r /tmp/train.txt

# =============================================================================
# Flash-Attention (required on T0 for perf parity with H200)
# =============================================================================
RUN pip install --no-build-isolation flash-attn==2.6.3

# =============================================================================
# Optional: xFormers (DINOv2 efficiency; safe to fail)
# =============================================================================
RUN pip install --no-cache-dir xformers || \
    echo "Warning: xFormers install failed; runtime will fall back to non-xformers paths"

# =============================================================================
# Project Source
# =============================================================================
COPY . /workspace/USAM/
WORKDIR /workspace/USAM
RUN pip install -e .

# =============================================================================
# Pre-download Gated Models (OPTIONAL on T0)
# =============================================================================
# Bake DINOv3 weights only if HF_TOKEN is provided. Without it, the image
# still builds; runtime falls back to MiniDinoBackbone for unit tests.
ENV HF_HOME=/opt/dinov3-cache \
    HUGGINGFACE_HUB_CACHE=/opt/dinov3-cache/hub \
    TRANSFORMERS_CACHE=/opt/dinov3-cache/transformers
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -n "$HF_TOKEN" ]; then \
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; \
    elif [ -f /run/secrets/hf_token ]; then \
        export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)"; \
    else \
        export HUGGING_FACE_HUB_TOKEN=""; \
    fi; \
    if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
        mkdir -p "${HF_HOME}" && \
        huggingface-cli download "${DINOV3_MODEL}" \
            --local-dir "/opt/dinov3-cache/${DINOV3_MODEL}" \
            --local-dir-use-symlinks False && \
        python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('${DINOV3_MODEL}'); print(f'✓ DINOv3 baked: hidden={m.config.hidden_size} patch={m.config.patch_size}')"; \
    else \
        echo "T0 build: HF_TOKEN not provided; skipping DINOv3 bake (MiniDinoBackbone fallback at runtime)"; \
    fi; \
    unset HUGGING_FACE_HUB_TOKEN

# =============================================================================
# Necessary Directories
# =============================================================================
RUN mkdir -p \
    /workspace/USAM/data/staged \
    /workspace/USAM/data/dino_cache \
    /workspace/USAM/checkpoints \
    /workspace/USAM/logs \
    /workspace/USAM/outputs

# =============================================================================
# Set PYTHONPATH
# =============================================================================
ENV PYTHONPATH=/workspace/USAM

# =============================================================================
# Verify Installation
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import einops, hydra, wandb, h5py, cv2, safetensors; print('✓ Train deps OK')" && \
    python -c "import flash_attn; print(f'✓ flash-attn: {flash_attn.__version__}')" && \
    python -c "from usam.encoders.tri_dino import TriDINOTower, TriDinoConfig; print('✓ TriDINOTower OK')" && \
    python -c "from usam.train import run; print('✓ usam.train OK')" && \
    python -c "from usam.conductor import PlanCache; print('✓ usam.conductor OK')"

EXPOSE 8080

CMD ["/bin/bash"]
```

- [ ] **Step 2: Commit**

```bash
git add docker/Dockerfile.local_a40
git commit -m "feat(docker): restyle T0 local-A40 image with optional DINOv3 bake

Adopts the unified Dockerfile style. Bake is wrapped in an
HF_TOKEN-presence check so devs without HF access can still build the
image (runtime falls back to MiniDinoBackbone for unit tests). Switches
base to pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel."
```

## Task C3: Restyle `Dockerfile.train_h200` (T2) keeping NGC base

**Files:**
- Modify: `docker/Dockerfile.train_h200` (full rewrite)

- [ ] **Step 1: Replace the file's contents**

Overwrite `docker/Dockerfile.train_h200` with:

```dockerfile
# =============================================================================
# USAM Training Docker Image (T2 — 500xH200 burst)
# Phase B pretrain + fine-tune. NGC base for Transformer-Engine + FP8.
# =============================================================================
#
# Base UNCHANGED from nvcr.io/nvidia/pytorch:25.01-py3 — Transformer-Engine
# is pre-built and ABI-matched against this container's CUDA/cuDNN/NCCL.
# Switching to pytorch/pytorch loses TE; do not change without coordination.

FROM nvcr.io/nvidia/pytorch:25.01-py3

LABEL maintainer="USAM Team"
LABEL description="USAM Phase B training image: TE/FP8 + baked DINOv3-ViT-L/16"

# Build-time args (NOT persisted into final image env).
ARG HF_TOKEN
ARG DINOV3_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m

# =============================================================================
# Environment Variables
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

ENV PYOPENGL_PLATFORM=egl \
    MUJOCO_GL=egl

WORKDIR /workspace

# =============================================================================
# System Dependencies (most ship in the NGC base)
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
        libegl1-mesa libegl1-mesa-dev libglvnd0 \
        libhdf5-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Build Tools
# =============================================================================
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies (USAM base + train tier)
# =============================================================================
# The NGC image ships torch + transformer-engine. requirements/base.txt
# pins torch==2.6.0 — pip will see this satisfied or upgrade only if needed.
# If pip tries to upgrade torch in a way that breaks TE ABI, drop the torch
# pin from requirements/base.txt.
COPY pyproject.toml /tmp/pyproject.toml
COPY requirements/base.txt /tmp/base.txt
COPY requirements/train.txt /tmp/train.txt
RUN pip install --no-cache-dir -r /tmp/base.txt \
    && pip install --no-cache-dir -r /tmp/train.txt

# =============================================================================
# Flash-Attention (required on T2)
# =============================================================================
RUN pip install --no-build-isolation flash-attn==2.6.3

# =============================================================================
# Project Source
# =============================================================================
COPY . /workspace/USAM/
WORKDIR /workspace/USAM
RUN pip install -e .

# =============================================================================
# Pre-download Gated Models (DINOv3-ViT-L/16) — REQUIRED on T2
# =============================================================================
ENV HF_HOME=/opt/dinov3-cache \
    HUGGINGFACE_HUB_CACHE=/opt/dinov3-cache/hub \
    TRANSFORMERS_CACHE=/opt/dinov3-cache/transformers
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -n "$HF_TOKEN" ]; then \
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; \
    elif [ -f /run/secrets/hf_token ]; then \
        export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)"; \
    else \
        echo "ERROR: T2 image requires HF_TOKEN. Pass it via --build-arg HF_TOKEN=\$HF_TOKEN" >&2; \
        exit 1; \
    fi \
    && mkdir -p "${HF_HOME}" \
    && huggingface-cli download "${DINOV3_MODEL}" \
        --local-dir "/opt/dinov3-cache/${DINOV3_MODEL}" \
        --local-dir-use-symlinks False \
    && python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('${DINOV3_MODEL}'); print(f'✓ DINOv3 baked: hidden={m.config.hidden_size} patch={m.config.patch_size}')" \
    && unset HUGGING_FACE_HUB_TOKEN

ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# NCCL tuning for H200 NVLink + IB fabric (overridable at runtime).
ENV NCCL_IB_HCA=mlx5 \
    NCCL_SOCKET_IFNAME=^lo,docker0

# =============================================================================
# Necessary Directories
# =============================================================================
RUN mkdir -p \
    /workspace/USAM/data/staged \
    /workspace/USAM/checkpoints \
    /workspace/USAM/logs \
    /workspace/USAM/outputs

# =============================================================================
# Set PYTHONPATH
# =============================================================================
ENV PYTHONPATH=/workspace/USAM

# =============================================================================
# Verify Installation
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import transformer_engine.pytorch as te; print('✓ Transformer-Engine OK')" && \
    python -c "import flash_attn; print(f'✓ flash-attn: {flash_attn.__version__}')" && \
    python -c "import einops, hydra, wandb, h5py, cv2, safetensors; print('✓ Train deps OK')" && \
    python -c "from usam.encoders.tri_dino import TriDINOTower, TriDinoConfig; print('✓ TriDINOTower OK')" && \
    python -c "from usam.train import run; print('✓ usam.train OK')"

EXPOSE 8080

CMD ["/bin/bash"]
```

- [ ] **Step 2: Commit**

```bash
git add docker/Dockerfile.train_h200
git commit -m "feat(docker): restyle T2 train-H200 image + bake DINOv3-ViT-L/16

Adopts the unified Dockerfile style while keeping the NGC base
(nvcr.io/nvidia/pytorch:25.01-py3) for Transformer-Engine FP8 support.
Adds REQUIRED build-time bake of facebook/dinov3-vitl16-pretrain-lvd1689m
+ offline-mode runtime so H200 nodes never need an HF token."
```

## Task C4: Update `docker/README.md`

**Files:**
- Modify: `docker/README.md`

- [ ] **Step 1: Read the existing file (53 lines, already known)** and replace the "Build" section + "Layer ordering" section with:

Replace the `## Build` section block (lines 18-25) with:
```markdown
## Build

All three images bake (or optionally bake, on T0) the gated DINOv3 weights
at build time, so compute nodes never need an HF token at runtime. You
must pass `HF_TOKEN` as a build arg (simple) or via BuildKit `--secret`
(strict; never appears in image layers).

```bash
# Simple build-arg path
export HF_TOKEN=hf_...   # token with access to facebook/dinov3-vitl16-pretrain-lvd1689m

docker build --build-arg HF_TOKEN=$HF_TOKEN \
             -f docker/Dockerfile.local_a40 -t usam:local-a40 .

docker build --build-arg HF_TOKEN=$HF_TOKEN \
             -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .

docker build --build-arg HF_TOKEN=$HF_TOKEN \
             -f docker/Dockerfile.train_h200 -t usam:train-h200 .

# Strict BuildKit secret path (token never embedded as an ARG)
DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,env=HF_TOKEN \
    -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .
```

T0 (`local_a40`) treats `HF_TOKEN` as **optional** — the build succeeds
without it and runtime falls back to `MiniDinoBackbone` for unit tests.
T1 and T2 require it; the build fails fast otherwise.

For the full operator runbook (build → smoke → ship → sbatch), see
[`docs/HOWTO_PREP_DINOV3.md`](../docs/HOWTO_PREP_DINOV3.md).
```

- [ ] **Step 2: Commit**

```bash
git add docker/README.md
git commit -m "docs(docker): build instructions for the HF_TOKEN bake flow

Documents both --build-arg and BuildKit --secret paths, notes T0's
optional-bake behavior vs. T1/T2's required-bake, links to the
upcoming HOWTO_PREP_DINOV3.md operator runbook."
```

## Task C5: Build T1 image on the A40 box (verification)

This task produces a real image; the user should oversee it. The plan supplies the exact command and the expected output.

- [ ] **Step 1: Confirm `HF_TOKEN` is exported**

Run: `echo "HF_TOKEN length: ${#HF_TOKEN}"`
Expected: a number > 0 (the token is set in your shell). If 0, export it: `export HF_TOKEN=<your-hf-token-with-dinov3-access>`.

- [ ] **Step 2: Build the T1 image**

```bash
cd /localhome/local-chrislin/USAM
DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,env=HF_TOKEN \
    -f docker/Dockerfile.prep_a100 \
    -t usam:prep-a100 .
```

Expected output (key lines):
```
...
=> [9/N] RUN ... huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m ...
=> [9/N]   ✓ DINOv3 baked: hidden=1024 patch=16
...
=> [N/N] RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
=> [N/N]   ✓ PyTorch: 2.6.0
=> [N/N]   ✓ CUDA available: True   (or False; not fatal at build time)
=> [N/N]   ✓ TriDINOTower OK
=> [N/N]   ✓ prep.stage_4_dino_cache OK
...
naming to docker.io/library/usam:prep-a100 done
```

If build fails:
- "Repository not found" → `HF_TOKEN` lacks access. Verify on huggingface.co.
- "ImportError: TriDINOTower" → Wave A wasn't completed. Run `pytest tests/unit/test_tri_dino.py` first.
- "no space left" → run `docker system prune` and retry.

- [ ] **Step 3: Confirm image size and that DINOv3 is offline-loadable**

```bash
docker images usam:prep-a100
docker run --rm usam:prep-a100 \
  python -c "
import os
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
assert m.config.patch_size == 16 and m.config.hidden_size == 1024
print('offline load: ok, params:', sum(p.numel() for p in m.parameters()))
"
```
Expected: image size ~5–6 GB; `offline load: ok, params: ~3.04e8`.

- [ ] **Step 4: No commit — this task is a verification gate.**

If Step 2 + Step 3 succeeded, proceed to Wave D. If anything failed, fix and re-run Wave C from the broken task.

---

# Wave D — A40 smoke test (manual verification gate)

**This wave runs commands; it produces no commits except the final HOWTO. Treat each step as a checkbox the operator marks off.**

## Task D1: 8-GPU TriDINOTower load smoke

- [ ] **Step 1: Run the 8-GPU forward test**

```bash
docker run --rm --gpus all \
  -v /localhome/local-chrislin/USAM:/workspace/USAM \
  usam:prep-a100 \
  python -c "
import torch, torch.multiprocessing as mp
def fwd(rank):
    torch.cuda.set_device(rank)
    from usam.encoders.tri_dino import TriDinoConfig, TriDINOTower
    cfg = TriDinoConfig(
        dinov3_ckpt='facebook/dinov3-vitl16-pretrain-lvd1689m',
        dinov3_arch='vit_l_16', image_size=448, patch_size=16, embed_dim=1024,
    )
    m = TriDINOTower(cfg).cuda(rank).eval()
    x = torch.randn(2, 3, 448, 448, device=f'cuda:{rank}')
    with torch.no_grad():
        out = m(x, modality='rgb')
    print(f'rank {rank}: {tuple(out.shape)}', flush=True)
mp.spawn(fwd, nprocs=8, join=True)
"
```
Expected: 8 lines `rank N: (2, 789, 1024)` (or same with whatever the loaded backbone reports for `1 + num_register_tokens + 784`). All 8 should print without error.

If any rank crashes:
- "CUDA out of memory" → unlikely on A40 48 GB but possible if other processes are using GPUs. Run `nvidia-smi` and free GPUs first.
- "no kernel image is available" → mismatched torch / driver. Check `nvidia-smi` and the image's torch CUDA version.

## Task D2: End-to-end DROID chunk 0

- [ ] **Step 1: Verify TFDS / DROID prereqs**

```bash
docker run --rm --gpus all \
  -v /localhome/local-chrislin/USAM:/workspace/USAM \
  -v /scratch/$USER/usam-smoke:/scratch/usam-smoke \
  -e USAM_SCRATCH=/scratch/usam-smoke \
  usam:prep-a100 \
  python -c "
import tensorflow_datasets as tfds
b = tfds.builder('droid_100', data_dir='/scratch/usam-smoke/tfds')
print('builder ok:', b.info.name)
"
```
Expected: `builder ok: droid_100`. If this fails because the TFDS data dir is empty, the `stage_2a_to_lerobot.droid` stage will trigger the download — that's expected for chunk 0.

- [ ] **Step 2: Run the full pipeline**

```bash
docker run --rm --gpus all \
  --shm-size=64g \
  -v /localhome/local-chrislin/USAM:/workspace/USAM \
  -v /scratch/$USER/usam-smoke:/scratch/usam-smoke \
  -e USAM_SCRATCH=/scratch/usam-smoke \
  usam:prep-a100 \
  bash -c '
    set -e
    cd /workspace/USAM
    echo "==== stage_2a (download + LeRobot stage) ===="
    python -m prep.stage_2a_to_lerobot.droid --chunk 0
    echo "==== stage_2b (flow) ===="
    python -m prep.stage_2b_compute_flow --source droid --chunk 0
    echo "==== stage_2c (depth) ===="
    python -m prep.stage_2c_compute_depth --source droid --chunk 0
    echo "==== stage_3 (canonical actions) ===="
    python -m prep.stage_3_canonical --source droid --chunk 0
    echo "==== stage_4 (DINO cache, 8 GPUs) ===="
    python -m prep.stage_4_dino_cache --source droid --chunk 0 --num-gpus 8
  '
```

(Each stage's exact CLI flags must match its `tyro` interface. If a stage uses different flag names — e.g. `--src` instead of `--source` — adjust here. Run `python -m prep.stage_2a_to_lerobot.droid --help` etc. to verify before kicking off the full pipeline.)

Expected: each stage prints its banner; no tracebacks; total wall time < 25 min on 8xA40.

While stage_4 runs, in another terminal:
```bash
watch -n 1 nvidia-smi
```
Expected: at least 6 of the 8 GPUs at >50% utilization during stage_4. (DROID chunks have ~50 episodes; with world_size=8 every rank should get ~6 episodes.)

- [ ] **Step 3: Verify the on-disk output**

```bash
ls -la /scratch/$USER/usam-smoke/dino_cache/droid/head_rgb/rgb/chunk-000/
```
Expected: `file-000.safetensors` through `file-007.safetensors`. (Fewer files if chunk has < 8 episodes — note the count; should still cover all episode_idxs.)

```bash
docker run --rm \
  -v /scratch/$USER/usam-smoke:/scratch/usam-smoke \
  usam:prep-a100 \
  python -c "
from pathlib import Path
import torch
from usam.dataloader.feature_cache import read_feature_shard

shards = sorted(Path('/scratch/usam-smoke/dino_cache/droid/head_rgb/rgb/chunk-000').glob('file-*.safetensors'))
print(f'{len(shards)} shards')
seen = set()
nonzero = 0
for s in shards:
    d = read_feature_shard(s)
    for ep, t in d.items():
        ep_int = int(ep)
        assert ep_int not in seen, f'duplicate ep {ep_int}'
        seen.add(ep_int)
        assert t.shape[1] == 65, t.shape
        assert t.shape[2] == 1024, t.shape
        if t.abs().sum() > 0:
            nonzero += 1
print(f'{nonzero} non-zero episodes; episode_idxs covered: {sorted(seen)}')
assert nonzero == len(seen), 'some episodes are zero — encoder did not run for them'
"
```
Expected: prints "N shards", a non-zero count equal to the unique episode count, no `AssertionError`.

## Task D3: Pytest gate inside the container

- [ ] **Step 1: Run the always-on sharding test**

```bash
docker run --rm --gpus all \
  -v /localhome/local-chrislin/USAM:/workspace/USAM \
  usam:prep-a100 \
  pytest tests/integration/test_dino_cache_sharding.py -v
```
Expected: 2 passed.

- [ ] **Step 2: Run the real-weights test (env-gated)**

```bash
docker run --rm --gpus all \
  -v /localhome/local-chrislin/USAM:/workspace/USAM \
  -e USAM_DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m \
  usam:prep-a100 \
  pytest tests/integration/test_dino_cache_real_weights.py -v
```
Expected: 1 passed.

- [ ] **Step 3: If both pass, Wave D is GREEN. Proceed to Wave E.**

If anything in Tasks D1–D3 fails, do not proceed; debug.

---

# Wave E — Slurm rollout + runbook

## Task E1: Build SIF on the A40 box

- [ ] **Step 1: Build the SIF**

```bash
mkdir -p /scratch/$USER
singularity build /scratch/$USER/usam_prep.sif docker-daemon://usam:prep-a100
```
Expected: `INFO:    Build complete: /scratch/$USER/usam_prep.sif`. Size ~5–6 GB.

- [ ] **Step 2: Verify the SIF is self-contained**

```bash
singularity exec --nv /scratch/$USER/usam_prep.sif \
  python -c "
import os; os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
print('SIF self-contained: ok')
"
```
Expected: `SIF self-contained: ok`. No commit.

## Task E2: Tidy `slurm/env.sh`

**Files:**
- Modify: `slurm/env.sh:14-21` (the `HUGGINGFACE_TOKEN` requirement comment)

- [ ] **Step 1: Replace the comment block**

Current text (lines 14-21):
```
#   HUGGINGFACE_TOKEN HF API token (read from a file in ~/.cache, never
#                     committed to the repo). Only needed on login nodes
#                     running the CommitScheduler — Slurm jobs themselves
#                     do not talk to the Hub.
```

Replace with:
```
#   HUGGINGFACE_TOKEN HF API token. Required ONLY on the login node
#                     running stage_6's CommitScheduler upload daemon.
#                     Slurm compute nodes do NOT need it: gated weights
#                     (DINOv3-ViT-L/16) are baked into the prep SIF, and
#                     compute jobs run with HF_HUB_OFFLINE=1.
```

- [ ] **Step 2: Commit**

```bash
git add slurm/env.sh
git commit -m "docs(slurm): clarify that compute nodes don't need HUGGINGFACE_TOKEN

Gated DINOv3 weights are baked into the prep SIF (Wave C) and compute
jobs run with HF_HUB_OFFLINE=1. The token is only needed on the login
node for the upload daemon."
```

## Task E3: Ship SIF to Slurm (manual operator step)

This task is the operator's handoff to Slurm. The plan documents the command; the user runs it from a shell with cluster access.

- [ ] **Step 1: rsync the SIF**

```bash
rsync -avh --progress \
  /scratch/$USER/usam_prep.sif \
  $USER@<slurm-login>:$HOME/usam_prep.sif
```
Replace `<slurm-login>` with the actual login host.

- [ ] **Step 2: On the login node, set USAM env vars**

```bash
ssh $USER@<slurm-login>
export USAM_REPO=$HOME/USAM
export USAM_SIF=$HOME/usam_prep.sif
echo "export USAM_REPO=$USAM_REPO" >> ~/.bashrc
echo "export USAM_SIF=$USAM_SIF"  >> ~/.bashrc
```

- [ ] **Step 3: One-chunk shakedown (optional but recommended)**

```bash
sbatch slurm/job.sbatch stage_4_dino_cache droid 0
squeue -u $USER
tail -f logs/usam-<jobid>.out
```
Expected: same pass criteria as Task D2 / D3 but on A100 hardware.

No commit — this is operational, not code.

## Task E4: Write `docs/HOWTO_PREP_DINOV3.md` from verified commands

**Files:**
- Create: `docs/HOWTO_PREP_DINOV3.md`

- [ ] **Step 1: Copy the verified commands from Waves C–E into a new file**

Create `docs/HOWTO_PREP_DINOV3.md`:

```markdown
# HOWTO — Reproduce the DINOv3 Prep Image on a Fresh A100 Host

This is the operator runbook for building the USAM Phase A prep image
with `facebook/dinov3-vitl16-pretrain-lvd1689m` baked in, smoke-testing
on a multi-GPU host, and submitting to Slurm. All commands here have
been verified end-to-end on the local 8xA40 box (see plan
`docs/superpowers/plans/2026-05-09-dinov3-prep-image.md`).

---

## 0. Prereqs

- Docker 20+ with BuildKit (`DOCKER_BUILDKIT=1`).
- HF token with access to `facebook/dinov3-vitl16-pretrain-lvd1689m`,
  exported as `HF_TOKEN`.
- ~10 GB free for the build context + image.
- ~80 GB free on `/scratch/$USER` for the smoke chunk + DINO cache.
- 8x A40 (smoke) or 8x A100 (prod).
- nvidia-container-toolkit installed.

## 1. Build the prep image

```bash
cd /path/to/USAM
DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,env=HF_TOKEN \
    -f docker/Dockerfile.prep_a100 \
    -t usam:prep-a100 .
```

Look for `✓ DINOv3 baked: hidden=1024 patch=16` in the build output.
Image size: ~5–6 GB.

## 2. Image self-test (no GPU needed)

```bash
docker run --rm usam:prep-a100 \
  python -c "
import os
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
assert m.config.patch_size == 16 and m.config.hidden_size == 1024
print('offline load: ok')
"
```

Expected: `offline load: ok`.

## 3. End-to-end DROID chunk 0 (~25 min on 8xA40)

```bash
docker run --rm --gpus all --shm-size=64g \
  -v /path/to/USAM:/workspace/USAM \
  -v /scratch/$USER/usam-smoke:/scratch/usam-smoke \
  -e USAM_SCRATCH=/scratch/usam-smoke \
  usam:prep-a100 \
  bash -c '
    set -e
    cd /workspace/USAM
    python -m prep.stage_2a_to_lerobot.droid --chunk 0
    python -m prep.stage_2b_compute_flow --source droid --chunk 0
    python -m prep.stage_2c_compute_depth --source droid --chunk 0
    python -m prep.stage_3_canonical --source droid --chunk 0
    python -m prep.stage_4_dino_cache --source droid --chunk 0 --num-gpus 8
  '
```

Expected: each stage exits 0; stage_4 produces 8 shard files
under `/scratch/$USER/usam-smoke/dino_cache/droid/<cam>/<mod>/chunk-000/`.

## 4. Pytest gate

```bash
# Always runs (no real weights needed)
docker run --rm --gpus all \
  -v /path/to/USAM:/workspace/USAM \
  usam:prep-a100 \
  pytest tests/integration/test_dino_cache_sharding.py -v

# Real weights (env-gated)
docker run --rm --gpus all \
  -v /path/to/USAM:/workspace/USAM \
  -e USAM_DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m \
  usam:prep-a100 \
  pytest tests/integration/test_dino_cache_real_weights.py -v
```

Expected: 2 passed in the first run, 1 passed in the second.

## 5. Build the Singularity SIF

```bash
mkdir -p /scratch/$USER
singularity build /scratch/$USER/usam_prep.sif docker-daemon://usam:prep-a100
```

## 6. Ship to Slurm

```bash
rsync -avh --progress /scratch/$USER/usam_prep.sif \
  $USER@<slurm-login>:$HOME/usam_prep.sif

# On the login node:
export USAM_REPO=$HOME/USAM
export USAM_SIF=$HOME/usam_prep.sif
```

## 7. Submit one chunk to Slurm

```bash
sbatch slurm/job.sbatch stage_4_dino_cache droid 0
squeue -u $USER
tail -f logs/usam-*.out
```

## 8. Re-baking with a different DINOv3 variant

```bash
DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,env=HF_TOKEN \
    --build-arg DINOV3_MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m \
    -f docker/Dockerfile.prep_a100 \
    -t usam:prep-a100-vitb .
```

## Troubleshooting

- `"Repository not found"` during build → token lacks access.
  Verify access at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m.
- `"CUDA out of memory"` on stage_4 → another process is using GPUs;
  run `nvidia-smi` and free them.
- All shards zero → encoder did not run on any rank; check `--num-gpus`
  matches what's visible inside the container.
- Build fails on `transformers.AutoModel.from_pretrained(...)` →
  `transformers==4.57.0` may not yet recognize this DINOv3 variant.
  Upgrade pin in `requirements/base.txt` and rebuild.
```

- [ ] **Step 2: Commit**

```bash
git add docs/HOWTO_PREP_DINOV3.md
git commit -m "docs: HOWTO_PREP_DINOV3 runbook for fresh A100 reproduction

Records the verified end-to-end commands from Waves C–E (build, smoke,
SIF, ship, sbatch). Linked from docker/README.md."
```

## Task E5: Spec correction commit (stage names)

**Files:**
- Modify: `docs/superpowers/specs/2026-05-09-dinov3-prep-image-design.md`

- [ ] **Step 1: Find every reference to `stage_3_flow_depth` in the spec**

Run: `grep -n "stage_3_flow_depth" docs/superpowers/specs/2026-05-09-dinov3-prep-image-design.md`

- [ ] **Step 2: Replace with `stage_2b_compute_flow + stage_2c_compute_depth`**

For each hit, the actual repo splits flow and depth into separate stages. Update the §1 motivation, §2 in-scope, §3.3, and §5 Step 4 accordingly.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-09-dinov3-prep-image-design.md
git commit -m "docs(spec): correct stage names — 2b (flow) + 2c (depth) + 3 (canonical)

The repo splits flow and depth into separate stages
(stage_2b_compute_flow, stage_2c_compute_depth), with stage_3_canonical
doing action canonicalization. Original spec used a conceptual
'stage_3_flow_depth' that doesn't exist in the code."
```

---

# Self-Review

After all five waves are complete:

**Spec coverage:**
- §3.1 Configs → Tasks A4 (3 YAMLs).
- §3.2 TriDinoConfig → Tasks A1, A2.
- §3.3 stage_4 (target_hw, embed_dim, multigpu) → Tasks A3, B2, B4.
- §3.3 stage_2b/2c audit → handled inside D2 (the smoke runs flow/depth as-is; if they're single-GPU on the A40, they still complete, and the multi-GPU push for those stages is deferred since they aren't on the critical path for the spec's pass criteria).
- §3.4 Tests → Tasks B1 (sharding), B3 (real-weights).
- §4 Dockerfiles + bake → Tasks C1 (T1), C2 (T0), C3 (T2), C4 (README).
- §5 A40 smoke → Tasks D1, D2, D3 (and C5 builds the image they run on).
- §6 Slurm rollout + runbook → Tasks E1, E2, E3, E4.
- §7 Acceptance → covered by Wave A green tests + Wave D green smoke + Wave E artifacts shipped.

**Placeholder scan:** No "TBD", "implement later", or "fill in details" appear above. Every code block is complete.

**Type consistency:** `encode_chunk_multigpu`'s signature is consistent across Task B2 (definition) and Tasks B3, B4 (callers). `DinoCacheConfig` adds a single new field `embed_dim: int = 768` consistently across A3/A6/B1/B2/B3.

**Known deferred work (not blocking spec acceptance):**
- Multi-GPU wrappers for `stage_2b_compute_flow` and `stage_2c_compute_depth`. Spec §3.3 mentions this conditionally; the smoke can complete in 25 min on 8xA40 with single-GPU flow/depth as the bottleneck. If wall time exceeds budget at Slurm scale, file a follow-up.
- Hash-pinning DINOv3 model revision via `--build-arg DINOV3_MODEL_REVISION=<sha>`. Spec §2 explicitly out-of-scope.
