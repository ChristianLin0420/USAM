# Spec — DINOv3-ViT-L/16 prep image + 8-GPU stage_4

**Status:** Approved (sections 1–5)
**Author:** chrislin@nvidia.com (with Claude)
**Date:** 2026-05-09
**Targets:** `docker/`, `prep/stage_4_dino_cache.py`, `usam/encoders/tri_dino.py`, `configs/model/*.yaml`, `configs/train/adapter_pretrain.yaml`, `slurm/env.sh`, `tests/integration/`, `docs/`

---

## 1. Motivation

The user has been granted HuggingFace access to `facebook/dinov3-vitl16-pretrain-lvd1689m`. The current USAM repo references `/datasets/dinov3/dinov3-vitl14-pretrain-lvd1689m` (patch-14) at local filesystem paths and has a single-process stage_4 DINO-cache that allocates 8 A100s on Slurm but only uses one. This spec covers:

- A clean **patch-14 → patch-16 cutover** across configs and code, anchored on the gated HF model id.
- **Baking the gated weights into all three Docker images** so Slurm compute nodes never need an HF token or internet.
- **Multi-GPU sharding** for `stage_4_dino_cache` (and `stage_3_flow_depth` if it's also single-GPU) so all 8 GPUs of a Slurm allocation actually do work.
- A **uniform Dockerfile style** across T0/T1/T2 (banner, sections, verify block, pre-download, headless rendering env), modeled on the Geo-Flow VLA reference image.
- An **A40 acceptance gate**: end-to-end DROID chunk 0 (stages 2a + 3 + 4) runs green on the local 8xA40 box before the `.sif` is shipped to Slurm.
- A **reproduction runbook** (`docs/HOWTO_PREP_DINOV3.md`) written *after* smoke from the actually-verified commands.

## 2. Scope

**In scope:**
1. Patch-16 cutover (configs, `usam/encoders/tri_dino.py`, `prep/stage_4_dino_cache.py`).
2. Multi-GPU wrapper for `prep/stage_4_dino_cache.py` (and audit `stage_3_flow_depth.py`; same wrapper if needed).
3. Restyled `docker/Dockerfile.local_a40`, `docker/Dockerfile.prep_a100`, `docker/Dockerfile.train_h200` in the unified format. T1 + T2 (and T0) bake DINOv3-ViT-L/16 at build time.
4. `docker/README.md` updated with `--build-arg HF_TOKEN=...` instructions and a link to the runbook.
5. `slurm/env.sh`: drop the `HUGGINGFACE_TOKEN` requirement note for compute jobs.
6. New integration tests:
   - `tests/integration/test_dino_cache_sharding.py` — proves rank partitioning with a mocked encoder; runs always.
   - `tests/integration/test_dino_cache_real_weights.py` — runs only when `USAM_DINOV3_CKPT` is set.
7. New runbook `docs/HOWTO_PREP_DINOV3.md`, written from the *verified* A40 commands after smoke passes.

**Out of scope:**
- Re-running prep on data already cached at patch-14. (Verified: no populated DINO caches exist; only zero-tensor placeholders in golden fixtures.)
- New prep stages, dispatcher changes, upload-daemon changes.
- Live Slurm submission. The `.sif` ship + first sbatch is operator-driven, not part of this spec's pass criteria.
- Updating training tier configs beyond ckpt-path swap (no architecture changes from this spec).
- Hash-pinning the DINOv3 model revision via `--build-arg DINOV3_MODEL_REVISION=<sha>`. Easy to add later; not required now.

**Assumptions to verify during implementation:**
- `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` exists on Docker Hub. Fallback: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` and let `requirements/base.txt`'s `torch==2.6.0` pin upgrade it.
- `facebook/dinov3-vitb16-pretrain-lvd1689m` exists and the user has access (used by the smoke config). If not, smoke config falls back to ViT-L/16 too.
- ViT-L/16 `hidden_size=1024`, `patch_size=16`. Verified at build time in the Dockerfile's `from_pretrained()` smoke load.
- DROID TFDS access works from inside the container on the A40 box.
- ~80 GB free on `/scratch/$USER` for staged DROID chunk 0.

## 3. Code & config changes

### 3.1 Configs

`configs/model/usam_1_4b.yaml` (production, ViT-L):
```yaml
dinov3_ckpt: facebook/dinov3-vitl16-pretrain-lvd1689m
dinov3_arch: vit_l_16
```

`configs/model/usam_350m_smoke.yaml`, `configs/train/adapter_pretrain.yaml` (smoke, ViT-B):
```yaml
dinov3_ckpt: facebook/dinov3-vitb16-pretrain-lvd1689m
dinov3_arch: vit_b_16
```

### 3.2 `usam/encoders/tri_dino.py`

- `TriDinoConfig.dinov3_arch` default: `"vit_b_14"` → `"vit_b_16"`.
- `TriDinoConfig.patch_size` default: `14` → `16`.
- `TriDinoConfig.image_size` default: `378` → `448`.
- `MiniDinoBackbone` defaults updated to match.
- New runtime assertion: `assert config.image_size % config.patch_size == 0`.
- Verify `config.embed_dim` equals the loaded HF model's `hidden_size`; raise on mismatch (ViT-L/16 → 1024, ViT-B/16 → 768).

### 3.3 `prep/stage_4_dino_cache.py`

- `DinoCacheConfig.target_hw`: `(378, 378)` → `(448, 448)`.
- `n_keep_tokens=64` retained. Pooling op: `F.adaptive_avg_pool2d(features, (8, 8))` over the 28×28 grid → 64 tokens.
- `_load_tri_dino` `dinov3_arch` default → `"vit_b_16"`.
- Parameterize zero-tensor placeholder by `embed_dim` (currently hardcoded 768).
- New `encode_chunk_multigpu(...)` wrapper:
  - Spawns `world_size` workers with `torch.multiprocessing.spawn`.
  - Each worker pins to one GPU (`torch.cuda.set_device(rank)`) and loads its own DINOv3 copy.
  - Sharding: `episodes` filtered by `ep_idx % world_size == rank`. Disjoint, no merge.
  - Output layout: rank-N writes `chunk-XXX/file-{N:03d}.safetensors`. Existing reader globs `file-*.safetensors`; no reader change needed.
  - CLI flag `--num-gpus N` (default `auto` → `torch.cuda.device_count()`; tests pass `1` for determinism).
- Audit `prep/stage_3_flow_depth.py`. If single-GPU, apply the same `mp.spawn` pattern.

### 3.4 Tests

- `tests/integration/test_dino_cache_sharding.py`: world_size=4, 10 synthetic episodes, mocked encoder → 4 shard files, episode partition is disjoint and complete.
- `tests/integration/test_dino_cache_real_weights.py`: `pytest.mark.skipif(not os.environ.get("USAM_DINOV3_CKPT"))` — runs `--num-gpus min(2, torch.cuda.device_count())` on `tiny_droid` fixture, asserts non-zero shards with shape `[*, 65, 1024]`.

## 4. Dockerfile changes (style + bake)

**Common style across all three** (from the Geo-Flow VLA reference):
- Banner header + `LABEL maintainer/description`.
- Section dividers `# ===...===`.
- `WORKDIR /workspace` then `WORKDIR /workspace/USAM`.
- Headless-rendering env (`PYOPENGL_PLATFORM=egl`, `MUJOCO_GL=egl`, `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute`).
- `TOKENIZERS_PARALLELISM=false`.
- `mkdir -p` for `data/`, `checkpoints/`, `logs/`, `outputs/`.
- Verify-installation block (`python -c "import X; print('✓ X OK')"`).
- `EXPOSE 8080` for wandb.
- `|| echo "Warning: ..."` fallback **only for optional deps**.

**Per-tier base + extras:**

| Tier | Base | Tier extras | DINOv3 bake |
|---|---|---|---|
| T0 local_a40 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | `requirements/train.txt`, `flash-attn==2.6.3 --no-build-isolation`, `xformers` (optional `\|\| echo`) | **Optional**: skipped if `HF_TOKEN` is empty (build still succeeds; runtime falls back to `MiniDinoBackbone` for unit tests) |
| T1 prep_a100 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | `requirements/prep.txt`, `rsync` + `zstd` for shard staging | **Required**: build fails if `HF_TOKEN` missing |
| T2 train_h200 | `nvcr.io/nvidia/pytorch:25.01-py3` (UNCHANGED — TE/FP8) | `requirements/train.txt`, `flash-attn==2.6.3 --no-build-isolation`, TE check in verify block | **Required**: build fails if `HF_TOKEN` missing |

**Bake mechanism (T0/T1/T2):**
- `ARG HF_TOKEN` + `ARG DINOV3_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m`.
- `RUN --mount=type=secret,id=hf_token,required=false ...` accepts either build-arg or BuildKit secret.
- Downloads to `/opt/dinov3-cache` via `huggingface-cli download`.
- Build-time `from_pretrained()` smoke load — fails the build if the bake didn't take. **(T1/T2 only — required tiers.)**
- For T0, the bake step is wrapped in `if [ -n "$HF_TOKEN" ]; then ... else echo "T0 build: HF_TOKEN not provided, skipping DINOv3 bake (MiniDinoBackbone fallback at runtime)"; fi` so it builds successfully without a token.
- `unset HUGGING_FACE_HUB_TOKEN` after download (no token in any image layer's env).
- T1/T2 final images set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`. T0 leaves these unset (so a dev can still pull other weights interactively if needed).

**Build commands (added to `docker/README.md`):**
```bash
# Build-arg path
docker build --build-arg HF_TOKEN=$HF_TOKEN \
             -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .

# BuildKit secret path (token never in build args / image layers)
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .
```

**Image size impact:** ~3 GB → ~5–6 GB (DINOv3 ViT-L weights ≈ 1.2 GB). Acceptable on Slurm; longer first push.

## 5. A40 smoke procedure (the acceptance gate)

Run on the local 8xA40 box. Total wall time target: ≤ 25 min.

1. **Build T1.** `docker build --build-arg HF_TOKEN=$HF_TOKEN -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .` — must succeed including the in-build `✓ DINOv3 baked: hidden=1024 patch=16` line.
2. **Image self-test.** Offline `from_pretrained()` returns the model. Confirms `OFFLINE=1` doesn't break load.
3. **8-GPU TriDINOTower load.** `mp.spawn(nprocs=8)` → each rank prints `[2, 1+R+784, 1024]` shape on its GPU.
4. **End-to-end DROID chunk 0.** Inside the container with `--gpus all`:
   ```bash
   python -m prep.stage_2a_to_lerobot.droid --chunk 0 --num-workers 16
   python -m prep.stage_3_flow_depth      --source droid --chunk 0 --num-gpus 8
   python -m prep.stage_4_dino_cache      --source droid --chunk 0 --num-gpus 8 \
       --dinov3-ckpt facebook/dinov3-vitl16-pretrain-lvd1689m
   ```
   (Exact flag names align with each stage's `tyro` interface during impl.)
5. **Pytest gate.** `pytest tests/integration/test_dino_cache_sharding.py` (always); `USAM_DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m pytest tests/integration/test_dino_cache_real_weights.py`.

**Pass criteria (all must hold):**
- All three stages exit 0; total wall time ≤ 25 min.
- `nvidia-smi` shows ≥ 6/8 GPUs busy during stage_4 (DROID chunks have ~50 episodes; full 8/8 expected).
- `${USAM_SCRATCH}/dino_cache/.../chunk-000/file-{000..007}.safetensors` exist; tensors non-zero.
- Per-shard episode shape `[T, 65, 1024]` (T frames at cache_fps=5, 64 keep + 1 CLS, embed_dim=1024 for ViT-L/16).
- Peak per-GPU memory < 12 GB.
- Sum of unique `episode_idx` across all 8 shard files == total episodes in chunk (no duplicates, no drops).

## 6. Slurm rollout & runbook

After A40 smoke passes:

1. **Build SIF on the A40 box.** `singularity build /scratch/$USER/usam_prep.sif docker-daemon://usam:prep-a100`. Weights travel inside the SIF.
2. **Ship.** `rsync -avh /scratch/$USER/usam_prep.sif <slurm-login>:$HOME/usam_prep.sif`. Set `USAM_SIF=$HOME/usam_prep.sif`.
3. **Tidy `slurm/env.sh`.** Remove the `HUGGINGFACE_TOKEN` requirement note for compute jobs (still needed only for the upload daemon on the login node).
4. **Single-chunk shakedown.** `sbatch slurm/job.sbatch stage_4_dino_cache droid 0`. Pass criteria identical to A40 step 4 but on A100 hardware.
5. **Production fanout.** Operator-driven via existing `prep/dispatch.py`. Out of this spec.

**Runbook deliverable** — `docs/HOWTO_PREP_DINOV3.md`, written *after* smoke passes from the verified commands. Sections: prereqs, build, image self-test, 8-GPU end-to-end smoke, ship to Slurm, sbatch one chunk, re-baking the image (DINOV3_MODEL build-arg), troubleshooting.

## 7. Acceptance for "spec done"

- Code/config changes from §3 land in a clean commit; existing tests pass.
- All three Dockerfiles restyled per §4; T0/T1/T2 build successfully with `HF_TOKEN` set.
- §5 smoke runs green on the 8xA40 box.
- New tests from §3.4 are present and passing (or skipping with the documented env-var gate).
- `docs/HOWTO_PREP_DINOV3.md` written from the verified commands.
- `slurm/env.sh` and `docker/README.md` updates land.

Live Slurm submission and production fanout are explicitly the operator's responsibility.
