# HOWTO — Local development on 8×A40

This guide covers end-to-end USAM development on a single 8×A40 box (the
T0 tier). Every command below is copy-pasteable; placeholders are
limited to obvious paths and are explained inline.

The plan reference is [`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
§8 (T0 testing harness).

---

## 1. Clone and install

```bash
git clone <USAM-remote-url> USAM && cd USAM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# flash-attn needs --no-build-isolation; only install on a CUDA box.
pip install --no-build-isolation flash-attn==2.6.3
```

`<USAM-remote-url>` is the git URL of the team's checkout — substitute
your fork or the upstream repo URL.

The `[dev]` extra pulls in pytest, ruff, and the test-only fixtures
synthesizer. Production-only packages (transformer-engine, libero,
robosuite) live in the `[train]` and `[eval]` extras and are not needed
for local development.

---

## 2. Build the local Docker image

```bash
docker build -f docker/Dockerfile.local_a40 -t usam:local-a40 .
```

This uses `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` as the base and
installs `requirements/base.txt` + `requirements/train.txt` plus
`flash-attn`. The image is suitable for unit tests, integration tests,
and the 100-step smoke train. See [`docker/README.md`](../docker/README.md)
for tier-by-tier image descriptions.

---

## 3. Generate test fixtures

USAM ships golden fixtures at `tests/golden_data/tiny_droid/`. The
fixtures are **auto-synthesized on demand** by the pytest fixture
`tiny_droid_root` (see `tests/conftest.py:39-52`) — there is no separate
`build_fixtures.py` script. The first test run materializes a 3-episode
synthetic stand-in to that path automatically.

If you need to materialize the fixture without running tests (e.g. to
inspect the on-disk layout):

```bash
python -c "
from pathlib import Path
from tests.golden_data._synthesize_tiny_droid import synthesize_tiny_droid
synthesize_tiny_droid(Path('tests/golden_data/tiny_droid'))
"
```

The synthesizer is deterministic (seeded RNG) so the fixture is
byte-identical across runs.

For the **real** LFS-tracked fixture (~200 MB across `tiny_droid`,
`tiny_agibot`, `tiny_robomind_franka`, `cached_dino`), pull via Git LFS:

```bash
git lfs install
git lfs pull
```

LFS-tracked fixtures take precedence: `_real_fixture_present()`
(`tests/conftest.py:31-36`) checks for `meta/info.json` and falls back
to synthesis only when missing.

---

## 4. Run unit tests

```bash
pytest tests/unit/ -x -v
```

Unit tests cover Tri-DINO shapes, LoRA gradient routing, plan-cache
bit-exactness, drift trigger logic, the two auxiliary heads, the unified
loss, dataloader behaviour, and the per-embodiment action
canonicalization. The full suite runs in well under 10 minutes on a
single CPU core; no GPU is required.

---

## 5. Run the CPU plumbing test

```bash
pytest tests/integration/test_smoke_train.py::test_smoke_train_cpu_plumbing -v
```

This test runs 5 training steps on the synthesized `tiny_droid` fixture
with the `usam_350m_smoke` config (further slimmed to a 2-layer / 64-dim
Player so the test fits in 60 s on a single core). It verifies the
whole loop wires up: dataloader → conductor → plan-cache refresh →
cache-dropout → player → unified loss → backward → optimizer.step.
Every loss must be finite; there is no statistical expectation of
monotonic decrease at 5 steps.

---

## 6. Run the full 8×A40 smoke train

```bash
bash scripts/train_smoke_a40.sh
```

The script:

* Auto-detects the GPU count from `nvidia-smi` (or `CUDA_VISIBLE_DEVICES`).
* Defaults to `configs/train/stage_b1_pretrain.yaml` +
  `configs/model/usam_350m_smoke.yaml` + `tests/golden_data/tiny_droid`.
* Runs `torchrun --standalone --nproc_per_node=$NPROC -m usam.train ...`
  for multi-GPU; falls back to a direct `python -m usam.train ...` for
  single-GPU or `--device cpu`.

Useful flags:

```bash
bash scripts/train_smoke_a40.sh --device cpu --max_steps 5    # plumbing only
bash scripts/train_smoke_a40.sh --auto_oom_reduce             # halve bs on first OOM
bash scripts/train_smoke_a40.sh --max_steps 200               # longer smoke
```

Hard guarantees enforced by `tests/integration/test_smoke_train.py::test_smoke_train`
(the same code path):

* loss is finite at every step;
* the 10-step moving average is monotonically non-increasing across
  the last 50 steps (microscopic drift tolerated);
* total wall-clock < 10 min.

The smoke run writes checkpoints + logs under
`runs/<YYYYmmdd-HHMMSS>-<short-uuid>/`.

---

## 6.5 Real-data pipeline + 8-GPU smoke train (Waves F–J)

§6 above exercises the synthetic-fixture smoke train. For a **real-data
end-to-end validation** — DROID download → SEA-RAFT flow → DA3
depth → DINOv3 cache → 2000-step train on 8 GPUs with wandb — use this
section.

### 6.5.1 Build the prep image once

```bash
# Set HF_TOKEN (with access to facebook/dinov3-vitl16-pretrain-lvd1689m
# and depth-anything/DA3MONO-LARGE).
printf '%s' "$HF_TOKEN" > ~/.hf_token && chmod 600 ~/.hf_token

DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,src=$HOME/.hf_token \
    -f docker/Dockerfile.prep_a100 \
    -t usam:prep-a100 .
```

The bake takes ~10–15 min on first build. It pre-downloads the
gated DINOv3 weights, the SEA-RAFT-M (Tartan-Spring) checkpoint, and
the DA3MONO-LARGE checkpoint into the image so Slurm nodes never need
HF credentials. See [`docker/README.md`](../docker/README.md).

### 6.5.2 Run the head-only prep pipeline

Head-only (egocentric) per the LDA-1B convention. The runner builds
the LeRobot v2.1 layout from the prep outputs.

```bash
docker run --rm --shm-size=8g --gpus all \
  -v /path/to/USAM:/workspace/USAM \
  -v /path/to/scratch:/workspace/output \
  -v /tmp/run_pipeline_head_only.py:/tmp/runner.py:ro \
  -e USAM_EPISODES_PER_CHUNK=2 -e TF_CPP_MIN_LOG_LEVEL=3 \
  -w /workspace/USAM usam:prep-a100 \
  bash -c 'pip install -q gcsfs tensorflow-cpu; python /tmp/runner.py'
```

Runs stages **2a (DROID → LeRobot)**, **2b (SEA-RAFT-M flow)**,
**2c (DA3MONO depth)**, **4 (DINOv3 ViT-L/16 cache, 8-GPU)**. Two-episode
chunk completes in ~3 min and writes ~340 MB.

`USAM_EPISODES_PER_CHUNK` overrides the default chunk size (256 episodes
for production). For a longer smoke, raise to 10–20.

### 6.5.3 Inspect feature quality (Wave I)

```bash
docker run --rm --gpus '"device=0"' \
  -v /path/to/USAM:/workspace/USAM \
  -v /path/to/scratch:/workspace/output \
  -w /workspace/USAM usam:prep-a100 \
  python tools/viz/dinov3_pca_gallery.py
```

Open `/path/to/scratch/viz/dinov3_chunk0/index.html` in a browser. For
each of N frames per (episode, camera) you get a 4-column side-by-side:

* **RGB** (the original 448×448 frame)
* **DINOv3 PCA** (1024-D patch tokens projected to 3 channels, 28×28 grid)
* **Depth** (DA3MONO-LARGE metric mm, viridis colormap)
* **Flow** (SEA-RAFT, HSV-encoded)

If the DINOv3 PCA panel shows coherent color regions around objects /
gripper / table edges, the encoder is producing semantically meaningful
features.

### 6.5.4 8-GPU smoke train with wandb (Wave J)

```bash
# One-time wandb credential file
printf '%s' "$WANDB_API_KEY" > ~/.wandb_key && chmod 600 ~/.wandb_key

docker run --rm --shm-size=16g --gpus all \
  -v /path/to/USAM:/workspace/USAM \
  -v /path/to/scratch:/workspace/output \
  -v $HOME/.wandb_key:/tmp/wandb_key:ro \
  -e WANDB_PROJECT=usam \
  -e USAM_VIZ_INTERVAL=50      `# logs DINOv3 PCA media every 50 steps` \
  -e USAM_MAX_STEPS=2000 \
  -e USAM_RAMP_STEPS=500       `# aux-head ramp shortened for the smoke` \
  -e USAM_NO_DDP=1             `# each rank trains its own copy; rank 0 logs` \
  -w /workspace/USAM usam:prep-a100 \
  bash -c '
    pip install -q wandb
    export WANDB_API_KEY="$(cat /tmp/wandb_key)"
    torchrun --standalone --nproc-per-node=8 tools/train/smoke_train_long.py
  '
```

Each rank runs on its own GPU (`USAM_NO_DDP=1` — no grad sync), seeds
are identical so all ranks compute the same updates. Total wall ~4 min.
Only rank 0 prints + writes to wandb.

For an **H200 production run**, raise `USAM_VIZ_INTERVAL` to 1000–5000
so media uploads don't compete with the long-run throughput.

### 6.5.5 What "good pretraining" looks like in wandb

Open the dashboard URL printed at the end of the run (e.g.
`https://wandb.ai/<entity>/usam/runs/<id>`). Healthy signs:

| Metric | Good | Bad |
|---|---|---|
| `loss/total` | decreasing curve through step 100 then exponential decay | flat or NaN |
| `loss/{action,rgb,depth,flow}` | all 4 decrease together | only `action` decreases (other heads not learning) |
| `grad/player` | non-zero throughout (~0.5–2 typical) | flatlines at 0 (Player not training) |
| `grad/encoder` | non-zero if LoRA wrappers caught the q_proj/k_proj/v_proj modules | flatlines at 0 (audit needed — see Wave A LoRA target-name policy) |
| `plan_cache/pairwise_cos` | near 0 (diverse plans) | rising toward 1 (Conductor collapse) |
| `plan_cache/emb_norm_mean` | stable | crashing toward 0 (cache death) |
| `data/{action_std,proprio_std,rgb_dino_std}` | non-zero, stable | falling to 0 (degenerate batches) |
| `weight/{geom,flow_act}` | ramping 0 → target over USAM_RAMP_STEPS | flat at 0 (ramp disabled) |
| `viz/dinov3_pca` (media) | semantic regions in patch PCA become more structured over time | uniform noise throughout |
| `step_time_ms` | stable after warmup (no slow drift up) | ramps up (memory leak risk) |

### 6.5.6 Sample wandb run (current main HEAD)

* 2000 steps on real DROID chunk 0 (2 episodes, head-only)
* Loss 96.5 → 1.5 (~60× reduction)
* All 12+ scalar diagnostics + 41 `viz/dinov3_pca` media items
* Run: <https://wandb.ai/crlc112358/usam/runs/6f8hc151>

---

## 7. Eval on a smoke checkpoint

The full LIBERO closed-loop evaluation runs on T2/T3; on T0 you can
exercise the open-loop ADE eval and the realtime smoke. Both rely on
the inference modules in `usam/inference/`.

There is no `scripts/eval_libero.sh` shipped today — open-loop eval is
invoked directly:

```bash
# Open-loop ADE on a fixture (Wave 4: body is currently a stub raising
# NotImplementedError; only the API contract is finalized)
python -c "
from usam.inference.openloop import run_openloop_eval, OpenLoopMetrics
# ... see usam/inference/openloop.py for the contract
"

# Realtime loop smoke (uses RealtimeController — see
# tests/integration/test_smoke_realtime.py for the canonical invocation)
pytest tests/integration/test_smoke_realtime.py -v
```

When the inference-engineer completes Wave 4 and a `scripts/eval_libero.sh`
lands, this section will be updated to:

```bash
bash scripts/eval_libero.sh <ckpt-path> <output-dir>
```

For now, see `usam/inference/realtime.py:RealtimeController` for the
single-step API the Wave-4 eval harness will wrap.
