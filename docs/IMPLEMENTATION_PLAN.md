# USAM Implementation Plan

**USAM = Unified Spatial Action Model**

A fork of LDA-1B (RSS 2026, arXiv:2602.12215) extending it to a tri-modal World-Action Model with explicit spatial-geometric supervision and cached hierarchical planning. This document is the canonical implementation reference; it lives in the repo as `docs/IMPLEMENTATION_PLAN.md`.

---

## Table of contents

1. [Project goals and non-goals](#1-project-goals-and-non-goals)
2. [Hardware tiers and what runs where](#2-hardware-tiers-and-what-runs-where)
3. [Repository topology](#3-repository-topology)
4. [Core architectural changes vs LDA-1B](#4-core-architectural-changes-vs-lda-1b)
5. [Data pipeline (Phase A)](#5-data-pipeline-phase-a)
6. [Pretraining (Phase B)](#6-pretraining-phase-b)
7. [Evaluation (Phase C)](#7-evaluation-phase-c)
8. [Local 8×A40 testing harness](#8-local-8a40-testing-harness)
9. [Docker images](#9-docker-images)
10. [Slurm integration](#10-slurm-integration)
11. [Module-by-module implementation specs](#11-module-by-module-implementation-specs)
12. [Testing strategy](#12-testing-strategy)
13. [Milestones and exit criteria](#13-milestones-and-exit-criteria)
14. [File-creation checklist](#14-file-creation-checklist)

---

## 1. Project goals and non-goals

### 1.1 Goals

| # | Goal | Concrete success metric |
|---|---|---|
| G1 | Ingest 5 Tier-1 robotic datasets into a unified format on HF Hub | DROID, AgiBot World 2026, RoboMIND 2.0, BridgeData V2, OXE-AugE all in `<org>/usam-<source>` repos with USAM-LeRobot v2.1 layout |
| G2 | Three modalities (RGB, depth, optical flow) projected to a shared DINOv3 latent space | Tri-DINO encoder produces aligned tokens; cross-modal InfoNCE > 0.6 on val |
| G3 | Joint denoising of action + tri-modal future latents with cross-modal consistency losses | Single MM-DiT, 4 task heads, 2 auxiliary heads (`L_geom`, `L_flow-act`) |
| G4 | Conductor + Player split with cosine-drift triggered Plan-KV-Cache | ≥3× inference speedup over per-step VLM forward at 30 Hz |
| G5 | All preprocessing fits 8×A100 Slurm with 4 h preemptible windows | Per-job preempt → resume loses ≤1 episode of progress |
| G6 | Pretraining fits 1 week on 500×H200 | Total compute ≤ 9 EFLOPs; cached DINO inputs cut training I/O ≥ 8× |
| G7 | All code is a thin overlay on LDA-1B | ≤ 2,000 LoC delta total; no MM-DiT core changes |

### 1.2 Non-goals (explicit)

- **Not** training a new VLM. Conductor stays frozen Qwen3-VL-4B from LDA-1B.
- **Not** training a new visual backbone. DINOv3 stays frozen except patch_embed adapters and LoRA.
- **Not** mixing human-egocentric data (Ego4D, EPIC, EgoVerse, etc.). Robot-only.
- **Not** rewriting the diffusion scheduler. We touch only inputs/outputs.
- **Not** building a serving stack. Real-time inference is for evaluation only.

### 1.3 Single sentence elevator pitch

> USAM extends LDA-1B's latent-dynamics paradigm by aligning RGB, depth, and optical flow in a shared DINOv3 space with cross-modal consistency losses, and decouples slow language understanding from fast control via a cosine-drift-triggered Plan-KV-Cache, enabling ≥3× faster real-time WAM inference at no quality cost.

---

## 2. Hardware tiers and what runs where

| Tier | Hardware | Wall-clock | Used for |
|---|---|---|---|
| **T0 — Local dev** | 8 × A40 (single node, ~48 GB each) | unlimited | Code dev, smoke tests, unit + integration tests, ≤ 0.1 % data slices |
| **T1 — Slurm prep** | 8 × A100 / job, 4 h walltime, preemptible, requeue | 6 weeks elapsed | Phase A pipeline: download, conversion, flow/depth compute, DINO caching, validation, upload |
| **T2 — H200 burst** | 500 × H200, 1 week | 7 days hard | Phase B pretraining + fine-tune |
| **T3 — Real-robot eval** | A40 desk + Franka/G1 | 2 weeks | Phase C evaluation |

The pipeline is designed so that **everything T2 needs is already produced and uploaded by T1**. T2 only reads cached fp16 DINO features + parquet metadata + small mp4 thumbnails. **No raw-video decoding on T2.**

---

## 3. Repository topology

```
USAM/                                    ← fork of LDA-1B
├── README.md                             ← USAM-specific README
├── LICENSE                               ← inherit MIT from LDA-1B
├── pyproject.toml                        ← +new optional extras: [prep], [train], [eval]
├── requirements/
│   ├── base.txt                          ← runtime deps (LDA-1B's, mostly)
│   ├── prep.txt                          ← +sea-raft, depth-anything-v2, decord, hf_xet, hf_transfer
│   ├── train.txt                         ← +flash-attn, transformer-engine
│   └── eval.txt                          ← +libero, robosuite, robocasa
├── docs/
│   ├── IMPLEMENTATION_PLAN.md            ← THIS FILE
│   ├── DATA_FORMAT.md                    ← USAM-LeRobot v2.1 spec
│   ├── ARCHITECTURE.md                   ← deeper dive on Tri-DINO + Conductor
│   ├── HOWTO_LOCAL_8A40.md
│   ├── HOWTO_SLURM_A100.md
│   └── HOWTO_H200.md
├── usam/                                 ← new code lives here, NOT inside lda/
│   ├── __init__.py
│   ├── encoders/
│   │   └── tri_dino.py                   ← Tri-modal DINO Tower with adapters
│   ├── conductor/
│   │   ├── __init__.py
│   │   ├── plan_cache.py                 ← Plan-KV-Cache
│   │   ├── drift.py                      ← cosine-drift trigger + f_drift MLP
│   │   ├── classifier.py                 ← subtask-completion head
│   │   └── conductor.py                  ← wraps Qwen3-VL-4B
│   ├── aux_heads/
│   │   ├── __init__.py
│   │   ├── depth_consistency.py          ← L_geom (Spearman rank, soft)
│   │   └── flow_action.py                ← L_flow-act (forward-action MLP)
│   ├── adapters/
│   │   └── lora.py                       ← LoRA helper for DINOv3 attention
│   ├── losses.py                         ← unified loss with all weights
│   ├── train.py                          ← USAM training entry (wraps lda.train)
│   ├── inference/
│   │   ├── realtime.py                   ← real-time loop with cache
│   │   └── openloop.py                   ← open-loop eval
│   └── dataloader/
│       ├── usam_lerobot.py               ← USAM-LeRobot v2.1 reader
│       ├── feature_cache.py              ← fp16 DINO cache reader
│       └── mixtures.py                   ← per-source sampling weights
├── prep/                                 ← Phase A pipeline (T1)
│   ├── __init__.py
│   ├── _base.py                          ← CheckpointedJob, signal handling
│   ├── _hub.py                           ← upload_large_folder + CommitScheduler
│   ├── _validation.py                    ← per-shard validation gates
│   ├── stage_0_download/
│   │   ├── droid.py
│   │   ├── agibot2026.py
│   │   ├── robomind.py
│   │   ├── bridge.py
│   │   └── oxe_auge.py
│   ├── stage_1_index.py
│   ├── stage_2a_to_lerobot/
│   │   ├── droid.py
│   │   ├── agibot2026.py
│   │   ├── robomind.py
│   │   ├── bridge.py
│   │   └── oxe_auge.py
│   ├── stage_2b_compute_flow.py          ← SEA-RAFT
│   ├── stage_2c_compute_depth.py         ← ZED stereo / DAv2
│   ├── stage_3_canonical.py              ← action canonicalization
│   ├── stage_4_dino_cache.py             ← Tri-DINO fp16 caching
│   ├── stage_5_validate.py
│   ├── stage_6_upload.py
│   ├── dispatch.py                       ← Slurm DAG dispatcher
│   └── adapter_pretrain.py               ← Phase A.5 depth/flow adapter pretrain
├── slurm/
│   ├── job.sbatch                        ← universal preemptible template
│   ├── env.sh                            ← module loads, conda activation
│   └── README.md
├── docker/
│   ├── Dockerfile.local_a40              ← T0 image
│   ├── Dockerfile.prep_a100              ← T1 image (Slurm)
│   ├── Dockerfile.train_h200             ← T2 image
│   └── README.md
├── configs/                              ← YAML configs
│   ├── data/
│   │   ├── droid.yaml
│   │   ├── agibot2026.yaml
│   │   ├── robomind.yaml
│   │   ├── bridge.yaml
│   │   ├── oxe_auge.yaml
│   │   └── camera_maps/                  ← per-config camera serial → canonical key
│   ├── model/
│   │   ├── usam_1_4b.yaml                ← main model
│   │   └── usam_350m_smoke.yaml          ← T0 smoke-test model
│   ├── train/
│   │   ├── stage_b1_pretrain.yaml
│   │   ├── stage_b2_finetune.yaml
│   │   └── adapter_pretrain.yaml
│   └── eval/
│       ├── libero.yaml
│       └── realtime.yaml
├── tests/
│   ├── unit/
│   │   ├── test_tri_dino.py
│   │   ├── test_plan_cache.py
│   │   ├── test_drift.py
│   │   ├── test_aux_heads.py
│   │   ├── test_dataloader.py
│   │   └── test_canonical_action.py
│   ├── integration/
│   │   ├── test_smoke_train.py           ← train 100 steps on 1 % data
│   │   ├── test_smoke_realtime.py
│   │   └── test_pipeline_end_to_end.py
│   └── golden_data/                      ← tiny LFS fixtures
├── scripts/                              ← convenience CLI
│   ├── prep_run_local.sh
│   ├── prep_submit_slurm.sh
│   ├── train_smoke_a40.sh
│   ├── train_h200.sh
│   └── eval_libero.sh
└── lda/                                  ← LDA-1B's original code, lightly modified
    ├── transformer.py                    ← +5 LoC: AdaLN gets proprio_proj
    ├── heads.py                          ← +10 LoC: depth_dino_head, flow_dino_head
    └── (everything else unchanged)
```

The principle: **everything new goes under `usam/` or `prep/`. The `lda/` directory is touched as little as possible** so that pulling upstream LDA-1B fixes is trivial.

---

## 4. Core architectural changes vs LDA-1B

### 4.1 Module-level diff

| File | Status | Purpose |
|---|---|---|
| `lda/transformer.py` | **modified** (+5 LoC) | AdaLN-Zero takes a `proprio_emb` modulation source |
| `lda/heads.py` | **modified** (+10 LoC) | Add `depth_dino_head`, `flow_dino_head` |
| `usam/encoders/tri_dino.py` | **new** | Tri-modal DINO Tower with depth/flow adapters + LoRA |
| `usam/adapters/lora.py` | **new** | LoRA layer factory for DINOv3 attention blocks |
| `usam/conductor/plan_cache.py` | **new** | Plan-KV-Cache holding pre-projected K, V across layers |
| `usam/conductor/drift.py` | **new** | f_drift MLP + cosine-drift trigger logic |
| `usam/conductor/classifier.py` | **new** | Subtask-completion head |
| `usam/conductor/conductor.py` | **new** | Wraps Qwen3-VL-4B with extraction of `e` and `p̂` |
| `usam/aux_heads/depth_consistency.py` | **new** | Soft Spearman rank loss (`L_geom`) |
| `usam/aux_heads/flow_action.py` | **new** | Forward-action MLP for `L_flow-act` |
| `usam/losses.py` | **new** | Unified loss with task balancing + GradNorm option |
| `usam/train.py` | **new** | Top-level training loop; reuses LDA-1B's optimizer + scheduler |
| `usam/dataloader/usam_lerobot.py` | **new** | Reads USAM-LeRobot v2.1 (parquet + cached features + mp4 thumbnails) |
| `usam/dataloader/feature_cache.py` | **new** | fp16 DINO feature shard reader (memory-mapped safetensors) |
| `usam/dataloader/mixtures.py` | **new** | Per-source sampling weights |
| `usam/inference/realtime.py` | **new** | Real-time control loop with cache |
| `usam/inference/openloop.py` | **new** | Open-loop ADE evaluation |

### 4.2 Forward pass at training time

Inputs (per training sample):
- `rgb_dino_seq[t-T:t+1]`, `depth_dino_seq[t-T:t+1]` — pre-cached fp16 features at 5 Hz, T = 4
- `proprio[t]` — embodiment-normalized 50-D vector
- `action_chunk[t:t+16]` — 7-D canonical-EE action chunk (or padded native)
- `instruction` — text string (used by Conductor on first call)
- `head_keyframe_rgb[t]` — single fp16 RGB-DINO frame (used by Conductor for visual context)
- `task_id` — one of {Policy, FDM, IDM, VisFcst}
- `noise_level` — diffusion timestep
- `subtask_label` — for the classifier head (from AgiBot's `instruction_segments`)

Forward pass:
1. **Conductor pass**: `Qwen3-VL-4B(instruction, head_keyframe_rgb)` → `e ∈ ℝ^{D_emb}`, `P̂ ∈ ℝ^{32 × D}`. *In training, this is run once per (sample, drop-out coin)*.
2. **Plan-Cache projection**: `K_layer = W_k_layer @ P̂`, `V_layer = W_v_layer @ P̂` for each Player layer.
3. **Player MM-DiT**: standard LDA-1B forward, with cross-attention reading from the cache; AdaLN-Zero now also conditioned on `proprio_emb`.
4. **Output heads**:
   - `action_head` → predicted action chunk velocity
   - `rgb_dino_head` → predicted future RGB-DINO velocity
   - `depth_dino_head` → predicted future Depth-DINO velocity
5. **Auxiliary heads**:
   - `f_drift_mlp(rgb_dino_cls[t], e_committed)` → predicted next `e` (for drift trigger)
   - `subtask_classifier(P̂, rgb_dino_cls[t])` → P(subtask completed)
6. **Losses**: see §4.3.

### 4.3 Loss equation (concrete)

```python
L_action  = flow_match(action_pred, action_gt)
L_rgb     = flow_match(rgb_dino_pred, rgb_dino_gt)
L_depth   = flow_match(depth_dino_pred, depth_dino_gt)

# auxiliary
L_geom     = soft_spearman_rank(decode_depth(depth_dino_pred), nearfield_cos(rgb_dino_pred))
L_drift    = mse(f_drift(rgb_dino_cls, e_committed), e_target)
L_subtask  = bce(subtask_classifier(P_hat, rgb_dino_cls), subtask_label)

L_total = (1.0 * L_action +
           1.0 * L_rgb +
           0.3 * L_depth +
           ramp(0.05, step) * L_geom +
           0.1 * L_drift +
           0.1 * L_subtask)
```

`ramp(0.05, step)` = linear ramp from 0 to 0.05 between steps 50K–100K. Aux geom starts disabled to avoid early-training instability.

### 4.4 Forward pass at inference time

```python
# at episode start
e_committed, P_hat_committed = conductor(instruction, keyframe_rgb)
plan_cache.refresh(P_hat_committed, e_committed, k_projs, v_projs, t=0)

for t in range(episode_len):
    rgb_dino   = dinov3.rgb(obs.rgb)
    depth_dino = dinov3.depth(obs.depth)
    flow_dino  = dinov3.flow(obs.flow)

    # cheap drift check
    e_now_estimate = f_drift(rgb_dino_cls=rgb_dino[..., 0, :], e_committed=e_committed)
    d_t = 1 - cos(e_committed, e_now_estimate)

    if should_refresh(t, d_t, last_refresh_t, plan_cache.committed_emb):
        e_committed, P_hat_committed = conductor(instruction, obs.rgb)  # fresh forward
        plan_cache.refresh(P_hat_committed, e_committed, k_projs, v_projs, t)

    action = player.denoise(rgb_dino, depth_dino, flow_dino, proprio,
                             plan_cache=plan_cache, n_steps=10)
    yield action
```

The Conductor does **not** run every step; the `f_drift` MLP gates it.

---

## 5. Data pipeline (Phase A)

### 5.1 Repository layout on HF Hub

One repo per source. Each has the same internal layout:

```
<org>/usam-<source>/
├── meta/
│   ├── info.json                ← codebase_version=v2.1, fps, features
│   ├── modality.json            ← USAM-specific (state/action dims per channel)
│   ├── tasks.parquet            ← episode → task descriptions
│   ├── episodes.parquet         ← episode_index, length, embodiment, etc.
│   ├── stats.safetensors        ← normalization stats
│   ├── embodiment.json          ← per-embodiment action canonicalization rule
│   └── conversion_log.jsonl     ← per-episode success/failure
├── data/
│   └── chunk-{000..NNN}/
│       └── file-{000..999}.parquet
├── videos/
│   ├── observation.images.head_rgb/chunk-XXX/file-YYY.mp4
│   ├── observation.images.head_depth/...                ← 16-bit HEVC
│   ├── observation.images.head_flow/...                 ← HSV-encoded h264
│   ├── observation.images.wrist_rgb/...
│   ├── observation.images.wrist_depth/...
│   └── observation.images.wrist_flow/...
└── features/                                            ← USAM cache layer
    ├── rgb/chunk-XXX/file-YYY.safetensors               ← fp16 [N, 65, 768]
    ├── depth/chunk-XXX/file-YYY.safetensors
    └── flow/chunk-XXX/file-YYY.safetensors
```

### 5.2 The 6 source-specific converters

Each converter lives in `prep/stage_2a_to_lerobot/<source>.py` and implements:

```python
class SourceConverter(CheckpointedJob):
    def list_episodes(self) -> list[EpisodeRef]: ...
    def convert_episode(self, ep: EpisodeRef) -> ConversionResult: ...
    def write_shard(self, results: list[ConversionResult]) -> Path: ...
```

The `convert_episode` method always returns the same internal record:

```python
@dataclass
class ConversionResult:
    episode_index: int
    embodiment: str
    fps: int
    cameras: dict[str, np.ndarray]      # canonical_key -> [T, H, W, 3]
    depth: dict[str, np.ndarray]        # optional, [T, H, W] uint16
    state: np.ndarray                   # [T, D_state] padded to 50
    state_mask: np.ndarray              # [50] bool
    action_native: np.ndarray           # [T, D_action] padded to 32
    action_mask: np.ndarray             # [32] bool
    action_canonical_ee: np.ndarray     # [T, 7]
    instructions: dict[str, list]       # level_1, level_2, level_3
    force_torque: np.ndarray | None     # [T, 6]
    timestamps: np.ndarray              # [T] float32
    raw_meta: dict
```

Source-specific quirks documented in §11.4.

### 5.3 Stage DAG

```
0_download   ──→  1_index   ──→  2a_to_lerobot ──┐
                                  2b_flow ───────┼──→  3_canonical ──→ 4_dino_cache ──→ 5_validate ──→ 6_upload
                                  2c_depth ──────┘
```

Each stage:
- Reads from a `manifests/<source>__<stage>.parquet` that lists chunk_id → status
- Processes a single chunk per Slurm job
- Writes its outputs to `/scratch/usam/<source>/<stage>/chunk-XXX/`
- Marks the manifest entry as `done` only after validation passes

### 5.4 Storage budget

| Component | Per-hour | × 10,500 h | × 3 cams |
|---|---|---|---|
| Parquet | ~10 MB | 105 GB | (camera-agnostic) |
| RGB MP4 (h264, 384², 30 fps) | 1.4 GB | 14.7 TB | × 3 = **44 TB** |
| Depth MP4 (HEVC 16-bit, 192², 15 fps) | 0.4 GB | 4.2 TB | × 3 = 12.6 TB |
| Flow MP4 (HSV h264, 384², 30 fps) | 1.0 GB | 10.5 TB | × 3 = 31.5 TB |
| Cached DINO features (5 fps, 65 tokens) | 0.5 GB | 5.3 TB | × 3 = **16 TB** |
| **Total on Hub** | | | **~104 TB** |

→ Apply for HF storage grant or use Enterprise tier. Local scratch will hold raw downloads (~70 TB) which never go to the Hub.

### 5.5 Upload strategy

**Hard rules** (these prevent the failure mode you hit before):
1. Never call `Dataset.push_to_hub`. Always `huggingface_hub.upload_large_folder` or `CommitScheduler`.
2. Each shard ≤ 5 GB. Each chunk dir ≤ 1,000 files.
3. Idempotency: filenames include a content hash of the source episode + version. Re-running never duplicates.
4. The upload daemon (`prep/_hub.py`'s `CommitScheduler`) runs on a long-lived login node, not inside Slurm jobs.
5. Slurm jobs only write to local scratch. They do not talk to the Hub.

---

## 6. Pretraining (Phase B)

### 6.1 Stages

| Stage | Time | Data | Active losses |
|---|---|---|---|
| **B0. Adapter init** | offline | result of `prep/adapter_pretrain.py` | depth/flow patch_embed + LoRA pretrained |
| **B1. Robot pretrain** | 5 days | All Tier-1 in USAM-LeRobot | full L_total from §4.3 |
| **B2. Embodiment fine-tune** | 1.5 days | Target embodiment subset | action λ=1, vis λ=0.1 each |
| **B3. Eval + ablation runs** | 0.5 days | LIBERO + RoboCasa-GR1 sim | none |

### 6.2 Optimizer config

```yaml
# configs/train/stage_b1_pretrain.yaml
optimizer:
  name: adamw
  lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.05
schedule:
  warmup_steps: 2000
  total_steps: 700000
  type: cosine
  min_lr: 1.0e-5
batch:
  global_size: 1024
  micro_size: 4
  grad_accum: 1     # 256 GPUs × 4 = 1024
precision:
  weights: bf16
  activations_te_fp8: true
  optimizer_states: bf16
parallelism:
  data_parallel: 250    # 500 / 2
  tensor_parallel: 2
  pipeline_parallel: 1
checkpoint:
  every_steps: 5000
  keep_last: 3
loss_weights:
  action: 1.0
  rgb: 1.0
  depth: 0.3
  flow: 0.3
  geom_max: 0.05      # ramp from 0 between 50K–100K
  flow_act_max: 0.05  # same
  drift: 0.1
  subtask: 0.1
cache_dropout_prob: 0.5     # train-time stale-plan robustness
plan_stale_window_frames: 60
```

### 6.3 Compute math

- 1.4B Player × bf16 fwd/bwd: ~ 8 PFLOP / step at batch 1024 × seq 65×4 (frames × tokens) → roughly 9 EFLOPs over 700K steps
- 500 H200 × 75% MFU × 1100 TFLOPS sustained ≈ 8.6 EFLOPs / 7 d, so we have ~30% margin
- Cached-DINO inputs cut per-step FLOPs ~10× vs the alternative of running DINOv3 inside the train loop

---

## 7. Evaluation (Phase C)

### 7.1 Eval surfaces

| Eval | Where | What it checks |
|---|---|---|
| **Smoke (T0)** | 8×A40, 100 steps | gradient flows, no NaN, loss decreases |
| **Open-loop ADE** | T2 last hour | action-prediction error on holdout |
| **LIBERO closed-loop** | T2 last hour | per-task success rate |
| **RoboCasa-GR1** | T3 | per-task success rate |
| **Real robot** | T3 | tabletop pick-and-place, dexterous (Galbot G1, Unitree G1) |
| **Inference timing** | T0 + T3 | wall-clock @ 30 Hz with vs without cache |

### 7.2 Ablations to budget

We MUST run these as part of the H200 burst (so we can paper-claim):

| Ablation | Variant | Cost on 500 H200 |
|---|---|---|
| **A1**: RGB-only vs Tri-modal | Drop depth+flow heads | ~2 days, run as 250-GPU half-cluster job |
| **A2**: Cache vs no-cache | Disable f_drift, full Conductor every step | inference-time eval only, no training |
| **A3**: Geom losses | Set μ_geom = μ_flow_act = 0 | ~2 days, half-cluster |
| **A4**: Robot-only vs +Ego4D | Ego4D mixed in 20% | optional, Phase Y |

A1 and A3 share the half-cluster slot serially; A2 is free.

---

## 8. Local 8×A40 testing harness

### 8.1 What we test on T0

- Unit tests for every new module (Tri-DINO, plan_cache, drift, aux heads, dataloader)
- Integration test: 100-step smoke train on a 1% data slice (≈ 10 hours of robot data)
- Real-time inference smoke at 10 Hz (full 30 Hz needs H200)

### 8.2 Scaled-down smoke model

`configs/model/usam_350m_smoke.yaml`:
- 12 layers, d=1024, h=16, ffn=4096 → ~350M params
- Action chunk size = 8 (vs 16)
- Tri-DINO uses ViT-S/14 (384-dim) for all three modalities (vs ViT-B for RGB)
- Sequence length 4 frames × 65 tokens

Fits comfortably on 8×A40 with bs=4 per GPU, gradient_accumulation=1.

### 8.3 Test data fixtures

`tests/golden_data/` contains LFS-tracked fixtures:
- `tiny_droid/` — 5 episodes, ≈ 30 s each
- `tiny_agibot/` — 5 episodes from AgiBot 2026 sample
- `tiny_robomind_franka/` — 3 episodes
- `cached_dino/` — pre-encoded DINO features for the above

These are committed to the repo (LFS-tracked, ≈ 200 MB total) so any developer can run smoke tests without network access.

---

## 9. Docker images

Three images, all based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`.

### 9.1 `Dockerfile.local_a40` (T0)

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git git-lfs ffmpeg \
    libgl1-mesa-glx libglib2.0-0 wget curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/USAM
COPY pyproject.toml requirements/base.txt requirements/train.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements/base.txt && \
    pip install -r requirements/train.txt
RUN pip install --no-build-isolation flash-attn==2.6.3
COPY . .
RUN pip install -e .
ENV PYTHONPATH=/workspace/USAM
CMD ["bash"]
```

### 9.2 `Dockerfile.prep_a100` (T1, Slurm)

Adds Phase A dependencies; *does not* install flash-attn (not needed for prep). Uses Singularity-compatible build (no `--privileged` required at runtime).

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git git-lfs ffmpeg \
    libgl1-mesa-glx libglib2.0-0 wget curl rsync zstd \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/USAM
COPY pyproject.toml requirements/base.txt requirements/prep.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements/base.txt && \
    pip install -r requirements/prep.txt
# huggingface upload deps
RUN pip install hf_xet hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
COPY . .
RUN pip install -e .
ENV PYTHONPATH=/workspace/USAM
CMD ["bash"]
```

### 9.3 `Dockerfile.train_h200` (T2)

Like local_a40 but with Transformer Engine for FP8.

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3
WORKDIR /workspace/USAM
COPY pyproject.toml requirements/base.txt requirements/train.txt ./
RUN pip install -r requirements/base.txt && \
    pip install -r requirements/train.txt
RUN pip install --no-build-isolation flash-attn==2.6.3
RUN pip install transformer-engine==1.11
COPY . .
RUN pip install -e .
ENV PYTHONPATH=/workspace/USAM
ENV NCCL_IB_HCA=mlx5
ENV NCCL_SOCKET_IFNAME=^lo,docker0
CMD ["bash"]
```

Build & push:
```
docker build -f docker/Dockerfile.local_a40 -t <reg>/usam:local-a40-v0.1 .
docker build -f docker/Dockerfile.prep_a100 -t <reg>/usam:prep-a100-v0.1 .
docker build -f docker/Dockerfile.train_h200 -t <reg>/usam:train-h200-v0.1 .
```

For Slurm A100 we convert to Singularity:
```
singularity build usam_prep.sif docker://<reg>/usam:prep-a100-v0.1
```

---

## 10. Slurm integration

### 10.1 `slurm/job.sbatch`

The universal preemptible template:

```bash
#!/bin/bash
#SBATCH --job-name=usam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8
#SBATCH --time=03:55:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh

STAGE=$1
SOURCE=$2
CHUNK=$3
EXTRA="${@:4}"

term_handler() {
  echo "[bash] caught USR1 → forwarding to python"
  kill -USR1 "$PYPID" 2>/dev/null || true
  wait "$PYPID"
  EXIT=$?
  if [[ $EXIT -eq 124 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit $EXIT
}
trap term_handler USR1

singularity exec --nv \
    --bind /scratch:/scratch \
    --bind /home:/home \
    "$USAM_SIF" \
    bash -lc "cd $USAM_REPO && python -m prep.${STAGE} \
        --source ${SOURCE} --chunk ${CHUNK} --resume ${EXTRA}" &
PYPID=$!
wait "$PYPID"
EXIT=$?
[[ $EXIT -eq 124 ]] && scontrol requeue "$SLURM_JOB_ID"
exit $EXIT
```

### 10.2 `slurm/env.sh`

Module loads, conda or singularity sif paths, HF token export, scratch dir.

### 10.3 Dispatcher

`prep/dispatch.py` walks the manifest DAG, finds chunks whose dependencies are `done` and whose own status is `pending`, and submits up to `MAX_PENDING` jobs at a time. Run as a long-lived background process on the login node.

---

## 11. Module-by-module implementation specs

This section gives every module a complete, self-contained spec. Anyone (or any agent) can implement a module by reading only its subsection.

### 11.1 `usam/encoders/tri_dino.py`

**Purpose**: One DINOv3 backbone, three input adapters, three modality-aware LoRA paths.

**Interface**:
```python
class TriDinoTower(nn.Module):
    def __init__(self,
                 dinov3_ckpt: str,
                 dinov3_arch: str = "vit_b_14",
                 lora_rank: int = 8,
                 freeze_rgb_patch_embed: bool = True): ...

    def forward(self, x: Tensor, modality: Literal["rgb","depth","flow"]) -> Tensor:
        """x: [B, C, H, W] (C=3 for rgb, 1 for depth, 2 for flow)
           returns: [B, N_tokens, D]"""

    @torch.no_grad()
    def extract_features(self, x, modality, n_keep_tokens: int = 64) -> Tensor:
        """fp16 cache extraction; returns [B, n_keep_tokens+1, D]"""
```

**Key implementation points**:
- Initialize `depth_patch.weight` from `mean(rgb_patch.weight, dim=1, keepdim=True)`.
- Initialize `flow_patch.weight` from `rgb_patch.weight[:, :2]`.
- LoRA wraps DINOv3's attention Q, K, V (not the MLP). Use the same LoRA modules for depth and flow paths but with separate `lora_modality_id` to keep them distinct.
- All non-trainable tensors live in fp16/bf16 to save memory at A40.

**Test** (`tests/unit/test_tri_dino.py`):
- Each modality forward produces `[B, 729, 768]` for ViT-B/14 at 384×384.
- `extract_features` returns `[B, 65, 768]` fp16 by default.
- Re-encoding the same input twice gives bit-exact (fp16) output.
- LoRA parameters have `requires_grad=True`; backbone does not (except patch_embed).

---

### 11.2 `usam/adapters/lora.py`

**Purpose**: LoRA factory that wraps `nn.Linear` modules in DINOv3's attention.

**Interface**:
```python
def apply_lora(model: nn.Module, r: int, target_module_names: list[str],
               modality_ids: list[str]) -> dict[str, nn.Module]:
    """
    Replace `target_module_names` (e.g. ['qkv', 'proj']) inside each transformer
    block with a LoRALinear that holds one LoRA path per modality_id.
    Returns dict of modality_id → list of LoRA modules for parameter grouping.
    """

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, modality_ids: list[str]): ...
    def forward(self, x, modality_id: str) -> Tensor: ...
```

**Test**: Frozen base + LoRA forward agrees with base forward when LoRA weights are zero. Nonzero LoRA changes output. Backward only updates LoRA params.

---

### 11.3 `usam/conductor/conductor.py`

**Purpose**: Wraps Qwen3-VL-4B and exposes `forward(instruction, keyframe_rgb) → (e, P̂)`.

**Interface**:
```python
class Conductor(nn.Module):
    def __init__(self, qwen_ckpt: str, n_plan_tokens: int = 32,
                 player_d_model: int = 2048):
        super().__init__()
        self.vlm = load_qwen3_vl(qwen_ckpt, frozen=True)
        self.plan_proj = nn.Linear(self.vlm.hidden_size, player_d_model)
        # `e` extraction: last hidden state of [EOS]; L2-normalize.

    @torch.no_grad()
    def forward(self, instruction: str | list[str], keyframe_rgb: Tensor)
        -> tuple[Tensor, Tensor]:
        """Returns:
          e:   [B, D_emb]      L2-normalized
          P̂:   [B, n_plan, D_player]
        """
```

**Test**: Output shapes match. `e` has unit norm. Identical inputs give bit-exact outputs.

---

### 11.4 `usam/conductor/plan_cache.py`

**Purpose**: Hold pre-projected K, V across all Player layers.

**Interface**:
```python
class PlanCache:
    def __init__(self, n_layers: int, d_model: int, n_p: int = 32, dtype=torch.bfloat16): ...

    def refresh(self, p_hat: Tensor, e: Tensor, k_projs: list[nn.Linear],
                v_projs: list[nn.Linear], t: int) -> None: ...

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        return self.k[layer_idx], self.v[layer_idx]

    def is_valid(self) -> bool: ...
    @property
    def committed_emb(self) -> Tensor: ...
```

**Test**: After refresh, `get(L)` returns expected shape `[n_p, d_model]`. Calling cross-attention with cached K/V vs recomputed K/V gives bit-exact output.

---

### 11.5 `usam/conductor/drift.py`

**Purpose**: f_drift MLP and the cosine-drift `should_refresh` decision.

**Interface**:
```python
class FDriftMLP(nn.Module):
    """Predicts e_now from (rgb_dino_cls, e_committed). 2-layer MLP, ~50K params."""
    def __init__(self, rgb_dino_dim: int = 768, e_dim: int = 4096, hidden: int = 256): ...
    def forward(self, rgb_dino_cls: Tensor, e_committed: Tensor) -> Tensor: ...

@dataclass
class DriftConfig:
    tau_hard: float = 0.20
    tau_soft: float = 0.06
    timer: int = 60               # 60 frames @ 30 Hz = 2 s
    timer_soft: int = 30          # 1 s

def should_refresh(t: int, d_t: float, last_refresh_t: int,
                   config: DriftConfig, subtask_completed: bool, episode_start: bool) -> bool: ...
```

**Calibration helper**:
```python
def calibrate_taus(dataset_loader, conductor, n_samples: int = 10_000) -> DriftConfig:
    """Compute empirical 50th and 90th percentiles of cosine distance
    between consecutive committed embeddings. Returns calibrated DriftConfig."""
```

**Test**: `should_refresh` returns True at t=0; True after timer expires; True when d_t > tau_hard; False otherwise.

---

### 11.6 `usam/conductor/classifier.py`

**Purpose**: Subtask-completion head trained on AgiBot World 2026 segment boundaries.

```python
class SubtaskCompletionHead(nn.Module):
    def __init__(self, p_hat_dim: int = 2048, rgb_dino_dim: int = 768, hidden: int = 512): ...
    def forward(self, p_hat: Tensor, rgb_dino_cls: Tensor) -> Tensor:
        """returns logits [B, 1]"""
```

Trained jointly with `L_subtask = bce(...)` weighted at 0.1.

---

### 11.7 `usam/aux_heads/depth_consistency.py`

**Purpose**: `L_geom` — soft-Spearman rank consistency between predicted depth and predicted RGB-DINO near-field similarity.

```python
class GeomConsistencyLoss(nn.Module):
    def __init__(self, dav2_distill_ckpt: str, nearfield_proto_ckpt: str): ...

    def forward(self, depth_dino_pred: Tensor, rgb_dino_pred: Tensor) -> Tensor:
        """depth_dino_pred, rgb_dino_pred: [B, T, N_patches, D]
           Returns scalar loss (lower is better; bounded [-1, 1])."""
```

Uses a frozen `decode_depth(depth_dino)` MLP head distilled from DAv2 in Phase A.5 and a frozen `nearfield_cos(rgb_dino)` prototype.

Soft Spearman from the `differentiable-rank` package or hand-rolled with Gumbel-soft-sort.

---

### 11.8 `usam/aux_heads/flow_action.py`

**Purpose**: `L_flow-act` — predicted flow magnitude must match what the action chunk should produce.

```python
class FlowActionConsistencyLoss(nn.Module):
    def __init__(self, proprio_dim: int = 50, action_chunk_dim: int = 7*16, hidden: int = 256): ...

    def forward(self, proprio: Tensor, action_chunk: Tensor, flow_dino_pred: Tensor) -> Tensor:
        """Returns scalar MSE loss."""
```

`g_phi` is a 2-layer MLP. `flow_magnitude(flow_dino_pred)` is a fixed transform: decode patch tokens → mean magnitude in HSV-V channel.

---

### 11.9 `usam/losses.py`

**Purpose**: Combine all losses with configurable weights and optional GradNorm.

```python
@dataclass
class LossWeights:
    action: float = 1.0
    rgb: float = 1.0
    depth: float = 0.3
    flow: float = 0.3
    geom: float = 0.0           # ramped externally
    flow_act: float = 0.0       # ramped externally
    drift: float = 0.1
    subtask: float = 0.1

class USAMUnifiedLoss(nn.Module):
    def __init__(self, weights: LossWeights, use_gradnorm: bool = False): ...
    def forward(self, predictions: dict, targets: dict, masks: dict)
        -> tuple[Tensor, dict[str, Tensor]]:
        """Returns (total_loss, per_loss_log_dict)."""
```

---

### 11.10 `usam/dataloader/usam_lerobot.py`

**Purpose**: Read USAM-LeRobot v2.1 datasets; at training time prefer cached fp16 features over decoding video.

```python
class USAMLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, repo_id: str, split: str = "train",
                 use_cached_features: bool = True,
                 modalities: list[str] = ["rgb", "depth", "flow"],
                 cameras: list[str] = ["head", "wrist"],
                 history_frames: int = 4, future_frames: int = 8,
                 action_chunk: int = 16,
                 fps_features: int = 5, fps_action: int = 30):
        ...
    def __getitem__(self, idx) -> dict[str, Tensor]:
        return {
            "rgb_dino_seq":    [-T..T, N, D],
            "depth_dino_seq":  ...,
            "flow_dino_seq":   ...,
            "proprio":         [D_state],
            "action_chunk":    [16, 7],
            "action_native":   [16, 32],
            "instruction":     str,
            "task_id":         int,
            "noise_level":     float,
            "subtask_label":   bool,
            "head_keyframe_rgb_dino": [N, D],
        }
```

**Streaming mode**: When `use_cached_features=False` (smoke test only), decode mp4 + run TriDinoTower inline.

---

### 11.11 `usam/dataloader/feature_cache.py`

Memory-mapped safetensors reader for fp16 DINO shards.

```python
class FeatureCache:
    def __init__(self, root: str, modality: str, dtype=torch.float16): ...
    def get(self, episode_index: int, frame_indices: Tensor) -> Tensor: ...
```

Uses `safetensors.safe_open` with `framework="pt"` and `device="meta"` for true mmap behavior. Multiple workers share the same mmap region (kernel page cache).

---

### 11.12 `usam/dataloader/mixtures.py`

```python
@dataclass
class SourceMixture:
    name: str
    repo_id: str
    weight: float

DEFAULT_TIER1_MIX = [
    SourceMixture("droid",       "<org>/usam-droid",       0.20),
    SourceMixture("agibot2026",  "<org>/usam-agibot2026",  0.35),
    SourceMixture("robomind",    "<org>/usam-robomind",    0.20),
    SourceMixture("bridge",      "<org>/usam-bridge",      0.15),
    SourceMixture("oxe_auge",    "<org>/usam-oxe-auge",    0.10),
]
```

Weights chosen so each source contributes a sensible chunk despite huge size differences.

---

### 11.13 `prep/_base.py`

```python
class CheckpointedJob:
    """Per-episode idempotent processing with SIGUSR1 graceful exit."""
    def __init__(self, source: str, stage: str, chunk: int): ...
    def list_episodes(self) -> Iterator[EpisodeRef]: ...
    def is_done(self, ep: EpisodeRef) -> bool: ...
    def process(self, ep: EpisodeRef) -> None: ...   # subclass implements
    def run(self):
        for ep in self.list_episodes():
            if self.is_done(ep): continue
            if self._stop_requested:
                self._flush(); sys.exit(124)
            self.process(ep)
            self.mark_done(ep)
        sys.exit(0)
```

---

### 11.14 `prep/_hub.py`

```python
def make_commit_scheduler(repo_id: str, folder: Path, every_min: int = 10) -> CommitScheduler: ...

def upload_chunk_final(folder: Path, repo_id: str, chunk_id: int) -> None:
    """Run upload_large_folder on a single chunk dir for final reconciliation."""
```

---

### 11.15 Source-specific converters

#### `prep/stage_2a_to_lerobot/droid.py`
- Reads via `tfds.load("droid", data_dir="gs://gresearch/robotics")`
- Camera mapping: `wrist_image_left → wrist_rgb`, `exterior_image_1_left → exterior_rgb`
- Action canonical = cartesian_velocity (already 6D + grip)
- Pulls language from `KarlP/droid` (cleaner) — fallback to RLDS field

#### `prep/stage_2a_to_lerobot/agibot2026.py`
- Already in LeRobot v2.1 — primarily a key-renaming + depth-PNG-to-HEVC step
- Camera mapping: `head → head_rgb`, `hand_left → wrist_rgb_left`, `hand_right → wrist_rgb_right`
- Promote `instruction_segments` to top-level columns (level_1, level_2, level_3)
- This is also our source of `subtask_label` ground truth

#### `prep/stage_2a_to_lerobot/robomind.py`
- Per-trajectory HDF5 → parquet + mp4
- **BGR-to-RGB conversion is mandatory** (assert + log)
- Tien Kung head_cam → `head_rgb`
- Drop simulation embodiment (`h5_simulation`) — keep only real

#### `prep/stage_2a_to_lerobot/bridge.py`
- RLDS at `gs://gresearch/robotics/bridge`
- 5 Hz → action_chunk = 4 in canonical
- `image_2 → wrist_rgb` if available; else skip wrist

#### `prep/stage_2a_to_lerobot/oxe_auge.py`
- Use OXE-AugE manifest to filter; drop sources without ego camera
- Per-source action canonicalization rule from manifest

---

### 11.16 `prep/stage_2b_compute_flow.py`

SEA-RAFT inference, batched, fp16. ~3 ms/frame on A100.

### 11.17 `prep/stage_2c_compute_depth.py`

Two paths:
- ZED stereo (DROID `droid_raw`)
- Depth-Anything-V2 fallback for mono-only sources, with low-quality flag in `modality.json`

### 11.18 `prep/stage_3_canonical.py`

Apply `embodiment.json` rules to map every embodiment's `action_native` → `action_canonical_ee[7]`.

### 11.19 `prep/stage_4_dino_cache.py`

Tri-DINO encoding + fp16 + 65-token-keep + safetensors writing. ~25 ms/batch on A100.

### 11.20 `prep/stage_5_validate.py`

The validation gate from §2.7 of the proposal. Pass → mark manifest done. Fail → log, do not mark.

### 11.21 `prep/stage_6_upload.py`

Idempotent `upload_large_folder` per chunk dir.

### 11.22 `prep/dispatch.py`

```python
class SlurmDispatcher:
    def __init__(self, max_pending: int = 64, sif: str = ...): ...
    def step(self) -> None:
        """One pass: scan manifests, find ready chunks, sbatch up to max_pending."""
    def run_forever(self, poll_seconds: int = 60): ...
```

### 11.23 `prep/adapter_pretrain.py`

Phase A.5 step. Trains Tri-DINO depth/flow patch_embed + LoRA on 5M frames sampled across sources. ~1 day on 8×A100. Two losses:
- Cross-modal InfoNCE between RGB-DINO and Depth-DINO at patch level
- MSE between RGB-`[CLS]` and projected Depth-`[CLS]`

Output: `checkpoints/tri_dino_adapter.pt` consumed by Phase B.

---

## 12. Testing strategy

### 12.1 Unit tests (must pass on T0 8×A40 in <10 min)

| Test | What it checks |
|---|---|
| `test_tri_dino.py` | shapes, frozen-vs-trainable params, fp16 cache extraction |
| `test_lora.py` | base equivalence at zero LoRA, gradient routing |
| `test_plan_cache.py` | cached cross-attn = recomputed cross-attn |
| `test_drift.py` | refresh decisions for known scenarios |
| `test_aux_heads.py` | `L_geom` is differentiable; `L_flow-act` shape sanity |
| `test_dataloader.py` | golden fixtures load; mmap cache works |
| `test_canonical_action.py` | each embodiment in `embodiment.json` round-trips |

### 12.2 Integration tests

| Test | Scope |
|---|---|
| `test_smoke_train.py` | 100 steps on `tiny_droid` fixture; loss decreases; no NaN |
| `test_smoke_realtime.py` | 100-step real-time loop; cache refresh fires expected number of times |
| `test_pipeline_end_to_end.py` | Run full prep DAG on `tiny_droid` (5 episodes) → check Hub-shape repo on local fake-Hub |

### 12.3 Slurm rehearsal

Before T2 burst:
- Submit a 3-job rehearsal on T1 with intentional pre-emption every 5 min using `scancel --signal=USR1`
- Verify episode loss < 1 across all interruptions

---

## 13. Milestones and exit criteria

| Week | Milestone | Exit criterion |
|---|---|---|
| 0 | Repo forked + skeleton | All directories created; CI runs unit tests on PR |
| 1 | DROID prep working end-to-end on T0 | `test_pipeline_end_to_end.py` green on tiny fixture |
| 2 | DROID prep deployed to T1; AgiBot 2026 + RoboMIND begun | DROID conversion 100% on Hub; manifest fully `done` |
| 3 | All 6 sources converted | `<org>/usam-*` repos all populated; total ≥ 9,000 hr |
| 4 | Adapter pretrain done | `tri_dino_adapter.pt` committed; downstream ablation shows improvement vs RGB-only |
| 5 | Code freeze; smoke train on T0 | `test_smoke_train.py` green; loss curves look sane |
| 6 | T2 burst | B1+B2+B3 complete; final checkpoint saved |
| 7 | Eval | LIBERO closed-loop > LDA-1B baseline by ≥5% on average |
| 8 | Paper draft | Ablations A1–A3 complete |

---

## 14. File-creation checklist

The following files **must** be created (in this order, ideally) for the project to be complete. Agents implementing this should treat each filename as a "ticket":

```
USAM/
├── README.md                                                  [doc agent]
├── pyproject.toml                                             [infra agent]
├── requirements/{base,prep,train,eval}.txt                    [infra agent]
├── docs/IMPLEMENTATION_PLAN.md                                [this file - already written]
├── docs/DATA_FORMAT.md                                        [doc agent]
├── docs/ARCHITECTURE.md                                       [doc agent]
├── docs/HOWTO_LOCAL_8A40.md                                   [doc agent]
├── docs/HOWTO_SLURM_A100.md                                   [doc agent]
├── docs/HOWTO_H200.md                                         [doc agent]
├── usam/__init__.py                                           [model-architect]
├── usam/encoders/tri_dino.py                                  [model-architect]
├── usam/adapters/lora.py                                      [model-architect]
├── usam/conductor/conductor.py                                [conductor-engineer]
├── usam/conductor/plan_cache.py                               [conductor-engineer]
├── usam/conductor/drift.py                                    [conductor-engineer]
├── usam/conductor/classifier.py                               [conductor-engineer]
├── usam/aux_heads/depth_consistency.py                        [losses-engineer]
├── usam/aux_heads/flow_action.py                              [losses-engineer]
├── usam/losses.py                                             [losses-engineer]
├── usam/dataloader/usam_lerobot.py                            [data-engineer]
├── usam/dataloader/feature_cache.py                           [data-engineer]
├── usam/dataloader/mixtures.py                                [data-engineer]
├── usam/train.py                                              [training-engineer]
├── usam/inference/realtime.py                                 [inference-engineer]
├── usam/inference/openloop.py                                 [inference-engineer]
├── prep/_base.py                                              [pipeline-engineer]
├── prep/_hub.py                                               [pipeline-engineer]
├── prep/_validation.py                                        [pipeline-engineer]
├── prep/stage_0_download/{droid,agibot2026,robomind,bridge,oxe_auge}.py    [pipeline-engineer]
├── prep/stage_1_index.py                                      [pipeline-engineer]
├── prep/stage_2a_to_lerobot/{droid,agibot2026,robomind,bridge,oxe_auge}.py [data-engineer]
├── prep/stage_2b_compute_flow.py                              [data-engineer]
├── prep/stage_2c_compute_depth.py                             [data-engineer]
├── prep/stage_3_canonical.py                                  [data-engineer]
├── prep/stage_4_dino_cache.py                                 [data-engineer]
├── prep/stage_5_validate.py                                   [pipeline-engineer]
├── prep/stage_6_upload.py                                     [pipeline-engineer]
├── prep/dispatch.py                                           [pipeline-engineer]
├── prep/adapter_pretrain.py                                   [model-architect]
├── slurm/{job.sbatch,env.sh,README.md}                        [infra agent]
├── docker/{Dockerfile.local_a40,prep_a100,train_h200,README.md}  [infra agent]
├── configs/data/{droid,agibot2026,robomind,bridge,oxe_auge}.yaml           [data-engineer]
├── configs/data/camera_maps/                                  [data-engineer]
├── configs/model/{usam_1_4b,usam_350m_smoke}.yaml             [model-architect]
├── configs/train/{stage_b1_pretrain,stage_b2_finetune,adapter_pretrain}.yaml     [training-engineer]
├── configs/eval/{libero,realtime}.yaml                        [inference-engineer]
├── tests/unit/test_*.py                                       [test agent + author of each module]
├── tests/integration/test_*.py                                [test agent]
├── tests/golden_data/                                         [test agent]
├── scripts/{prep_run_local,prep_submit_slurm,train_smoke_a40,train_h200,eval_libero}.sh  [infra agent]
├── lda/transformer.py                                         [model-architect]   # +5 LoC
└── lda/heads.py                                               [model-architect]   # +10 LoC
```

Total: ≈ 60 new files, 2 modified files. Distributed across 7 agents (see prompt document).
