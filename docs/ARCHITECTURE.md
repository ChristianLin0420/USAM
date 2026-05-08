# USAM architecture

Source-of-truth pointers in this document use the form
`↳ implemented in <path>:<line range>`. Section ordering follows the
forward pass at training time (encoder → player → cache → losses → train
loop → inference).

## 1. Elevator pitch

> USAM extends LDA-1B's latent-dynamics paradigm by aligning RGB, depth,
> and optical flow in a shared DINOv3 space with cross-modal consistency
> losses, and decouples slow language understanding from fast control via
> a cosine-drift-triggered Plan-KV-Cache, enabling >=3x faster real-time
> WAM inference at no quality cost.

The repository is a thin overlay on LDA-1B (~2,000 LoC delta). Everything
new lives under `usam/` (runtime) and `prep/` (Phase A pipeline); the
upstream `lda/` directory carries only two surgical edits documented in
section 3 below.

↳ canonical reference: [`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §1.3.

---

## 2. Tri-DINO Tower

The Tri-DINO tower is one frozen DINOv3 ViT-B/14 (or ViT-L/14) backbone
with three input adapters and modality-aware LoRA paths inside its
attention blocks. Each modality (`rgb`, `depth`, `flow`) has its own
`Conv2d` patch_embed accepting a different number of input channels.

**Patch embedding initialization** — the depth and flow patch_embeds are
seeded from the RGB conv weights so they start near a useful subspace:

* `depth_patch.weight` = `mean(rgb_patch.weight, dim=1, keepdim=True)`
  (one input channel; the bias is copied verbatim).
* `flow_patch.weight` = `rgb_patch.weight[:, :2]` (two input channels;
  bias copied).

All three patch_embeds remain trainable; the rest of the backbone is
frozen except the LoRA A/B matrices.

**Modality-aware LoRA routing** — every Q/K/V `nn.Linear` in the
backbone is wrapped by `LoRALinear` (one A/B pair per modality). The
tower routes the active modality by writing a single attribute
(`module._usam_active_modality`) on each `LoRALinear` and monkey-patching
its `forward` to dispatch on that attribute. This is a per-instance
stateful pattern — **do not call `forward(modality_a)` and
`forward(modality_b)` concurrently on the same encoder instance**; call
modalities sequentially. The realtime loop and the Phase A DINO cacher
both honor this contract.

**Image size** — the canonical input is `image_size=378` (not 384).
`27 × 14 = 378` gives an exact `27 × 27 = 729` ViT-B/14 patch grid; the
cache keeps the [CLS] token plus the first 64 patch tokens (65 total) per
modality. The plan refers to "384²" colloquially; the YAML configs and
the `cache_n_keep_tokens=64` setting are the binding contract. The
`TriDinoConfig` dataclass default is 384 but every shipped model YAML
overrides it to 378.

**Frozen-vs-trainable invariants** — the constructor walks
`self.backbone.parameters()` once and disables gradients, then re-enables
them on the three patch_embeds and (via `apply_lora`) the LoRA matrices.
`trainable_parameters()` and `lora_parameters()` are public iterators for
the optimizer-group setup.

`extract_features(x, modality, n_keep_tokens=64)` is the cache-extraction
entry point: it returns `[CLS] | first n_keep_tokens patch tokens` in
fp16 (register tokens dropped), suitable for direct write into the
safetensors feature shards consumed by `usam.dataloader.feature_cache`.

↳ implemented in `usam/encoders/tri_dino.py:256-532` (`TriDINOTower`,
`TriDinoConfig`, `MiniDinoBackbone`).

---

## 3. MM-DiT modifications

Three small, gated extensions to the upstream MM-DiT block / model live
in a single LDA-1B file (`mmdit_cross_attn.py`). All three are no-ops
when their config flag is `False`, so unmodified LDA-1B call sites
continue to work bit-exactly.

1. **proprio_proj into AdaLN-Zero.** When `enable_proprio_cond=True`,
   the model adds a third modulation source `proprio_proj(proprio)` to
   the AdaLN-Zero `time_cond`. The projection is a single `nn.Linear`
   from `proprio_dim` (default 50) into the MM-DiT inner dim. The
   smoke and 1.4B model configs both set this flag to `true`.

2. **+depth/flow heads.** The image-branch trunk features are projected
   through two extra `nn.Linear` heads when `enable_depth_head=True` /
   `enable_flow_head=True`. Output dim matches the existing
   `image_proj_out`. The forward returns a dict with
   `image_tokens` / `action_tokens` / `depth_tokens` / `flow_tokens`
   when either head is enabled, and the original `(image_pred,
   action_pred)` tuple otherwise.

3. **+`kv_cache=` kwarg on each block.** When passed, the cross-
   attention `to_k` / `to_v` projections are skipped — pre-projected
   tensors are looked up via `kv_cache[(layer_idx, branch)]` for
   `branch ∈ {"image", "action"}`. The lookup helper accepts either a
   plain `Mapping` or a `PlanCache`-like object with a
   `get(layer_idx, branch)` method. The `kv_cache=None` branch is
   bit-exact identical to pre-USAM behaviour.

↳ implemented in
`lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py:54-628`
(`_kv_from_cache`, `_cached_cross_attention`, `MMDiTBlock.forward`,
`MMDiT.__init__`, `MMDiT.forward`).

---

## 4. Conductor + PlanCache

The Conductor / Player split decouples slow language-vision
understanding (Qwen3-VL-4B) from the high-rate Player MM-DiT.

**Conductor** wraps a frozen Qwen3-VL-4B and exposes `encode(observation,
instruction) -> ConductorOutput(e, P_hat, e_raw)`. The plan tokens
`P_hat` are the **last** `n_plan_tokens` (=32 by default) of the
backbone's hidden state, projected through `plan_proj` into the Player's
`d_model`. The drift reference embedding `e` is the [EOS] hidden state,
projected through `e_proj` to a small (default 64-D) space then
L2-normalized. A `MockConductorBackbone` (~5 K params) is shipped for
unit testing and the smoke train.

↳ implemented in `usam/conductor/conductor.py:124-330` (`Conductor`,
`ConductorOutput`, `MockConductorBackbone`).

**PlanCache** stores pre-projected K/V across all Player layers, for
**both** the image and action cross-attention branches. The cache is
indexed as `cache.get(layer_idx, branch)` with `branch ∈ {"image",
"action"}`. `refresh(p_hat, e, k_projs_image, v_projs_image,
k_projs_action, v_projs_action, t)` walks every layer once, projects
`P_hat` through that layer's K/V linears, and stores the result in
bf16 (configurable). A bounded ring buffer of refreshed snapshots
(`history_size=8` by default) backs the training-time cache-dropout
helper. `load_state(snapshot)` and `snapshot()` round-trip an immutable
`PlanCacheState` for that purpose.

↳ implemented in `usam/conductor/plan_cache.py:50-308` (`PlanCache`,
`PlanCacheState`).

**Cosine-drift trigger.** The Player runs every step; the Conductor
runs only when the drift trigger fires. The trigger uses cosine distance
on **L2-normalized [EOS] embeddings**, NOT raw KL on softmaxed hidden
states — the docstring quotes the SemEval-2020 distributional-shift
detection literature for the rationale. The cheap `f_drift` MLP
(under 100 K params) predicts `e_now` from `(rgb_dino_cls, e_committed)`
and `should_refresh` returns True when ANY of:

* `episode_start` (or `t == 0` if not specified),
* `last_refresh_t < 0` (no refresh ever),
* `t - last_refresh_t >= timer_hard` (default 60 frames = 2 s @ 30 Hz),
* `d_t > tau_hard` (default 0.20),
* `d_t > tau_soft` (default 0.06) AND `t - last_refresh_t >= timer_soft`
  (default 30 frames = 1 s),
* `subtask_completion_logit > 0` from the classifier head.

`calibrate_taus(drift_log)` sets `tau_hard` to the empirical P90 and
`tau_soft` to the P50 from a held-out cross-subtask drift sample.

↳ implemented in `usam/conductor/drift.py:50-326` (`DriftConfig`,
`FDriftMLP`, `should_refresh`, `calibrate_taus`, `cosine_distance`).

**Subtask-completion classifier.** Two-layer MLP that pools a 16-frame
window of `[CLS]` + proprio (mean-pooled) and concatenates with the
current `e_t`. Positive logit feeds back into `should_refresh` to
bracket subtask boundaries. Trained jointly with `L_subtask = bce(...)`,
weighted at 0.1.

↳ implemented in `usam/conductor/classifier.py:36-127`
(`SubtaskCompletionHead`).

**Cache dropout.** With probability `p` (default 0.5), the training-time
helper substitutes the live cache with a uniformly-random snapshot
within a bounded staleness window (default 60 frames). Implemented as a
non-mutating clone — the live cache is never poisoned with stale state.

↳ implemented in `usam/conductor/cache_dropout.py:46-118`
(`apply_cache_dropout`).

---

## 5. Auxiliary losses

**`L_geom` — soft Spearman rank consistency.** A frozen DAv2-distill
MLP decodes predicted depth-DINO patches to an inverse-depth proxy; a
frozen near-field RGB prototype decodes predicted RGB-DINO patches to
a cosine similarity. The two per-patch scalar streams are compared via
a hand-rolled differentiable Spearman rank correlation (sigmoid pairwise
relaxation, NOT a dependency on the `differentiable-rank` package). The
returned loss is `-rho`, bounded in `[-1, 1]`. The two frozen heads are
seeded from optional checkpoints (`dav2_distill_ckpt`,
`nearfield_proto_ckpt`); when absent a warning is emitted (test-only
path).

↳ implemented in `usam/aux_heads/depth_consistency.py:43-329`
(`GeomConsistencyLoss`, `soft_rank`, `soft_spearman`).

**`L_flow-act` — forward-action consistency.** A trainable 2-layer MLP
`g_phi(proprio, action_chunk_flat) -> scalar` predicts the mean flow
magnitude that the action chunk should produce. The empirical target is
computed from the predicted flow-DINO patch tokens via a fixed
deterministic decode head: a unit-norm mixing weight (seeded with a
fixed RNG, registered as a non-persistent buffer) gives a per-patch
scalar, squared (smooth, non-negative), then mean-pooled over patches.
Loss = MSE(pred, target). The decode head is differentiable — gradients
flow through it back into `flow_dino_pred` so the consistency objective
also pulls the flow head.

↳ implemented in `usam/aux_heads/flow_action.py:32-223`
(`FlowActionConsistencyLoss`, `flow_magnitude`).

**Unified loss aggregator.** The `LossWeights` dataclass enumerates the
exact field names consumed downstream:
`action`, `rgb`, `depth`, `flow`, `geom`, `flow_act`, `drift`,
`subtask`. `USAMUnifiedLoss.forward(predictions, targets, masks)`
returns `(total_loss, per_loss_dict)` where the per-loss dict keys
**exactly** match the `LossWeights` fields. The aggregator accepts both
`predictions["rgb"]` and `predictions["image"]` for the RGB head (the
plan calls it `rgb`, the MM-DiT exposes `image_proj_out` / `image`),
and computes a zero scalar (on the right device/dtype) for any aux
loss whose inputs are absent so the per-loss log dict is always
populated.

↳ implemented in `usam/losses.py:69-356` (`LossWeights`,
`USAMUnifiedLoss`).

---

## 6. Training loop

`usam.train` wraps LDA-1B's optimizer/scheduler scaffolding and adds
USAM-specific wiring on top of the LDA-1B Player. The smoke build
substitutes `SmokePlayer` (a ~5 M-param Transformer encoder) for the
real MM-DiT so the CPU plumbing test can run without flash-attn or
transformer-engine.

**Cache-dropout call site.** Plan §6.2 specifies that the Player must
learn robustness against plans up to `plan_stale_window_frames` stale.
The dropout call lives at
`usam/_train_helpers.py:680-685` inside `USAMTrainModel.training_step`,
**immediately after** every `_refresh_plan_cache(...)` call:

```python
self._refresh_plan_cache(head_keyframe, t=step)
active_cache = apply_cache_dropout(
    self.plan_cache,
    t=step,
    p=self.cfg.cache_dropout_p,
    window=self.cfg.cache_dropout_window,
)
```

The history is populated automatically by `PlanCache.refresh`; the
training loop just calls `apply_cache_dropout(cache, t=step)`.

↳ implemented in `usam/_train_helpers.py:670-744`
(`USAMTrainModel.training_step`).

**Ramped loss weights.** `compute_ramped_weights(base, step,
geom_target, flow_act_target, ramp_steps=50_000)` linearly ramps the
`geom` and `flow_act` weights from 0 to their YAML targets across the
first 50,000 steps:

```
step <= 0         -> geom = flow_act = 0.0
0 < step < 50_000 -> w = target * step / 50_000
step >= 50_000    -> w = target  (clamped)
```

The other six weights pass through unchanged. The plan §4.3 mentions a
50K–100K ramp; the implementation uses a single `ramp_steps` knob
(default 50,000) for both losses.

↳ implemented in `usam/_train_helpers.py:184-243`
(`compute_ramped_weights`).

**Precision detection.** `detect_precision(force_cpu)` returns a
`PrecisionPlan` based on `torch.cuda.get_device_capability()`:

* H200 (`(9, 0)`) + `transformer_engine` importable → BF16 weights +
  TE FP8 activations.
* Any other CUDA → BF16 weights, no FP8.
* No CUDA / `force_cpu=True` → FP32 weights, no FP8.

↳ implemented in `usam/_train_helpers.py:137-178` (`detect_precision`,
`PrecisionPlan`).

**Checkpoint format.** Every `every_steps` (=5,000 by default) the
`CheckpointManager` writes
`<output_dir>/checkpoints/checkpoint_step{step:08d}.pt` and updates
`<output_dir>/checkpoints/latest_step.txt` (a side-channel marker for
external watchers). The payload is a dict with `state_dict`,
`optimizer`, `scheduler`, `step`, `run` (the `RunMetadata` dataclass
with `run_id` = `<YYYYmmdd-HHMMSS>-<short-uuid>` and `git_sha` from
`git rev-parse --short HEAD`), `best_val_loss`, `val_loss`, and
`timestamp`. `keep_last=3` non-best checkpoints are retained; the best
by validation loss is saved separately as `checkpoint_best.pt`.

↳ implemented in `usam/_train_helpers.py:750-857` (`CheckpointManager`).

---

## 7. Inference

**Realtime controller.** `RealtimeController.step(rgb, depth?, flow?,
proprio, instruction?)` implements the §4.4 pseudocode:

1. encode each modality through Tri-DINO,
2. predict `e_now_estimate = f_drift(rgb_dino_cls, e_committed)`,
3. compute `d_t = 1 - cos(e_committed, e_now_estimate)`,
4. ask `should_refresh(...)`; if True, run the Conductor and refresh
   the PlanCache,
5. denoise an action chunk via `player(rgb_dino, depth_dino, flow_dino,
   proprio, plan_cache, n_steps=10)` — the Player consumes the cache
   via the `kv_cache=` kwarg into the MM-DiT.

The controller is agnostic to the Player's exact signature; the smoke
test injects a mock player. The real MM-DiT wiring matches the
`kv_cache[(layer_idx, branch)]` contract documented in section 3.

↳ implemented in `usam/inference/realtime.py:97-385`
(`RealtimeController`, `StepResult`).

**Open-loop ADE.** `run_openloop_eval(policy, dataset, action_chunk=16,
device="cuda")` is currently a stub: the API contract is fixed
(`OpenLoopMetrics(ade, fde, n_samples, per_horizon_l2)`) but the body
raises `NotImplementedError`. The Wave 4 inference-engineer fills it in
once the dataloader API and Player wiring are finalized.

↳ implemented in `usam/inference/openloop.py:19-81`
(`OpenLoopMetrics`, `run_openloop_eval`).
