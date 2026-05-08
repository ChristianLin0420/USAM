# HOWTO — H200 burst training (T2)

This guide covers the 500×H200 1-week burst pretrain (the T2 tier). The
plan reference is [`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
§6 (pretraining) and §9.3 (H200 image).

---

## 1. Build the train image

```bash
docker build -f docker/Dockerfile.train_h200 -t usam:train-h200 .
```

The base is `nvcr.io/nvidia/pytorch:25.01-py3` (per
[`docker/README.md`](../docker/README.md)) which ships with
`transformer-engine` pre-installed against matched CUDA/cuDNN. The
Dockerfile additionally installs `requirements/base.txt` +
`requirements/train.txt` and `flash-attn==2.6.3` via `pip install
--no-build-isolation`.

Push the resulting image to your cluster registry so each H200 node can
pull it (substitute your registry URL):

```bash
docker tag usam:train-h200 <registry>/usam:train-h200
docker push <registry>/usam:train-h200
```

---

## 2. FP8 + Transformer Engine activation

FP8 is **autodetected** at runtime — there is no manual flag.
`usam._train_helpers.detect_precision()`
(`usam/_train_helpers.py:137-178`) inspects
`torch.cuda.get_device_capability()` and enables TE FP8 only when:

1. capability is `(9, 0)` (Hopper / H200), AND
2. `import transformer_engine.pytorch` succeeds.

Otherwise (A40 / A100 / older Hopper without TE installed / CPU) the
plan falls back to BF16 weights with no FP8. The decision is logged at
startup as `"GPU cap=... BF16 weights"` or `"H200 cap=(9, 0) BF16
weights + TE FP8 activations"` so it's easy to verify.

The `precision:` block in
`configs/train/stage_b1_pretrain.yaml:23-26` records the **request**
(`activations_te_fp8: true`); the runtime detector enforces hardware
gating on top.

---

## 3. FSDP / TP layout

`usam.train.maybe_wrap_distributed`
(`usam/train.py:194-231`) is the entry point for distributed wrapping.
It wraps with HuggingFace `accelerate` (DeepSpeed plugin) when:

1. CUDA is available,
2. `torchrun` populated `RANK` and `WORLD_SIZE`, and
3. `accelerate` imports cleanly.

Otherwise it is a no-op (single-GPU) or a `.to(device, dtype)` move.

The `parallelism:` block in
`configs/train/stage_b1_pretrain.yaml:18-21` is the H200 production
layout:

```yaml
parallelism:
  data_parallel: 250         # 500 H200 / 2 TP
  tensor_parallel: 2
  pipeline_parallel: 1
```

---

## 4. Run Stage B1 pretrain

`scripts/train_h200.sh` **generates** an sbatch file but does **not**
submit it (the team-lead reviews and runs `sbatch` manually). Usage:

```bash
bash scripts/train_h200.sh stage_b1   # writes runs/h200_stage_b1.sbatch
bash scripts/train_h200.sh stage_b2   # writes runs/h200_stage_b2.sbatch
```

After review:

```bash
sbatch runs/h200_stage_b1.sbatch
```

The generated sbatch file:

* targets `--nodes=125 --ntasks-per-node=1 --gres=gpu:h200:4` (4 GPUs
  per node × 125 nodes = 500 H200, matching the parallelism block);
* uses `--signal=B:USR1@600 --requeue` for graceful preemption (same
  pattern as the Slurm A100 prep jobs);
* sets the H200 + Mellanox NCCL env vars
  (`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=^lo,docker0`,
  `NCCL_ASYNC_ERROR_HANDLING=1`);
* runs `torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
  -m usam.train --config configs/train/stage_b1_pretrain.yaml
  --model configs/model/usam_1_4b.yaml --data $DATA_REPO --output_dir
  runs/usam_stage_b1-$SLURM_JOB_ID`.

Override the data path, walltime, or partition via env vars before
running the script:

```bash
DATA_REPO=datasets/usam_pretrain_mixture \
WALLTIME=7-00:00:00 \
PARTITION=h200 \
ACCOUNT=usam \
bash scripts/train_h200.sh stage_b1
```

---

## 5. Checkpoint resume

The `CheckpointManager`
(`usam/_train_helpers.py:750-857`) writes:

```
<output_dir>/checkpoints/checkpoint_step{step:08d}.pt   # rolling, keep_last=3
<output_dir>/checkpoints/checkpoint_best.pt             # best by val loss
<output_dir>/checkpoints/latest_step.txt                # side-channel marker:
                                                        # "<step>\n<filename>\n"
```

Each `.pt` is a single torch `save_file` payload with:

* `state_dict` — model state dict;
* `optimizer` — optimizer state dict;
* `scheduler` — LR scheduler state dict (or `None`);
* `step` — global step at save time;
* `run` — `RunMetadata` dict with `run_id` (=
  `<YYYYmmdd-HHMMSS>-<short-uuid>`), `git_sha` (= `git rev-parse --short
  HEAD` at training start, `"unknown"` when unavailable), `config_path`,
  `started_at`;
* `best_val_loss`, `val_loss`, `timestamp`.

There is no shipped `load_checkpoint` helper today — resume is the
team-lead's manual reload via `torch.load(path)` followed by
`model.load_state_dict(payload["state_dict"])` /
`optimizer.load_state_dict(payload["optimizer"])`. Wave 4 will
formalize this into a `--resume <path>` CLI flag on `usam.train`.

To find the latest checkpoint:

```bash
cat <output_dir>/checkpoints/latest_step.txt
```

This file is touched after every successful save and lets external
watchers (e.g. periodic eval jobs) poll without parsing filenames.

---

## 6. WandB integration

WandB is **optional**. The training loop logs through Python's
`logging` module by default; setting the `WANDB_API_KEY` env var
opts in to WandB-backed logging via `accelerate`'s tracker integration.

```bash
export WANDB_API_KEY=<your-wandb-key>
sbatch runs/h200_stage_b1.sbatch
```

When `WANDB_API_KEY` is unset (the default for the CPU plumbing test
and most smoke runs), the run logs to stdout only, the run dir is the
sole artifact, and no network calls are attempted.
