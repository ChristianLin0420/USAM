"""Long-run smoke train on real DROID chunk via torchrun (8-GPU DDP).

Usage (inside the prep container):

  torchrun --standalone --nproc-per-node=8 \
      /workspace/USAM/tools/train/smoke_train_long.py

Honors these env vars:
  WANDB_API_KEY        Set to push to wandb. Without it, stdout-only.
  WANDB_PROJECT        Defaults to "usam".
  USAM_VIZ_INTERVAL    How often to log media (DINOv3 PCA panel). Default 100.
                       Set higher (1000-5000) for production runs to avoid
                       wasting time on visualization.
  USAM_MAX_STEPS       Override the default 2000 steps for shorter/longer runs.
  USAM_RAMP_STEPS      Override the loss-weight ramp (default 50_000).
  USAM_OUTPUT_DIR      Where to write checkpoints + logs. Default /workspace/output/train_long.

Each rank runs this script via torchrun. Rank 0 builds the LeRobot v2.1
layout (idempotent — skipped if present), then all ranks reach a
barrier before training so the dataloader sees a complete layout.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path


STAGED = Path("/workspace/output/staged")
DINO_CACHE = Path("/workspace/output/dino_cache")
LEROBOT = Path("/workspace/output/lerobot_v2_1")


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _build_lerobot_layout_if_missing() -> None:
    """Rank-0-only: assemble LeRobot v2.1 root from prep outputs."""
    info_path = LEROBOT / "meta" / "info.json"
    if info_path.exists():
        return  # already built

    import pyarrow as pa
    import pyarrow.parquet as pq

    LEROBOT.mkdir(parents=True, exist_ok=True)
    (LEROBOT / "meta").mkdir(parents=True, exist_ok=True)
    (LEROBOT / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Symlink the staged parquet as file-000.parquet (loader expects 3-digit ids).
    src_parquet = next(iter((STAGED / "data" / "chunk-000").glob("file-*.parquet")))
    dst_parquet = LEROBOT / "data" / "chunk-000" / "file-000.parquet"
    if dst_parquet.exists() or dst_parquet.is_symlink():
        dst_parquet.unlink()
    dst_parquet.symlink_to(src_parquet.resolve())

    # Read parquet for episode metadata
    tbl = pq.read_table(str(src_parquet)).to_pandas()
    ep_lengths = tbl.groupby("episode_index").size()
    n_episodes = int(ep_lengths.shape[0])
    avg_len = int(ep_lengths.mean())

    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "fps_features": 5,
        "source": "droid",
        "embodiment": "droid_franka",
        "n_episodes": n_episodes,
        "n_frames_per_episode": avg_len,
    }
    info_path.write_text(json.dumps(info, indent=2))

    eps = [
        {"episode_index": int(ep), "length": int(ep_lengths[ep]),
         "chunk": 0, "file": 0, "embodiment": "droid_franka"}
        for ep in sorted(ep_lengths.index)
    ]
    pq.write_table(pa.Table.from_pylist(eps), str(LEROBOT / "meta" / "episodes.parquet"))

    feat_root = LEROBOT / "features"
    for cam_dir in sorted(DINO_CACHE.iterdir()):
        cam = cam_dir.name
        for mod_dir in sorted(cam_dir.iterdir()):
            mod = mod_dir.name
            chunk_dir = mod_dir / "chunk-000"
            dst_dir = feat_root / cam / mod / "chunk-000"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for shard in sorted(chunk_dir.glob("file-*.safetensors")):
                dst = dst_dir / shard.name
                if not dst.exists():
                    dst.symlink_to(shard.resolve())

    print(f"[rank 0] built LeRobot v2.1 layout at {LEROBOT}", flush=True)


def _maybe_init_dist() -> None:
    """torch.distributed init if RANK / WORLD_SIZE / LOCAL_RANK are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {os.environ.get('RANK', '0')}] %(levelname)s %(message)s",
    )
    warnings.filterwarnings("ignore")

    print(f"RANK={os.environ.get('RANK', '0')} "
          f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '0')} "
          f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}", flush=True)

    _maybe_init_dist()

    if _is_rank_zero():
        _build_lerobot_layout_if_missing()

    # Barrier so other ranks wait for rank 0's setup before reading.
    try:
        import torch
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception as e:  # pragma: no cover
        print(f"[rank {os.environ.get('RANK', '?')}] barrier skipped: {e}", flush=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from usam.train import TrainArgs, load_yaml, run as train_run

    repo = Path("/workspace/USAM")
    max_steps = int(os.environ.get("USAM_MAX_STEPS", "2000"))
    output_dir = Path(os.environ.get("USAM_OUTPUT_DIR", "/workspace/output/train_long"))

    args = TrainArgs(
        config=repo / "configs" / "train" / "stage_b1_pretrain.yaml",
        model_config=repo / "configs" / "model" / "usam_1_4b.yaml",
        data=LEROBOT,
        output_dir=output_dir,
        max_steps=max_steps,
        device="auto",
        seed=0,
        auto_oom_reduce=False,
        log_every=10,
    )

    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)

    # Match our real DROID chunk's actual feature fps + the loader's 8-frame
    # action chunk emission (the 1.4B config's action_horizon=16 doesn't
    # match what the loader produces today).
    train_cfg.setdefault("data", {})["fps_action"] = 15
    train_cfg["data"]["fps_features"] = 5
    model_cfg.setdefault("action_head", {})["action_horizon"] = 8
    # Aux-head ramp: optional override (default 50_000 from the YAML).
    ramp_steps_override = os.environ.get("USAM_RAMP_STEPS")
    if ramp_steps_override is not None:
        train_cfg.setdefault("loss_weights", {})["ramp_steps"] = int(ramp_steps_override)
    # Print the viz interval for visibility in stdout.
    print(f"USAM_VIZ_INTERVAL={os.environ.get('USAM_VIZ_INTERVAL', '100')}", flush=True)

    losses = train_run(args, train_cfg, model_cfg)
    if _is_rank_zero():
        print(f"\ndone. {len(losses)} steps. "
              f"first={losses[0]:.3f} last={losses[-1]:.3f} "
              f"min={min(losses):.3f} mean_last_50={sum(losses[-50:])/min(50, len(losses)):.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
