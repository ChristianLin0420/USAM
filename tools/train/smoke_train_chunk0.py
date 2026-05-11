"""Wave J: construct LeRobot v2.1 layout from our real DROID outputs, then
run usam.train for ~20 steps.

We have:
  /workspace/output/staged/ep_<hash>/   per-episode artifacts (npy + meta)
  /workspace/output/staged/data/chunk-000/file-da17a5.parquet  per-frame parquet
  /workspace/output/dino_cache/<cam>/<mod>/chunk-000/file-*.safetensors  features

Need to assemble a LeRobot v2.1 root at /workspace/output/lerobot_v2_1/:
  meta/info.json
  meta/episodes.parquet
  data/chunk-000/file-*.parquet                     (the staged parquet)
  features/<cam>/<mod>/chunk-000/file-*.safetensors  (the DINO cache)
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import warnings
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    staged = Path("/workspace/output/staged")
    dino_cache = Path("/workspace/output/dino_cache")
    out = Path("/workspace/output/lerobot_v2_1")

    print("=" * 60, flush=True)
    print("Wave J: build LeRobot v2.1 root + run smoke train", flush=True)
    print("=" * 60, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ---------- inventory + episode meta ----------
    import pyarrow.parquet as pq
    src_parquet = next(iter((staged / "data" / "chunk-000").glob("file-*.parquet")))
    # The dataloader keys filenames on episodes.parquet's integer `file`
    # column → file-{N:03d}.parquet. Our DROID shard is hash-suffixed
    # (file-da17a5.parquet); symlink it as file-000.parquet for the loader.
    dst_parquet = out / "data" / "chunk-000" / "file-000.parquet"
    if dst_parquet.exists() or dst_parquet.is_symlink():
        dst_parquet.unlink()
    dst_parquet.symlink_to(src_parquet.resolve())
    print(f"data parquet: {dst_parquet}", flush=True)

    tbl = pq.read_table(str(src_parquet))
    df = tbl.to_pandas()
    ep_lengths = df.groupby("episode_index").size()
    n_episodes = int(ep_lengths.shape[0])
    avg_len = int(ep_lengths.mean())
    print(f"episodes: {n_episodes}, avg_len={avg_len}, total_frames={int(ep_lengths.sum())}", flush=True)

    # meta/info.json
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "fps_features": 5,
        "source": "droid",
        "embodiment": "droid_franka",
        "n_episodes": n_episodes,
        "n_frames_per_episode": avg_len,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"info.json: {info}", flush=True)

    # meta/episodes.parquet
    import pyarrow as pa
    eps = [
        {"episode_index": int(ep), "length": int(ep_lengths[ep]),
         "chunk": 0, "file": 0, "embodiment": "droid_franka"}
        for ep in sorted(ep_lengths.index)
    ]
    eps_tbl = pa.Table.from_pylist(eps)
    pq.write_table(eps_tbl, str(out / "meta" / "episodes.parquet"))
    print(f"episodes.parquet: {len(eps)} rows", flush=True)

    # ---------- features symlinks ----------
    feat_root = out / "features"
    for cam_dir in sorted(dino_cache.iterdir()):
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
            print(f"  features/{cam}/{mod}: {len(list(dst_dir.glob('file-*.safetensors')))} shards", flush=True)

    # ---------- run training ----------
    print("\n" + "=" * 60, flush=True)
    print("running usam.train with usam_1_4b (matches 1024-d ViT-L/16 cache)", flush=True)
    print("=" * 60, flush=True)

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from usam.train import TrainArgs, load_yaml, run as train_run

    repo = Path("/workspace/USAM")
    args = TrainArgs(
        config=repo / "configs" / "train" / "stage_b1_pretrain.yaml",
        model_config=repo / "configs" / "model" / "usam_1_4b.yaml",
        data=out,
        output_dir=Path("/workspace/output/train_smoke"),
        max_steps=20,           # smoke gate: 20 steps is enough to validate wiring
        device="auto",
        seed=0,
        auto_oom_reduce=False,
        log_every=1,
    )
    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    # Use the real-data feature fps from our chunk (5).
    train_cfg.setdefault("data", {})["fps_action"] = 15
    train_cfg["data"]["fps_features"] = 5
    # Match the dataloader's 8-frame action chunk (the loader uses
    # history_frames=4 + action_chunk=8 = 12 by default on the 5-fps
    # feature track; the 1.4B production config's action_horizon=16
    # would need the loader to emit 16-frame chunks instead). The
    # flow_act aux head is constructed with action_chunk_dim =
    # action_dim * action_horizon, so 7 * 8 = 56 matches the loader's
    # output and avoids the assertion mismatch at flow_action.py:201.
    model_cfg.setdefault("action_head", {})["action_horizon"] = 8

    losses = train_run(args, train_cfg, model_cfg)
    print(f"\ntrain ran; got {len(losses)} loss values", flush=True)
    if losses:
        print(f"  first: {losses[0]:.4f}", flush=True)
        print(f"  last:  {losses[-1]:.4f}", flush=True)
        print(f"  mean:  {sum(losses) / len(losses):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
