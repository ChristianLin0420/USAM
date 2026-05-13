# USAM-LeRobot v2.1 data format

USAM extends the LeRobot v2.1 layout with a per-modality fp16 DINO
feature cache and a small set of subtask-aware metadata columns. This
document is the binding contract between Phase A (`prep/`) writers and
Phase B (`usam/dataloader/`) readers. The plan reference is
[`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §5.

---

## 1. Repository layout on HF Hub

One repo per source. Internal layout:

```
<org>/usam-<source>/
├── meta/
│   ├── info.json              codebase_version=v2.1, fps, features
│   ├── modality.json          USAM extension (state/action dims per channel)
│   ├── tasks.parquet          episode -> task descriptions
│   ├── episodes.parquet       episode_index, length, embodiment, ...
│   ├── stats.safetensors      normalization stats
│   ├── embodiment.json        per-embodiment action canonicalization rule
│   └── conversion_log.jsonl   per-episode success/failure
├── data/
│   └── chunk-{000..NNN}/
│       └── file-{000..999}.parquet
├── videos/
│   ├── observation.images.head_rgb/chunk-XXX/file-YYY.mp4
│   ├── observation.images.head_depth/...               16-bit HEVC
│   ├── observation.images.head_flow/...                HSV-encoded h264
│   ├── observation.images.wrist_rgb/...
│   ├── observation.images.wrist_depth/...
│   └── observation.images.wrist_flow/...
└── features/                                           USAM cache layer
    ├── rgb/chunk-XXX/file-YYY.safetensors              fp16 [T, 65, D]
    ├── depth/chunk-XXX/file-YYY.safetensors
    └── flow/chunk-XXX/file-YYY.safetensors
```

Per-source repos: `<org>/usam-droid`, `<org>/usam-agibot2026`,
`<org>/usam-robomind`, `<org>/usam-bridge`, `<org>/usam-oxe-auge`.

---

## 2. ConversionResult — the unified internal record

Every Stage-2a converter (`prep/stage_2a_to_lerobot/<source>.py`)
returns the same dataclass before sharding. The fields below are the
**actual** fields shipped today (defined in the converters, not in
`prep/_base.py`):

```python
@dataclass
class ConversionResult:
    episode_index: int
    embodiment: str                       # one of the 6 embodiment keys
    fps: int                              # native action fps for this source
    cameras: dict                         # canonical_key -> np.ndarray [T, H, W, 3] uint8
    depth: dict                           # canonical_key -> np.ndarray [T, H, W] uint16
    state: np.ndarray                     # [T, 50] padded proprio
    state_mask: np.ndarray                # [50] bool — which dims are valid
    action_native: np.ndarray             # [T, 32] padded native-frame action
    action_mask: np.ndarray               # [32] bool
    action_canonical_ee: np.ndarray       # [T, 7] canonical EE-velocity frame
    instructions: dict                    # level_1 / level_2 / level_3 (lists of str)
    force_torque: Optional[np.ndarray]    # [T, 6] — preserved when present
    timestamps: np.ndarray                # [T] float32, seconds
    raw_meta: dict                        # source-specific bookkeeping
```

This record is canonical for the converter layer; it is then exploded
into the parquet shards (one row per frame) for the on-disk layout
described in §3 below.

> Two notes on the in-tree code:
> * The base class `prep/_base.ConversionResult` (line 119) is a more
>   permissive shape (`episode_index`, `episode_id`, `payload: dict`)
>   used by the Stage 1 / Stage 6 plumbing. The richer dataclass above is
>   the one shipped by every source converter
>   (`prep/stage_2a_to_lerobot/droid.py:49-69` and re-imported by
>   `agibot2026.py`, `robomind.py`, `bridge.py`).
> * `EpisodeRef` likewise has two shapes in the codebase
>   (`prep/_base.py:89-115` exposes `episode_id, source, raw_path,
>   extra`; the per-source converters use `episode_index, source,
>   shard_hash`). Treat the per-source flavor as the binding contract
>   for Stage 2a — the discrepancy is on the team-lead's clean-up list.

↳ implemented in `prep/stage_2a_to_lerobot/droid.py:49-79`
(re-exported by every other Stage-2a converter).

---

## 3. On-disk parquet schema (`data/chunk-XXX/file-YYY.parquet`)

Per-frame columns expected by `usam.dataloader.usam_lerobot`:

| Column | Dtype | Shape | Notes |
|---|---|---|---|
| `episode_index` | int64 | scalar | grouping key; the loader filters by this |
| `proprio` | fp32 | `[50]` | padded; valid mask is `state_mask` |
| `state_mask` | bool | `[50]` | replicated per row |
| `action_native` | fp32 | `[32]` | padded |
| `action_mask` | bool | `[32]` | replicated per row |
| `action_canonical_ee` | fp32 | `[7]` | canonical EE velocity + gripper |
| `timestamps` | fp32 | scalar | seconds |
| `level_1` / `level_2` / `level_3` | str | scalar | subtask-segment text |
| `subtask_label` | bool | scalar | True at subtask boundary frames |
| `instruction` | str | scalar | optional, primary text instruction |

The dataloader is permissive: missing columns are zero-filled. AgiBot
World 2026 promotes `instruction_segments` to `level_1` / `level_2` /
`level_3` per `prep/stage_2a_to_lerobot/agibot2026.py:9-22`.

↳ implemented in `usam/dataloader/usam_lerobot.py:60-417`
(`_load_episode_frames`, `USAMLeRobotDataset.__getitem__`).

---

## 4. mp4 / HEVC encoding specs

| Modality | Resolution | Codec | Pixel format | Quality | fps |
|---|---|---|---|---|---|
| RGB (`head_rgb`, `wrist_rgb*`) | 378×378 | h264 | yuv420p | crf=23 | source (5–30) |
| Depth (`*_depth`) | 192×192 | HEVC | gray16le | 16-bit | typically 15 |
| Flow (`*_flow`) | 378×378 | h264 (HSV-encoded) | yuv420p | crf=23 | source |

**Why 378, not 384?** The Tri-DINO encoder uses ViT-B/14 (or ViT-L/14)
with patch size 14. `27 × 14 = 378`, giving `27² = 729` patch tokens, of
which the cache keeps 64 plus the [CLS] token (65 total). The
implementation plan refers to "384²" colloquially for the storage
budget; the binding contract is 378. Both shipped model YAMLs override
`image_size: 378` (`configs/model/usam_1_4b.yaml:18-21` and
`configs/model/usam_350m_smoke.yaml:14-15`).

---

## 5. `modality.json` extension (Isaac-GR00T pattern)

Each repo carries a `meta/modality.json` describing the per-channel
state and action dimensions. The schema mirrors the Isaac-GR00T
`modality.json` pattern: a flat dict mapping channel name to
`{dim, dtype, unit, padded_to}`. USAM additionally declares which depth
streams are "low-quality" (Depth-Anything-V2 distilled, no stereo
ground truth) so the geom-loss head can down-weight them.

The exact JSON is produced by Stage 2a from the per-source converter
configuration; readers consult it via `meta/info.json` only when
absolutely necessary — the dataloader prefers the parquet column types.

---

## 6. Per-source quirks

These are the actual quirks implemented in the converters; consult the
file for the binding behavior.

* **DROID** — pulls language from a snapshot of the `KarlP/droid`
  cleaner-annotations dataset and falls back to the RLDS
  `language_instruction` field when the cleaner has no entry. Camera
  map: `exterior_image_1_left -> head_rgb`, `wrist_image_left ->
  wrist_rgb`. 7-D native action layout = `[lin_vel_xyz, ang_vel_xyz,
  grip]`; canonicalization rule = `ee_velocity_passthrough`.
  ↳ `prep/stage_2a_to_lerobot/droid.py:38-149`.

* **AgiBot World 2026** — already ships in LeRobot v2.1 layout with
  USAM-relevant extensions. The `instruction_segments` array is
  **promoted to top-level columns** `level_1`, `level_2`, `level_3` —
  these columns are the **ground truth** for the subtask classifier
  (`L_subtask`); losing them silently would break the conductor's head.
  Camera map: `head -> head_rgb`, `hand_left -> wrist_rgb_left`,
  `hand_right -> wrist_rgb_right`. 24-D bimanual native action;
  canonicalization rule = `joint_delta_to_ee_finite_diff` — the
  converter pre-fills the first 7 padded columns with the right-arm EE
  velocity stream so Stage 3 is a passthrough.
  ↳ `prep/stage_2a_to_lerobot/agibot2026.py:9-54`.

* **RoboMIND** — per-trajectory HDF5 files. **BGR-to-RGB conversion is
  mandatory.** The converter samples a middle frame, asserts via a
  channel-mean heuristic that the array is BGR (blue-mean exceeds
  red-mean by `>=5`), and aborts the chunk if detection is ambiguous
  (`|c0 - c2| < threshold`) rather than risking silent miscoloring.
  Tien Kung `head_cam -> head_rgb`. Simulation embodiments
  (`h5_simulation`) are dropped at enumeration time. 14-D bimanual
  joint-position native action; canonicalization rule =
  `joint_position_to_ee_finite_diff` (converter pre-fills the first 7
  columns from the URDF-driven EE pose stream).
  ↳ `prep/stage_2a_to_lerobot/robomind.py:1-101`.

* **Bridge V2** — RLDS at `gs://gresearch/robotics/bridge`. 5 Hz; the
  7-D native action is already a delta-pose + gripper that doubles as a
  velocity (constant dt). Camera map: `image_0 -> head_rgb`, `image_2
  -> wrist_rgb` if present (`image_1` is auxiliary and skipped).
  Canonicalization rule = `ee_velocity_passthrough`.
  ↳ `prep/stage_2a_to_lerobot/bridge.py:1-39`.

---

## 7. Cache layout

Per-modality fp16 DINO features live under `features/<camera>/<modality>/`
with one safetensors shard per parquet shard, grouped by chunk:

```
features/<camera>/<modality>/chunk-XXX/file-YYY.safetensors
```

Each shard contains one tensor per episode keyed `ep_{episode_index:08d}`
with shape `[T_features, N_tokens, D]` (e.g. `[T, 65, 768]` for ViT-B/14
at 5 Hz with 64 patch tokens kept) and dtype `float16`. A sidecar JSON
(`<shard>.index.json`) lists the per-episode `(T, N, D)` for fast
discovery.

The reader uses `safetensors.safe_open(..., framework="pt",
device="cpu")` — true mmap semantics, kernel page cache shared across
DataLoader workers, no per-worker duplication.

↳ implemented in `usam/dataloader/feature_cache.py:30-219`
(`FeatureCache`, `write_feature_shard`).

---

## 8. Action canonicalization

The 4 supported embodiment keys (registered at
`prep/embodiment.json`) and their canonicalization rule names:

| Embodiment | Native dim | Rule |
|---|---|---|
| `droid_franka` | 7 | `ee_velocity_passthrough` |
| `agibot_g1` | 24 | `joint_delta_to_ee_finite_diff` |
| `robomind_tien_kung` | 14 | `joint_position_to_ee_finite_diff` |
| `bridge_widowx` | 7 | `ee_velocity_passthrough` |

The canonical EE frame is **7-D**:
`[lin_vel_xyz (m/s), ang_vel_xyz (rad/s), gripper (0=open, 1=closed)]`,
with bounds `[-2, 2]`, `[-π, π]`, `[0, 1]` enforced by the validation
gate. The four rule names recognized by `prep.stage_3_canonical` are:
`ee_velocity_passthrough`, `ee_pose_finite_diff`,
`joint_delta_to_ee_finite_diff`, and `joint_position_to_ee_finite_diff`.
Extending USAM to a new embodiment is a single
JSON entry plus (if a new rule name is needed) the corresponding
implementation in `prep/stage_3_canonical.py`.

↳ canonical reference: `prep/embodiment.json` and
`prep/stage_3_canonical.py`.
