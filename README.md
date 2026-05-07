# USAM — Unified Spatial Action Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

USAM extends [LDA-1B](https://arxiv.org/abs/2602.12215)'s latent-dynamics
paradigm by aligning RGB, depth, and optical flow in a shared DINOv3 space
with cross-modal consistency losses, and decouples slow language understanding
from fast control via a cosine-drift-triggered Plan-KV-Cache, enabling >=3x
faster real-time WAM inference at no quality cost.

USAM is a thin overlay on LDA-1B (~2,000 LoC delta). Everything new lives
under `usam/` and `prep/`; `lda/` is touched only minimally.

---

## How do I...

| Task | Where to look |
|---|---|
| Read the canonical design | [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) |
| Understand each module's API | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Understand the data layout on the Hub | [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) |
| Set up an 8xA40 dev box (T0) | [`docs/HOWTO_LOCAL_8A40.md`](docs/HOWTO_LOCAL_8A40.md) |
| Run Phase A prep on Slurm A100 (T1) | [`docs/HOWTO_SLURM_A100.md`](docs/HOWTO_SLURM_A100.md) |
| Submit the H200 burst (T2) | [`docs/HOWTO_H200.md`](docs/HOWTO_H200.md) |
| Pick the right Docker image | [`docker/README.md`](docker/README.md) |
| Read the agent charter | [`docs/AGENT_CHARTER.md`](docs/AGENT_CHARTER.md) |

## Quick install (dev)

```bash
git clone <this repo> USAM && cd USAM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# flash-attn requires --no-build-isolation; only install on a CUDA box:
pip install --no-build-isolation flash-attn==2.6.3
```

To build the matching Docker image instead, see [`docker/README.md`](docker/README.md).

## Repository layout (high-level)

```
USAM/
├── usam/        new runtime code (encoders, conductor, aux heads, dataloader, train, inference)
├── prep/        Phase A pipeline (T1 — Slurm preprocessing)
├── lda/         LDA-1B's original code, lightly modified (+5 LoC, +10 LoC)
├── configs/     YAML configs for data / model / train / eval
├── tests/       unit + integration + golden_data
├── slurm/       universal preemptible job template
├── docker/      three Dockerfiles, one per hardware tier
└── docs/        canonical plan, HOWTOs, architecture refs
```

For the full directory tree and per-file ownership, see
[`docs/IMPLEMENTATION_PLAN.md` Section 3](docs/IMPLEMENTATION_PLAN.md).

## Citation

USAM builds on LDA-1B; if USAM is useful in your work, please also cite the
original paper:

```bibtex
@article{lyu2026lda,
  title={LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion},
  author={Lyu, Jiangran and Liu, Kai and Zhang, Xuheng and Liao, Haoran and Feng, Yusen and Zhu, Wenxuan and Shen, Tingrui and Chen, Jiayi and Zhang, Jiazhao and Dong, Yifei and others},
  journal={arXiv preprint arXiv:2602.12215},
  year={2026}
}
```

## License

MIT (inherited from LDA-1B). See [`LICENSE`](LICENSE).
