# USAM Docker images

Three images, one per hardware tier. Each image installs the **minimum**
deps for that tier — never the union.

| Tier | Image | Base | Use |
|---|---|---|---|
| **T0** local dev (8xA40) | `Dockerfile.local_a40` | `nvidia/cuda:12.4.1-devel-ubuntu22.04` | Code dev, unit + integration tests, smoke train. |
| **T1** prep (8xA100 Slurm) | `Dockerfile.prep_a100` | `nvidia/cuda:12.4.1-devel-ubuntu22.04` | Phase A pipeline only: download, flow/depth, DINO caching, HF upload. |
| **T2** burst (500xH200) | `Dockerfile.train_h200` | `nvcr.io/nvidia/pytorch:25.01-py3` | Phase B pretrain + fine-tune. Transformer-Engine + FP8. |

## Which image do I want?

- "I'm writing or testing code on my workstation": **`local_a40`**.
- "I'm running prep jobs on Slurm": **`prep_a100`** (then `singularity build` it).
- "I'm submitting the H200 burst": **`train_h200`**.

## Build

```
docker build -f docker/Dockerfile.local_a40 -t usam:local-a40 .
docker build -f docker/Dockerfile.prep_a100 -t usam:prep-a100 .
docker build -f docker/Dockerfile.train_h200 -t usam:train-h200 .
```

## Singularity (T1 only)

```
singularity build usam_prep.sif docker-daemon://usam:prep-a100
```

## Layer ordering

Each `Dockerfile.*` copies `pyproject.toml` + the relevant `requirements/*.txt`
**before** copying the rest of the source tree. Editing a Python source file
does not bust the pip-install layer.

## flash-attn

`flash-attn==2.6.3` is installed via `pip install --no-build-isolation` inside
`local_a40` and `train_h200`. It is intentionally **not** in
`requirements/train.txt` — its build requires torch to be importable and the
package author warns against resolution from a normal requirements file. The
prep image does **not** install flash-attn.

## transformer-engine

Lives in the NGC base of `train_h200` and is therefore pre-installed against
matched CUDA/cuDNN. The A40 image deliberately ships without it (A40 has no
FP8 hardware).
