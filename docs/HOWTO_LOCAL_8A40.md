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
