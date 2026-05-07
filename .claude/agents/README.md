# USAM agent team

Eleven specialist subagents implement USAM under team-lead orchestration. Every agent reads `docs/AGENT_CHARTER.md` and the relevant section of `docs/IMPLEMENTATION_PLAN.md` before starting.

## Roster

| File | Agent | Owns | Wave |
|---|---|---|---|
| `10_model_architect.md` | `model-architect` | `usam/encoders/`, `usam/adapters/`, two surgical edits inside `lda/model/modules/action_model/flow_matching_head/...`, model YAMLs, Phase A.5 adapter pretrain | 1 |
| `20_conductor_engineer.md` | `conductor-engineer` | `usam/conductor/`, `usam/inference/realtime.py`, `usam/inference/openloop.py` (stub) | 2 |
| `30_losses_engineer.md` | `losses-engineer` | `usam/aux_heads/`, `usam/losses.py` | 2 |
| `40_data_engineer.md` | `data-engineer` | `usam/dataloader/`, `prep/stage_2a_to_lerobot/*`, `prep/stage_2b/2c/3/4`, `configs/data/`, fixture generator | 1 (P1) + 2 (P2) |
| `50_pipeline_engineer.md` | `pipeline-engineer` | `prep/_base.py`, `prep/_hub.py`, `prep/dispatch.py`, `prep/stage_0/1/5/6`, `slurm/*`, `tests/integration/test_pipeline_end_to_end.py` | 1 (P1) + 4 (P2) |
| `60_training_engineer.md` | `training-engineer` | `usam/train.py`, `configs/train/*.yaml`, `tests/integration/test_smoke_train.py` | 3 |
| `70_inference_engineer.md` | `inference-engineer` | `usam/inference/openloop.py` (full), `configs/eval/`, `scripts/eval_libero.sh` | 4 |
| `80_infra_engineer.md` | `infra-engineer` | `pyproject.toml`, `requirements/*.txt`, `docker/*`, `README.md`, `.github/workflows/ci.yaml` | 1 |
| `90_test_engineer.md` | `test-engineer` | `tests/golden_data/`, `tests/conftest.py`, `pytest.ini`, integration test maintenance | 4 |
| `95_doc_engineer.md` | `doc-engineer` | `docs/ARCHITECTURE.md`, `docs/DATA_FORMAT.md`, `docs/HOWTO_*.md`, top-level `README.md` (with infra-engineer) | 4 |
| `99_reviewer.md` | `reviewer` | Read-only code review after every wave | after each wave |

## Dependency graph

```
                          infra-engineer ────────────┐
                                                     │
   model-architect ──────┐                           │
                         ▼                           │
                  conductor-engineer                 │
                         │                           │
                         ▼                           ▼
                   losses-engineer  ◄── pipeline-engineer (P1)
                         │                           │
                         └──────┬────────────────────┘
                                ▼
                       training-engineer
                                │
                  ┌─────────────┼─────────────────┐
                  ▼             ▼                 ▼
        inference-engineer   test-engineer   pipeline-engineer (P2)
                                                  │
                                          doc-engineer (Wave 4)
```

## Communication protocol

Every agent returns:

```
STATUS: success | partial | blocked
FILES_CREATED: <list>
FILES_MODIFIED: <list>
TESTS_ADDED: <list>
TESTS_PASSING: yes | no | n/a
HANDOFF: <next agent or "team lead">
NOTES: <2-5 lines>
```

The team lead routes handoffs and re-tasks on `blocking`/`major` reviewer findings. See `docs/AGENT_CHARTER.md` for the full rules.

## Wave plan

| Wave | Agents (parallel within a wave) | Output |
|---|---|---|
| 1 | model-architect, data-engineer (P1), pipeline-engineer (P1), infra-engineer | encoder, runtime dataloader, prep base, infra |
| 2 | conductor-engineer, losses-engineer, data-engineer (P2) | slow/fast split, aux losses, all 6 source converters |
| 3 | training-engineer | train loop + smoke test |
| 4 | inference-engineer, test-engineer, doc-engineer, pipeline-engineer (P2) | eval, fixtures, docs, dispatcher |

After every wave: reviewer pass + commit.
