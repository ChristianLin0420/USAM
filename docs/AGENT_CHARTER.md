# USAM agent team charter

Project-wide rules every USAM subagent must follow. Read this **before** starting any task. Read `docs/IMPLEMENTATION_PLAN.md` second.

## Source of truth

- The canonical design document is `docs/IMPLEMENTATION_PLAN.md`. Always read the section relevant to your scope before writing code. If a task instruction seems to disagree with the plan, the plan wins. If the plan is ambiguous, return `STATUS=blocked` rather than guessing.

## Code style

- Python 3.10. Type-hint everything (`from __future__ import annotations` at the top of every new file).
- Docstrings are NumPy-style on every public class/function.
- One assertion per public function checking input shapes and dtypes.
- Use `torch.Tensor` (not `np.ndarray`) inside model code. Convert only at the IO boundary.
- Prefer `einops.rearrange` over manual reshape/permute.
- Every new file starts with `# SPDX-License-Identifier: MIT` (USAM inherits LDA-1B's MIT license).

## Repository layout (never violate)

- New runtime code lives under `usam/`. Phase A pipeline lives under `prep/`. Tests live under `tests/`. Configs live under `configs/`.
- The original `lda/` is touched as little as possible — only the two files identified in the model-architect agent's prompt.
- `docs/IMPLEMENTATION_PLAN.md` is read-only. If it is wrong, return `STATUS=blocked` with the discrepancy.

## Definition of "done" for any module

1. The file exists with full type-hinted code, docstrings, and one assertion per public function.
2. A unit test in `tests/unit/test_<module_name>.py` exists and passes locally with `pytest`.
3. A 1-paragraph entry in `docs/ARCHITECTURE.md` describes its public API (doc-engineer fills this in Wave 4 — leave a `<!-- TODO: <module> -->` marker if your module isn't covered yet).
4. The module's listing in `docs/IMPLEMENTATION_PLAN.md §14` is checked off (`- [x]`).

## Tools and constraints

- Default to read-before-write. Always `Read` an existing file before `Edit`-ing it.
- Never run training jobs (no `python -m usam.train`). Run only unit tests (`pytest tests/unit/...`) and lightweight integration tests (`pytest tests/integration/...`).
- Never invoke `huggingface-cli login` or `huggingface-cli upload` from inside an agent. Generate the code; the team lead executes uploads.
- For Slurm scripts: write the file, but never `sbatch`. The team lead runs Slurm.
- Never `git push`. The team lead handles all remote operations.

## Scope discipline

- Do not edit files outside your declared scope. If you discover that another agent's file blocks your work, return `STATUS=partial` with `HANDOFF: <agent>` and a description of what they need to change.
- Do not refactor adjacent code "while you're there". A bug fix doesn't need surrounding cleanup.
- Do not add backwards-compatibility shims, feature flags, or unused parameters. The codebase is brand new — change code freely.

## Communication protocol

When you finish a task, return a structured summary:

```
STATUS: success | partial | blocked
FILES_CREATED: <list of paths>
FILES_MODIFIED: <list of paths>
TESTS_ADDED: <list of test files>
TESTS_PASSING: yes | no | n/a
HANDOFF: <next agent name or "team lead">
NOTES: <2-5 lines — surprises, deviations from plan, items the next agent must know>
```

- Hand off explicitly. Do not assume the next agent picks up automatically.
- If you discover a contradiction in `docs/IMPLEMENTATION_PLAN.md`, return `STATUS=blocked` with `NOTES` describing it. Do not silently fix the plan.
- If a unit test you wrote fails, return `STATUS=partial`, leave the failing test in place, and describe the failure in `NOTES`.

## Reviewer rules

The `reviewer` agent runs after every wave. It cannot write code; it produces a structured report. Severity levels: `blocking | major | minor | nit`. Only `blocking` and `major` are mandatory to fix. The team lead routes `blocking`/`major` issues back to the original author for a second pass; the reviewer never edits.
