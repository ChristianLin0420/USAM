# SPDX-License-Identifier: MIT
"""Unit tests for the LIBERO finetune CLI entry point.

Covers three things:

1. ``--help`` output advertises every documented CLI flag.
2. :func:`usam.finetune_libero.load_model_config` parses the smoke YAML
   and recognises the ``encoder`` section.
3. End-to-end dry-run: with the LIBERO dataloader and model construction
   patched out, :func:`usam.finetune_libero.run_finetune` runs three
   stub steps and writes ``finetune_ckpt.pt`` to the output dir.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch import nn

PY = sys.executable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_help(module: str) -> str:
    """Return stdout of ``python -m <module> --help``.

    Identical idiom to :func:`tests.unit.test_prep_cli._run_help`.
    """
    out = subprocess.run(
        [PY, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert out.returncode == 0, (
        f"`python -m {module} --help` exited {out.returncode}\n"
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    )
    return out.stdout


# ---------------------------------------------------------------------------
# (a) argparse surface
# ---------------------------------------------------------------------------
def test_argparse_has_required_flags() -> None:
    """Every documented flag must be present in ``--help``."""
    text = _run_help("usam.finetune_libero")
    expected = [
        "--base-ckpt",
        "--libero-data",
        "--output-dir",
        "--max-steps",
        "--model-config",
        "--suite",
        "--learning-rate",
        "--batch-size",
        "--eval-every",
        "--log-every",
    ]
    missing = [flag for flag in expected if flag not in text]
    assert not missing, (
        f"`python -m usam.finetune_libero --help` is missing flags: {missing}\n"
        f"full help text:\n{text}"
    )


# ---------------------------------------------------------------------------
# (b) config loading
# ---------------------------------------------------------------------------
def test_config_loading() -> None:
    """``load_model_config`` accepts the smoke YAML and surfaces encoder/player/action_head."""
    from usam.finetune_libero import load_model_config

    smoke_path = REPO_ROOT / "configs" / "model" / "usam_350m_smoke.yaml"
    assert smoke_path.exists(), f"smoke YAML missing at {smoke_path}"
    cfg = load_model_config(smoke_path)

    assert isinstance(cfg, dict)
    assert "encoder" in cfg, "encoder section missing from parsed config"
    assert "player" in cfg
    assert "action_head" in cfg
    # Spot-check a known field for confidence.
    assert cfg["encoder"].get("type") == "tri_dino"


def test_config_loading_rejects_missing_section(tmp_path: Path) -> None:
    """A YAML missing one of the required sections must raise ``ValueError``."""
    from usam.finetune_libero import load_model_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("encoder:\n  type: tri_dino\n")  # no player / action_head
    with pytest.raises(ValueError, match="missing required section"):
        load_model_config(bad)


# ---------------------------------------------------------------------------
# (c) end-to-end dry-run
# ---------------------------------------------------------------------------
class _TinyStubModel(nn.Module):
    """One-linear-layer stub so ``state_dict`` / ``to(device)`` work cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)


class _TinyLiberoDataset(torch.utils.data.Dataset):
    """Yields three batch-of-1 synthetic samples — enough for max-steps=3."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rgb": torch.zeros(3, 8, 8),
            "proprio": torch.zeros(9),
            "action": torch.zeros(7),
        }


def test_main_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end run with patched dataloader + tiny stub model.

    Asserts the checkpoint file is written to ``<tmp_path>/out/finetune_ckpt.pt``.
    """
    from usam import finetune_libero

    # --- patch model construction to a tiny stub ---
    monkeypatch.setattr(
        finetune_libero,
        "build_model_from_config",
        lambda cfg: _TinyStubModel(),
    )

    # --- patch the LIBERO dataloader to a tiny synthetic one ---
    def _fake_loader(args: Any) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            _TinyLiberoDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    monkeypatch.setattr(finetune_libero, "build_libero_dataloader", _fake_loader)

    # --- ensure wandb stays a no-op even if the dev env has it set ---
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    # --- create dummy base ckpt (empty state_dict so load_state_dict succeeds) ---
    base_ckpt = tmp_path / "dummy.pt"
    torch.save({"state_dict": {}}, str(base_ckpt))

    libero_dir = tmp_path / "libero"
    libero_dir.mkdir()

    out_dir = tmp_path / "out"

    argv = [
        "--base-ckpt", str(base_ckpt),
        "--libero-data", str(libero_dir),
        "--output-dir", str(out_dir),
        "--max-steps", "3",
        "--model-config", str(REPO_ROOT / "configs" / "model" / "usam_350m_smoke.yaml"),
        "--device", "cpu",
        "--log-every", "1",
        "--batch-size", "1",
    ]
    exit_code = finetune_libero.main(argv)
    assert exit_code == 0

    ckpt_path = out_dir / "finetune_ckpt.pt"
    assert ckpt_path.exists(), f"expected checkpoint at {ckpt_path}, not found"

    # Verify the saved payload shape.
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "model" in payload
    assert "step" in payload
    assert "config" in payload
    assert payload["suite"] == "libero_10"
