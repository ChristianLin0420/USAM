# SPDX-License-Identifier: MIT
"""Per-source raw downloaders for the USAM Phase A pipeline.

Each module under this package exposes a :func:`download` function and a
``__main__`` CLI that, given the YAML config for one source, fetches the
raw dataset to a local cache. Heavy backends (``huggingface_hub``,
``tensorflow_datasets``) are imported lazily inside ``download`` so simply
importing the module never fails.

All downloaders accept ``--dry-run`` and refuse to perform any network IO
in that mode; this is the path exercised by unit/integration tests.
"""
from __future__ import annotations
