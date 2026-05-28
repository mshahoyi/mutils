"""Sweep orchestrator: fan a vanilla python function over a parameter grid onto
Modal GPUs, write per-cell shards (format determined by an optional user-supplied
saver — defaults to parquet for `list[dict]` returns, pickle otherwise) to a
HuggingFace dataset, stream GPU utilisation back to a local TUI.

Public API:
    from mutils.sweep import Sweep, Cell
    from mutils.sweep import default_saver  # if you want to wrap it in your own
    from mutils.sweep import find_shards, load_shards, load_pickles

Usage lives in `configs/*.py`; the CLI is in `mutils.sweep.cli`.
"""

from .storage import default_saver, find_shards, load_pickles, load_shards
from .types import Cell, Sweep, SweepEnv

__all__ = [
    "Cell",
    "Sweep",
    "SweepEnv",
    "default_saver",
    "find_shards",
    "load_pickles",
    "load_shards",
]
