"""Sweep + Cell dataclasses.

A `Sweep` declares: a function, a parameter grid, a GPU spec, an HF dataset to
write shards to, and a concurrency limit. The orchestrator turns that into N
`Cell` instances (one per grid point), each with a deterministic content hash
used as the shard filename. Re-running the same Sweep against the same dataset
will skip cells whose shards already exist.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: object) -> str:
    """Filesystem-safe slug. Keeps letters/digits/`._-`; collapses everything else to `-`."""
    s = str(value).replace("/", "__")
    s = _SLUG_RE.sub("-", s)
    return s.strip("-_") or "x"


@dataclass(frozen=True)
class Cell:
    """One point in the parameter grid."""

    fn_path: str  # "experiments.act_fuzz_random:run"
    params: dict[str, Any]

    @property
    def hash(self) -> str:
        """12-char content hash of (fn_path, params). Stable across processes."""
        payload = json.dumps({"fn": self.fn_path, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def shard_basename(self) -> str:
        """Filename without extension. Encodes EVERY grid param as
        `key=slug(value)` (sorted by key for determinism) plus the 12-char hash
        as a final disambiguator. Long but transparent — `find_shards()` can
        filter by any subset of params via substring match.
        """
        parts = [f"{k}={_slug(v)}" for k, v in sorted(self.params.items())]
        if parts:
            return f"shard-{'-'.join(parts)}-{self.hash}"
        return f"shard-{self.hash}"

    @property
    def shard_name(self) -> str:
        """Default parquet name. The actual extension depends on the return
        type at upload time (`.parquet` for `list[dict]`, `.pkl` otherwise) —
        callers that need to address either should use `shard_basename`.
        """
        return f"{self.shard_basename}.parquet"

    @property
    def label(self) -> str:
        """Short human-readable label for the TUI/logs. Prefers `model_id` if present."""
        if "model_id" in self.params:
            return str(self.params["model_id"])
        return ", ".join(f"{k}={v}" for k, v in self.params.items())


@dataclass
class Sweep:
    """Declarative sweep specification.

    Args:
        fn: The function to run per cell. Vanilla python callable taking grid
            params as kwargs; can return anything serialisable. Referenced by
            import path (`module:function`) so the Modal worker re-imports it.
        grid: Parameter name → list of values. Cartesian product gives the cells.
        output: HF dataset spec like `hf://owner/repo` or `hf://owner/repo/subdir`.
            Each cell writes its shard files there (one or more, decided by saver).
        saver: Optional `(return_value, cell, dest_dir) -> list[Path]` callable.
            Writes whatever files it wants into `dest_dir`; the runner uploads
            exactly those files to `output`. Filenames should contain
            `cell.shard_basename` (or at least the 12-char hash) so resume
            detection finds them. If None, uses the built-in `default_saver`
            which dispatches list[dict]→parquet, anything else→pickle.
        output_local: If set, the TUI auto-pulls all shards to this local
            directory once every cell reaches a terminal state.
        gpu: Modal GPU spec: "T4", "A10G", "A100-40GB", "H100", optionally with
            ":N" for multi-GPU.
        concurrency: Max parallel Modal containers. Defaults to len(grid).
        timeout: Per-cell timeout in seconds.
        env: Extra env vars to set in the container (in addition to .env).
    """

    fn: Callable[..., Any]
    grid: dict[str, list[Any]]
    output: str
    saver: Callable[..., list] | None = None
    output_local: str | None = None
    gpu: str = "A10G"
    concurrency: int | None = None
    timeout: int = 60 * 60  # 1h per cell
    env: dict[str, str] = field(default_factory=dict)

    @property
    def fn_path(self) -> str:
        """`module:function` import path for the worker side."""
        return f"{self.fn.__module__}:{self.fn.__name__}"

    @property
    def saver_path(self) -> str | None:
        """`module:function` for the saver, or None to use the default saver
        on the worker side."""
        if self.saver is None:
            return None
        return f"{self.saver.__module__}:{self.saver.__name__}"

    def cells(self) -> list[Cell]:
        """Cartesian product of `grid` → list of Cell instances."""
        keys = list(self.grid.keys())
        out = []
        for combo in itertools.product(*[self.grid[k] for k in keys]):
            params = dict(zip(keys, combo, strict=True))
            out.append(Cell(fn_path=self.fn_path, params=params))
        return out
