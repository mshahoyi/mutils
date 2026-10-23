"""Disk caching for expensive notebook steps: rerunning a cell loads the saved result instead of recomputing it."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

__all__ = ["cached", "fingerprint"]

T = TypeVar("T")


def fingerprint(*parts: Any) -> str:
    """Short stable hash of `parts`: arrays / tensors by their values, DataFrames by their contents, dicts, lists and
    tuples recursively, anything else by its repr."""
    h = hashlib.sha256()
    for part in parts:
        _update(h, part)
    return h.hexdigest()[:16]


def _update(h: Any, obj: Any) -> None:
    if hasattr(obj, "detach"):  # torch tensor (any dtype, any device)
        obj = obj.detach().float().cpu().numpy()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        h.update(obj.to_json(orient="split", default_handler=str).encode())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            _update(h, k)
            _update(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _update(h, x)
        h.update(b"]")
    elif not isinstance(obj, (str, bytes)) and hasattr(obj, "__array__"):  # numpy / mlx arrays
        a = np.asarray(obj)
        if a.dtype == object:
            _update(h, a.tolist())
        else:
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    else:
        h.update(repr(obj).encode())


def cached(path: str | Path, compute: Callable[[], T], key: Any = None, refresh: bool = False) -> T:
    """Return what `compute()` returned when it was last saved to `path` under the same `key`; otherwise run it, pickle
    the result to `path` and return it.

    `key` should capture everything the result depends on (settings, input data, upstream results): it is fingerprinted
    and stored next to the result (`<path>.key`), so changing any of it recomputes. `refresh=True` forces a recompute.
    The result is written to a temporary file first, so an interrupted run never leaves a half-written cache.
    """
    path, fp = Path(path), fingerprint(key)
    key_file = path.with_name(path.name + ".key")
    if not refresh and path.exists() and key_file.exists():
        if key_file.read_text() == fp:
            print(f"[cache] loaded {path.name}")
            with open(path, "rb") as f:
                return pickle.load(f)
        print(f"[cache] {path.name}: inputs changed, recomputing")
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    key_file.write_text(fp)
    return value
