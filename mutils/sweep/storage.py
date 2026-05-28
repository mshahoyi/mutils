"""Per-cell shards in a HuggingFace dataset repo, plus local-folder loaders.

Output spec is `hf://owner/repo` or `hf://owner/repo/subdir`. Each cell writes
one shard whose filename embeds every grid param plus a 12-char hash. Format
depends on the function's return value:

* `list[dict]`   → `<basename>.parquet` (one row per element, with `cell_hash`
                   and the cell params injected as columns)
* anything else  → `<basename>.pkl` (pickle of `{"value": ..., "cell_hash":
                   ..., "params": ...}`)

To resume, list the repo, find shards by trailing 12-char hash, run only
missing cells. To LOAD locally after pulling, use `find_shards(folder, **kw)`
or `load_shards(folder, **kw)` to filter by any subset of grid params.
"""

from __future__ import annotations

import os
import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .types import Cell, Sweep, _slug

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_HF_URI_RE = re.compile(r"^hf://(?P<owner>[^/]+)/(?P<repo>[^/]+)(?:/(?P<subdir>.+?))?/?$")

# Shard files start with `shard-`, contain a 12-char hex hash as the last
# `-`-delimited component before any extension/suffix, and may use any
# extension (parquet/pkl/json/npy/csv/...). The regex tolerates extra suffix
# segments after the hash so multi-file savers work.
_SHARD_HASH_RE = re.compile(r"^shard-(?:.+-)?([0-9a-f]{12})(?:\..+|-.+)?$")


@dataclass(frozen=True)
class HFTarget:
    """Parsed HF dataset destination."""

    owner: str
    repo: str
    subdir: str = ""

    @property
    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def path_in_repo(self, shard_name: str) -> str:
        return f"{self.subdir}/{shard_name}" if self.subdir else shard_name


def parse_output(spec: str) -> HFTarget:
    m = _HF_URI_RE.match(spec)
    if not m:
        raise ValueError(f"Output must be hf://owner/repo[/subdir], got {spec!r}")
    return HFTarget(owner=m["owner"], repo=m["repo"], subdir=m["subdir"] or "")


def resolve_hf_token() -> str | None:
    """HF_TOKEN from env, falling back to the project-root `.env`.

    The local CLI doesn't auto-source .env on every shell startup (per user
    convention), so the explicit fallback here matters when running pulls /
    pre-launch repo lookups from a fresh terminal.

    Accepts both dotenv-style `HF_TOKEN=...` and shell-style `export HF_TOKEN=...`
    lines — the .env in this project is sourced by the user's shell init, so
    it uses `export`, but other tooling writes plain assignment.
    """
    if tok := os.environ.get("HF_TOKEN"):
        return tok
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if line.startswith("HF_TOKEN="):
            v = line.split("=", 1)[1].strip()
            if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
                v = v[1:-1]
            return v or None
    return None


def existing_shards(target: HFTarget, token: str | None = None) -> set[str]:
    """Return the set of shard hashes already present in the dataset repo.

    Returns empty set if the repo doesn't exist yet.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(target.repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()

    prefix = f"{target.subdir}/" if target.subdir else ""
    out = set()
    for f in files:
        if not f.startswith(prefix):
            continue
        name = f[len(prefix) :]
        # Anchored on the trailing 12-char hex hash before .parquet/.pkl/.pickle —
        # tolerates both old `shard-<hash>` and new `shard-k1=v1-...-<hash>` formats.
        m = _SHARD_HASH_RE.match(name)
        if m:
            out.add(m.group(1))
    return out


def missing_cells(sweep: Sweep, token: str | None = None) -> list[Cell]:
    """Return cells whose shards are NOT yet in the HF dataset."""
    target = parse_output(sweep.output)
    done = existing_shards(target, token=token)
    return [c for c in sweep.cells() if c.hash not in done]


def default_saver(return_value: Any, cell: Cell, dest_dir: Path) -> list[Path]:
    """Built-in saver. Dispatches on return type:

    * `list[dict]`  → `<basename>.parquet`. `cell_hash` and the cell params
                      are injected as columns so downstream code can read all
                      shards together and group by them.
    * `dict`        → `<basename>.pkl`, payload
                      `{"value": ..., "cell_hash": ..., "params": ...}`.
    * other types   → not supported. Pass a custom `saver=` to your `Sweep`
                      and serialize however you want.

    Empty `list[dict]` and `None` are rejected (likely silent failures).
    Returns the list of paths actually written (always length 1 for the
    default saver).
    """
    if return_value is None:
        raise ValueError(f"Cell {cell.label} returned None; refusing to save empty shard")

    is_rows = isinstance(return_value, list) and all(isinstance(r, dict) for r in return_value)
    if is_rows and not return_value:
        raise ValueError(f"Cell {cell.label} returned an empty list; refusing to save empty shard")

    if is_rows:
        print("saving file into parquet")
        df = pd.DataFrame(return_value)
        df["cell_hash"] = cell.hash
        for k, v in cell.params.items():
            if k not in df.columns:
                df[k] = v
        path = dest_dir / f"{cell.shard_basename}.parquet"
        df.to_parquet(path, index=False)
    elif isinstance(return_value, dict):
        print("saving file into pickle")
        path = dest_dir / f"{cell.shard_basename}.pkl"
        with path.open("wb") as f:
            pickle.dump({"value": return_value, "cell_hash": cell.hash, "params": cell.params}, f)
    else:
        raise TypeError(
            f"default_saver only handles list[dict] (→ parquet) and dict (→ pickle); "
            f"got {type(return_value).__name__} for cell {cell.label}. "
            f"Pass a custom `saver=` to your Sweep to serialize this type."
        )
    return [path]


def upload_shard(
    target: HFTarget,
    cell: Cell,
    return_value: Any,
    saver: Callable[[Any, Cell, Path], list[Path]] | None = None,
    token: str | None = None,
    create_repo: bool = True,
) -> list[str]:
    """Serialize via `saver` (defaults to `default_saver`) and upload every
    file the saver wrote to the HF dataset target.

    The saver receives a per-cell tmp dir, writes whatever files it wants
    there, and returns the paths it wrote. Each file's basename becomes the
    path-in-repo (under the target's subdir if any).

    Returns the list of paths-in-repo of the uploaded shards.
    """
    import tempfile

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    saver = saver or default_saver

    api = HfApi(token=token)
    if create_repo:
        try:
            api.repo_info(target.repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            api.create_repo(target.repo_id, repo_type="dataset", private=True, exist_ok=True)

    uploaded: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"sweep-{cell.hash}-") as tmp:
        tmp_path = Path(tmp)
        written = saver(return_value, cell, tmp_path)
        if not written:
            raise ValueError(f"Cell {cell.label}: saver wrote no files")
        for p in written:
            p = Path(p)
            if not p.is_file():
                raise FileNotFoundError(f"Cell {cell.label}: saver claimed {p} but it doesn't exist")
            path_in_repo = target.path_in_repo(p.name)
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=target.repo_id,
                repo_type="dataset",
                commit_message=f"sweep: {cell.label} ({p.name})",
            )
            uploaded.append(path_in_repo)
    return uploaded


# ---------------------------------------------------------------------------
# Local-folder loaders — for downstream code that has already pulled the
# shards (via `pgmp-sweep pull` or Sweep.output_local auto-pull) and wants to
# filter by grid params without re-parsing filenames by hand.
# ---------------------------------------------------------------------------


def find_shards(folder: str | Path, **filters: Any) -> list[Path]:
    """Return shard file paths in `folder` whose filename encodes every given
    `key=value` filter. Matching is substring-based on `f"{key}={_slug(value)}"`,
    using the same slug rules that produced the filenames.

    Examples:
        find_shards("data/sweeps/foo")                          # all shards
        find_shards("data/sweeps/foo", model_id="Qwen/Qwen3-4B")
        find_shards("data/sweeps/foo", model_id="...", n_iterations=100)

    Returns sorted list of paths (.parquet and .pkl); empty if nothing matches.
    Pass zero filters to get every shard in the folder.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"find_shards: folder does not exist: {folder}. (Empty matches return [] — only the folder itself missing raises.)"
        )
    tags = [f"{k}={_slug(v)}" for k, v in filters.items()]
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        # Canonical shard form: ends with `-<12hex>.<ext>`. Rejects malformed
        # `shard-foo.parquet` files that happen to share the prefix.
        if not _SHARD_HASH_RE.match(p.name):
            continue
        if all(tag in p.name for tag in tags):
            out.append(p)
    return out


def load_shards(folder: str | Path, **filters: Any) -> pd.DataFrame:
    """Concat every matching parquet shard into one DataFrame.

    Pickle shards are skipped (not row-shaped); use `find_shards` + manual
    `pickle.load` to get those. Empty DataFrame if nothing matches.
    """
    paths = [p for p in find_shards(folder, **filters) if p.suffix == ".parquet"]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_pickles(folder: str | Path, **filters: Any) -> list[dict]:
    """For sweeps whose function returned non-row objects.

    Returns one dict per matching pickle shard, each shaped like
    `{"value": ..., "cell_hash": ..., "params": ...}` — same payload that
    `upload_shard` wrote.
    """
    paths = [p for p in find_shards(folder, **filters) if p.suffix in (".pkl", ".pickle")]
    out = []
    for p in paths:
        with p.open("rb") as f:
            out.append(pickle.load(f))
    return out
