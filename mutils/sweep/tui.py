"""Live TUI: parse `[gpu]` and `[status]` lines out of `modal app logs --follow`
and render a refreshing rich table.

Stdout from a container looks like:
    [status] cell=abcd1234 status=warming_up label=meta-llama/Meta-Llama-3-8B
    [gpu] cell=abcd1234 util=87 mem_used=23456 mem_total=40960
    ... user prints ...
    [status] cell=abcd1234 status=running
    [gpu] cell=abcd1234 util=92 mem_used=24102 mem_total=40960
    [status] cell=abcd1234 status=done rows=1200

The TUI tracks one Row per cell, updated in place.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

GPU_RE = re.compile(r"\[gpu\] cell=(?P<hash>\w+) util=(?P<util>\d+) mem_used=(?P<used>\d+) mem_total=(?P<total>\d+)")
STATUS_RE = re.compile(r"\[status\] cell=(?P<hash>\w+) status=(?P<status>\w+)(?P<rest>.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class CellRow:
    cell_hash: str
    label: str = "?"
    status: str = "queued"
    util: int | None = None
    mem_used: int | None = None
    mem_total: int | None = None
    rows: int | None = None
    err: str | None = None
    last_update: datetime = field(default_factory=datetime.now)

    def mem_str(self) -> str:
        if self.mem_used is None or self.mem_total is None:
            return "—"
        return f"{self.mem_used / 1024:.1f}/{self.mem_total / 1024:.1f} GB"

    def util_str(self) -> str:
        return "—" if self.util is None else f"{self.util}%"

    def status_text(self) -> Text:
        color = {
            "queued": "dim",
            "skipped": "dim cyan",
            "warming_up": "yellow",
            "running": "cyan",
            "uploading": "blue",
            "done": "green",
            "failed": "bold red",
        }.get(self.status, "white")
        return Text(self.status, style=color)


def _parse_kv(rest: str) -> dict[str, str]:
    return dict(KV_RE.findall(rest))


def _render(rows: dict[str, CellRow], total_cells: int) -> Table:
    table = Table(title="pgmp-sweep", expand=True)
    table.add_column("cell", overflow="fold")
    table.add_column("status", justify="left")
    table.add_column("util%", justify="right")
    table.add_column("mem", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("note", overflow="fold")

    # sort: running first, then warming, then done, then failed/queued/skipped
    order = {
        "running": 0,
        "warming_up": 1,
        "uploading": 2,
        "queued": 3,
        "done": 4,
        "skipped": 5,
        "failed": 6,
    }
    for r in sorted(rows.values(), key=lambda r: (order.get(r.status, 99), r.label)):
        note = r.err or ""
        table.add_row(
            r.label,
            r.status_text(),
            r.util_str(),
            r.mem_str(),
            "—" if r.rows is None else str(r.rows),
            note,
        )

    done = sum(1 for r in rows.values() if r.status == "done")
    skipped = sum(1 for r in rows.values() if r.status == "skipped")
    failed = sum(1 for r in rows.values() if r.status == "failed")
    avg_util = [r.util for r in rows.values() if r.util is not None and r.status in ("running", "uploading")]
    util_str = f"avg util {sum(avg_util) / len(avg_util):.0f}%" if avg_util else "no util data"
    skip_str = f" · {skipped} skipped" if skipped else ""
    table.caption = f"{done}/{total_cells} done · {failed} failed{skip_str} · {util_str}  (Ctrl-C to detach; sweep keeps running)"
    return table


def _all_terminal(rows: dict[str, "CellRow"], total_cells: int) -> bool:
    """True iff at least `total_cells` rows are in a terminal state.

    Counts real rows (done/failed) AND skipped placeholders — a skipped cell
    will never receive log events that turn the placeholder into a real row,
    but it's still terminal.
    """
    terminal = sum(1 for r in rows.values() if r.status in ("done", "failed", "skipped"))
    return terminal >= total_cells


def _auto_pull(
    output: str,
    output_local: str,
    console: Console,
    cell_hashes: list[str] | None = None,
) -> None:
    """Snapshot the sweep's shards from the HF dataset to a local dir.

    If `cell_hashes` is given, restrict the pull to files whose name contains
    one of those 12-char hashes — essential when multiple sweeps share an HF
    repo without a subdir (otherwise unrelated experiments' shards leak in).
    With `cell_hashes=None`, falls back to grabbing every `shard-*.*` in the
    target subdir (legacy / standalone-monitor behavior).

    Tolerates the repo not existing yet (every cell failed before any upload).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    from .storage import parse_output, resolve_hf_token

    target = parse_output(output)
    local_dir = Path(output_local).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    if cell_hashes:
        prefix = f"{target.subdir}/" if target.subdir else ""
        allow_patterns = [f"{prefix}shard-*{h}.*" for h in cell_hashes]
    else:
        allow_patterns = [f"{target.subdir}/shard-*.*" if target.subdir else "shard-*.*"]
    token = resolve_hf_token()
    console.print(f"[cyan]pulling {output} → {local_dir} ({len(allow_patterns)} pattern(s))[/]")
    try:
        snapshot_download(
            repo_id=target.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=token,
        )
    except RepositoryNotFoundError:
        if token is None:
            console.print(
                f"[red]repo {target.repo_id} returned 404, but no HF_TOKEN was found locally. "
                f"If the repo is private this looks identical to a missing repo — "
                f"set HF_TOKEN in env or the project's .env file and re-run `pgmp-sweep pull`.[/]"
            )
        else:
            console.print(f"[yellow]repo {target.repo_id} doesn't exist yet — nothing to pull.[/]")
        return
    console.print("[green]pull complete.[/]")


def monitor(
    app_id: str,
    total_cells: int | None = None,
    expected_labels: list[str] | None = None,
    skipped_labels: list[str] | None = None,
    output: str | None = None,
    output_local: str | None = None,
    cell_hashes: list[str] | None = None,
) -> None:
    """Stream `modal app logs --follow <app-id>`, parse, render live.

    If `output_local` is set (along with `output` and `total_cells`), the TUI
    auto-pulls the shards once every cell reaches a terminal state. Pass
    `cell_hashes` (the hashes of cells in *this* sweep) so the pull is
    restricted to those — important when the HF repo is shared across
    experiments.

    `expected_labels` are runnable cells; `skipped_labels` are cells whose
    shards already exist in the HF dataset and were not relaunched. Skipped
    rows count toward `total_cells` and as terminal for completion detection.
    """
    rows: dict[str, CellRow] = {}
    skipped_set = set(skipped_labels or [])
    # Pre-populate placeholder rows. Skipped ones get status="skipped" so they
    # don't sit at "queued" forever waiting for events that will never come.
    for i, label in enumerate(expected_labels or []):
        placeholder_hash = f"_pending_{i}"
        initial = "skipped" if label in skipped_set else "queued"
        rows[placeholder_hash] = CellRow(cell_hash=placeholder_hash, label=label, status=initial)
    # Also include skipped labels that weren't in expected_labels (defensive)
    expected_set = set(expected_labels or [])
    for j, label in enumerate(skipped_labels or []):
        if label in expected_set:
            continue
        placeholder_hash = f"_skipped_{j}"
        rows[placeholder_hash] = CellRow(cell_hash=placeholder_hash, label=label, status="skipped")

    console = Console()
    cmd = ["modal", "app", "logs", "-f", app_id]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if proc.stdout is None:
        raise RuntimeError(f"failed to spawn `{' '.join(cmd)}`")

    auto_pull_done = False  # latch so we don't re-pull on every refresh

    try:
        with Live(_render(rows, total_cells or 0), console=console, refresh_per_second=2) as live:
            for raw in proc.stdout:
                line = raw.rstrip()
                m = GPU_RE.search(line)
                if m:
                    h = m["hash"]
                    row = rows.setdefault(h, CellRow(cell_hash=h, label=h))
                    row.util = int(m["util"])
                    row.mem_used = int(m["used"])
                    row.mem_total = int(m["total"])
                    row.last_update = datetime.now()
                    if row.status in ("queued", "warming_up"):
                        row.status = "running"
                    live.update(_render(rows, total_cells or len(rows)))
                    continue

                m = STATUS_RE.search(line)
                if m:
                    h = m["hash"]
                    row = rows.setdefault(h, CellRow(cell_hash=h, label=h))
                    kv = _parse_kv(m["rest"])
                    if "label" in kv:
                        row.label = kv["label"]
                        # if we created a placeholder for this label, drop it
                        for ph in [k for k, v in rows.items() if k.startswith("_pending_") and v.label == row.label]:
                            del rows[ph]
                    row.status = m["status"]
                    if "rows" in kv:
                        row.rows = int(kv["rows"])
                    if "err" in kv:
                        row.err = f"{kv['err']}: {kv.get('msg', '')}"
                    row.last_update = datetime.now()
                    live.update(_render(rows, total_cells or len(rows)))

                    # Auto-pull on completion
                    if not auto_pull_done and output and output_local and total_cells and _all_terminal(rows, total_cells):
                        auto_pull_done = True
                        live.stop()
                        _auto_pull(output, output_local, console, cell_hashes=cell_hashes)
                        return
                    continue

                # unstructured line — don't render, but stream to a side log so we don't lose it
                # (skip for v1: keep TUI clean)
    except KeyboardInterrupt:
        console.print("[yellow]detached.[/] sweep continues; re-attach with `pgmp-sweep monitor " + app_id + "`")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 2:
        print("usage: python -m utils.sweep.tui <app-id>")
        sys.exit(1)
    monitor(sys.argv[1])
