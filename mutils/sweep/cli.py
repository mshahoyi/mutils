"""`pgmp-sweep` CLI: run, monitor, pull, logs.

Usage:
    pgmp-sweep run     configs/act_fuzz_random_one.py [--no-detach] [--dry-run]
    pgmp-sweep monitor <app-id>
    pgmp-sweep pull    hf://owner/repo[/subdir] <local-dir>
    pgmp-sweep logs    <app-id>

The `run` command shells out to `modal run [--detach] utils/sweep/runner.py`.
We parse the printed app id from Modal's stdout. Going through the modal CLI
(rather than `app.run()` programmatically) is what keeps `@app.function` at
module scope — Modal rejects function definitions inside other functions.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Module path to the Modal app file. Resolved relative to project root at runtime
# so the CLI works regardless of cwd.
RUNNER_REL_PATH = "utils/sweep/runner.py"

APP_ID_RE = re.compile(r"\b(ap-[A-Za-z0-9]+)\b")


def _project_root() -> Path:
    """Walk up from this file to the project root (the one with pyproject.toml)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find project root (no pyproject.toml in any parent)")


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a config file defining `sweep`."),
    detach: bool = typer.Option(True, "--detach/--no-detach", help="Detach from the Modal app after spawning (default)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the cell plan without launching."),
    monitor_after: bool = typer.Option(True, "--monitor/--no-monitor", help="Open the live TUI after launch."),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run all cells even if their shards already exist in the HF dataset."),
):
    """Launch a sweep from a config file."""
    from .runner import _import_sweep
    from .storage import existing_shards, parse_output, resolve_hf_token
    from .tui import monitor as monitor_cmd

    sweep = _import_sweep(config.resolve())
    cells = sweep.cells()

    # Compute which cells are already done so the TUI can mark them "skipped"
    # rather than leaving them at "queued" forever. With --force, treat
    # nothing as done.
    target = parse_output(sweep.output)
    done_hashes = set() if force else existing_shards(target, token=resolve_hf_token())
    runnable = [c for c in cells if c.hash not in done_hashes]
    skipped = [c for c in cells if c.hash in done_hashes]
    cell_hash_list = [c.hash for c in cells]  # used to scope auto-pull below

    if dry_run:
        typer.echo(f"[sweep] {len(cells)} cells: {len(runnable)} to run, {len(skipped)} already done" + (" (force=on)" if force else ""))
        for c in cells:
            tag = "skipped" if c.hash in done_hashes else "queued "
            typer.echo(f"   • [{tag}] {c.label}  ({c.hash})")
        return

    # Everything already done → no Modal launch needed. Refresh the local copy
    # and exit. Pull is filtered to THIS sweep's cell hashes so we don't drag
    # in shards from unrelated sweeps that share the same HF repo.
    if not runnable:
        typer.echo(f"[sweep] all {len(cells)} cells already in {sweep.output} — skipping launch.")
        if sweep.output_local:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import RepositoryNotFoundError

            local_dir = Path(sweep.output_local).expanduser().resolve()
            local_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{target.subdir}/" if target.subdir else ""
            allow_patterns = [f"{prefix}shard-*{h}.*" for h in cell_hash_list]
            typer.echo(f"[sweep] pulling {sweep.output} → {local_dir} ({len(allow_patterns)} cell(s))")
            try:
                snapshot_download(
                    repo_id=target.repo_id,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                    token=resolve_hf_token(),
                )
                typer.echo("[sweep] pull complete.")
            except RepositoryNotFoundError:
                typer.echo(f"[sweep] repo {target.repo_id} doesn't exist yet — nothing to pull.")
        return

    project_root = _project_root()
    runner_path = project_root / RUNNER_REL_PATH
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd += [str(runner_path), "--config", str(config.resolve())]
    if force:
        cmd.append("--force")
    # NB: cell timeout comes from `sweep.timeout` in the config, read by
    # runner.py at module load via argv-pre-parse of `--config`. Don't add
    # `--timeout` here — the sweep config is the source of truth.

    typer.echo(f"[sweep] {' '.join(cmd)}")

    # Stream modal's output to the terminal AND tee it so we can grep for the app id.
    proc = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if proc.stdout is None:
        raise SystemExit("failed to spawn modal subprocess")

    app_id: str | None = None
    for raw in proc.stdout:
        sys.stdout.write(raw)
        sys.stdout.flush()
        if app_id is None:
            m = APP_ID_RE.search(raw)
            if m:
                app_id = m.group(1)
        # In detach mode modal prints the spawn confirmation then exits — we can stop reading
        # once we've got both the app id and the [sweep] spawned line.
        if detach and app_id and "spawned" in raw:
            break

    if not detach:
        proc.wait()
        return

    # Detach mode: leave modal subprocess running just long enough to flush, then move on
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    if app_id is None:
        typer.echo("\n[sweep] could not find app id in modal output; cannot launch monitor.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n[sweep] app_id={app_id}")
    if monitor_after:
        pull_msg = f" auto-pulling to {sweep.output_local} on completion." if sweep.output_local else ""
        typer.echo(f"[sweep] opening live TUI. Ctrl-C detaches; sweep keeps running.{pull_msg}\n")
        monitor_cmd(
            app_id,
            total_cells=len(cells),
            expected_labels=[c.label for c in cells],
            skipped_labels=[c.label for c in skipped],
            output=sweep.output,
            output_local=sweep.output_local,
            cell_hashes=cell_hash_list,  # so auto-pull only fetches THIS sweep's shards
        )


@app.command()
def monitor(
    app_id: str = typer.Argument(..., help="Modal app id (printed by `run`)."),
    output: str | None = typer.Option(
        None, "--output", help="hf URI like hf://owner/repo or hf://owner/repo/subdir; required for --pull-to."
    ),
    pull_to: Path | None = typer.Option(None, "--pull-to", help="Local dir; pull shards here once all cells are terminal."),
    total_cells: int | None = typer.Option(None, "--total", help="Cell count (used to detect completion when re-attaching)."),
):
    """Re-attach to an existing sweep's live TUI. Optional auto-pull on completion."""
    from .tui import monitor as monitor_cmd

    if pull_to and not (output and total_cells):
        raise typer.BadParameter("--pull-to requires --output and --total")
    monitor_cmd(
        app_id,
        total_cells=total_cells,
        output=output,
        output_local=str(pull_to) if pull_to else None,
    )


@app.command()
def logs(app_id: str = typer.Argument(...)):
    """Raw `modal app logs` passthrough."""
    os.execvp("modal", ["modal", "app", "logs", app_id])


@app.command()
def pull(
    output: str = typer.Argument(..., help="hf URI like hf://owner/repo or hf://owner/repo/subdir"),
    local_dir: Path = typer.Argument(..., file_okay=False),
):
    """Download all shards (any extension) from an HF dataset to a local directory."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    from .storage import parse_output, resolve_hf_token

    target = parse_output(output)
    local_dir.mkdir(parents=True, exist_ok=True)
    allow_pat = f"{target.subdir}/shard-*.*" if target.subdir else "shard-*.*"
    typer.echo(f"[sweep] pulling {output} → {local_dir}")
    try:
        snapshot_download(
            repo_id=target.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=[allow_pat],
            token=resolve_hf_token(),
        )
    except RepositoryNotFoundError:
        typer.echo(f"[sweep] repo {target.repo_id} doesn't exist yet — nothing to pull.")
        return
    typer.echo("[sweep] done.")


def main() -> None:  # pragma: no cover
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
