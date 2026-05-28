"""Library entry points for the consumer's Modal sweep runner.

The consumer's project owns a thin `sweep_runner.py` (scaffold via
`mutils-sweep init`) that:
  1. Declares `ENV: SweepEnv = SweepEnv(app_name=..., ...)` as a literal.
  2. Builds image/secrets/volume via `build_image / build_secrets / build_volume`.
  3. Defines one `@app.function(gpu="...")` worker per GPU type, each
     delegating its body to `run_cell_body`.
  4. Defines `run_shepherd` delegating to `shepherd_body`.
  5. Defines `@app.local_entrypoint() main(...)` delegating to `dispatch_sweep`.

Why the consumer owns the file: Modal requires `@app.function` at module
scope AND requires that local and container module-load produce identical
Modal Image/App/Secret/Volume object ids. The only reliable way to feed
project-specific values into both sides identically is a Python literal
inside a file that Modal exec's verbatim on both ends — namely the runner
itself. argv-conditional values do NOT propagate to container module-load
(documented gotcha — Modal binds decorator metadata at deploy time from
the local side, so `resolve_timeout_from_argv` is the lone safe exception).

`mutils-sweep init` writes the template; the consumer fills in ENV and
which GPUs they want. See `_template_sweep_runner.py`.
"""

from __future__ import annotations

import importlib
import importlib.util as iu
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import modal

    from .types import SweepEnv

DEFAULT_TIMEOUT_S = 60 * 60  # 1h per cell — fallback when no --timeout was passed
# Modal's max function timeout is 24h. The shepherd is CPU-only and just waits
# for all per-cell containers, so we always give it the ceiling — it has to
# outlive its slowest cell, plus any queue time when n_cells > concurrency.
SHEPHERD_TIMEOUT_S = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Module-load helpers — called once, locally, before @app.function decorators
# in the consumer's runner evaluate. On container re-import argv has no
# --config and the timeout falls back to default; harmless because Modal uses
# the locally-registered decorator timeout at runtime.
# ---------------------------------------------------------------------------


def _argv_get(flag: str) -> str | None:
    """Pre-parse a flag value from sys.argv at module load."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def resolve_timeout_from_argv(default: int = DEFAULT_TIMEOUT_S) -> int:
    """Read `sweep.timeout` from the --config file in argv at module load.

    Local invocation has --config; we import the user's config and return
    `sweep.timeout` for the @app.function decorator. Container re-import
    has no --config and falls back to `default`; harmless because Modal
    uses the locally-registered timeout at runtime.
    """
    config_path = _argv_get("--config")
    if not config_path:
        return default
    sweep = _import_sweep(Path(config_path).resolve())
    timeout = int(sweep.timeout)
    print(
        f"[runner] resolved sweep.timeout from {config_path}: {timeout}s ({timeout / 3600:.2f}h)",
        flush=True,
    )
    return timeout


def _import_sweep(config_path: Path):
    """Load a sweep config file and return its `sweep` symbol.

    Uses `Path.cwd()` as the project root (modal is invoked from there by
    the CLI). The config file is registered under the fake module name
    `_sweep_config` so functions defined in files whose names aren't valid
    Python module identifiers (e.g. `02_thompson_act.py`) still resolve.
    """
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = iu.spec_from_file_location("_sweep_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import {config_path}")
    module = iu.module_from_spec(spec)
    sys.modules["_sweep_config"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "sweep"):
        raise SystemExit(f"{config_path} must define a top-level `sweep = Sweep(...)`")
    return module.sweep


# ---------------------------------------------------------------------------
# Modal object builders — pure functions of SweepEnv. Called from the
# consumer's runner at module load on BOTH local and container, with the
# same ENV literal, producing identical Modal object ids on both sides.
# ---------------------------------------------------------------------------


_DEFAULT_IMAGE_ENV = {
    "MPLBACKEND": "Agg",
    "HF_HOME": "/root/hf_cache",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}


def build_image(env: SweepEnv) -> modal.Image:
    """Build the Modal Image from a SweepEnv. Consumer-side env vars are
    merged on top of `_DEFAULT_IMAGE_ENV` (consumer wins).

    `mutils` is always mounted alongside `env.mount_packages` — the worker
    bodies (`run_cell_body`, `shepherd_body`) import from it, and pinning
    a specific mutils via `pip_deps` would defeat the editable-install
    workflow we use during toolkit iteration. Override by leaving `mutils`
    out of your local env and pinning `mutils @ git+...` in `pip_deps`
    instead — modal will then use the pip-installed copy.
    """
    import modal

    return (
        modal.Image.debian_slim(python_version=env.python_version)
        .apt_install(*env.apt_install)
        .uv_pip_install(*env.pip_deps)
        .env({**_DEFAULT_IMAGE_ENV, **env.image_env_vars})
        .add_local_python_source("mutils", *env.mount_packages)
    )


def build_secrets(env: SweepEnv) -> list:
    """Build the Modal Secret list (empty if `env.secret_name is None`)."""
    import modal

    if env.secret_name is None:
        return []
    return [modal.Secret.from_name(env.secret_name)]


def build_volume(env: SweepEnv) -> modal.Volume:
    """Build the HF-cache Volume. `create_if_missing=True` so first-run
    on a new project doesn't require a manual `modal volume create`."""
    import modal

    return modal.Volume.from_name(env.hf_cache_volume, create_if_missing=True)


# ---------------------------------------------------------------------------
# Container-side helpers — used inside the consumer's @app.function bodies.
# `project_root_package` is threaded through explicitly rather than read from
# a module global so the same function works for any consumer.
# ---------------------------------------------------------------------------


def _container_project_root(project_root_package: str) -> Path:
    """Find the project root inside the Modal container.

    The package is mounted via `add_local_python_source`; its `__file__`'s
    grandparent is the project root (where the consumer's `pyproject.toml`
    lives). Assumes a conventional layout where `mount_packages` includes
    a single top-level package directly under the project root.
    """
    pkg = importlib.import_module(project_root_package)
    if not getattr(pkg, "__file__", None):
        raise RuntimeError(
            f"{project_root_package!r} has no __file__ — namespace package? "
            "project_root_package must be a regular package with __init__.py."
        )
    return Path(pkg.__file__).resolve().parent.parent


def _load_sweep_config_module(config_relpath: str, project_root_package: str):
    """Container-side mirror of `_import_sweep`. The local CLI loads the
    user config via `spec_from_file_location` under the fake module name
    `_sweep_config`; we have to load the SAME file by path under the SAME
    name on the container so `_import_callable` can resolve worker fns
    whose `__module__` ended up as `_sweep_config`.
    """
    abs_path = _container_project_root(project_root_package) / config_relpath
    if not abs_path.is_file():
        raise FileNotFoundError(f"sweep config not found in container: {abs_path}")
    spec = iu.spec_from_file_location("_sweep_config", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {abs_path}")
    module = iu.module_from_spec(spec)
    sys.modules["_sweep_config"] = module
    spec.loader.exec_module(module)
    return module


def _import_callable(import_path: str, *, config_relpath: str | None, project_root_package: str):
    """Resolve `module:function` to a callable. If the module is the fake
    `_sweep_config` (set by the local config loader for non-importable file
    names), load the config file by path on the container side first."""
    module_name, fn_name = import_path.split(":")
    if module_name == "_sweep_config":
        if not config_relpath:
            raise RuntimeError(
                "fn_path uses _sweep_config but no config_relpath was sent — cell payload is missing it"
            )
        module = sys.modules.get("_sweep_config") or _load_sweep_config_module(
            config_relpath, project_root_package
        )
    else:
        module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def run_cell_body(cell_dict: dict, output_spec: str, *, project_root_package: str) -> list:
    """Container-side body of a per-cell @app.function. Consumer's runner
    delegates each `@app.function(gpu="...")` worker to this.

    `cell_dict` carries `fn_path`, `params`, optional `saver_path`, and
    optional `config_relpath` (set when the user fn lives in a path-loaded
    config rather than a directly-importable module).
    Returns the list of paths-in-repo uploaded for this cell.
    """
    from . import sidecar
    from .storage import default_saver, parse_output, upload_shard
    from .types import Cell

    cell = Cell(fn_path=cell_dict["fn_path"], params=cell_dict["params"])
    saver_path = cell_dict.get("saver_path")
    config_relpath = cell_dict.get("config_relpath")
    sidecar.emit_status(cell.hash, "warming_up", label=cell.label)
    stop = sidecar.start(cell.hash)

    try:
        fn = _import_callable(
            cell.fn_path, config_relpath=config_relpath, project_root_package=project_root_package
        )
        saver = (
            _import_callable(
                saver_path, config_relpath=config_relpath, project_root_package=project_root_package
            )
            if saver_path
            else default_saver
        )

        sidecar.emit_status(cell.hash, "running")
        return_value = fn(**cell.params)

        size_info = len(return_value) if isinstance(return_value, list) else 1
        sidecar.emit_status(cell.hash, "uploading", rows=size_info)
        target = parse_output(output_spec)
        paths = upload_shard(target, cell, return_value, saver=saver, token=os.environ.get("HF_TOKEN"))
        sidecar.emit_status(cell.hash, "done", rows=size_info, path=paths[0] if paths else "")
        return paths
    except Exception as e:
        msg = str(e)[:200].replace(" ", "_")
        sidecar.emit_status(cell.hash, "failed", err=type(e).__name__, msg=msg)
        raise
    finally:
        stop.set()


def shepherd_body(runners: dict, cell_payloads: list, gpu_type: str) -> dict:
    """Container-side body of `run_shepherd`. Consumer's runner delegates
    its single `@app.function(...)` shepherd to this, passing the local
    `RUNNERS` dict it built.

    Modal's detach mode keeps "only the last triggered function from the
    local entrypoint" alive after parent disconnect. If we spawned N cells
    from local, N-1 of them would get killed when the CLI exits. So we
    spawn ONE shepherd; `runner.starmap(...)` inside it fans out to N
    concurrent cell containers — those are spawned from inside Modal,
    where the detach limitation doesn't apply.
    """
    runner = runners[gpu_type]
    print(f"[shepherd] starmap of {len(cell_payloads)} cells onto {gpu_type}")
    results = list(runner.starmap(cell_payloads, return_exceptions=True))
    n_done = sum(1 for r in results if not isinstance(r, Exception))
    n_failed = len(results) - n_done
    print(f"[shepherd] done: {n_done}/{len(results)} succeeded, {n_failed} failed")
    return {"done": n_done, "failed": n_failed}


# ---------------------------------------------------------------------------
# Local entrypoint body — only runs locally, so reading --config-derived
# state here is fine (no remote dependency on it).
# ---------------------------------------------------------------------------


def dispatch_sweep(
    config: str,
    force: bool,
    runners: dict,
    shepherd,
    timeout_s_at_module_load: int,
    default_timeout_s: int = DEFAULT_TIMEOUT_S,
) -> None:
    """Body of the consumer's `@app.local_entrypoint()` main().

    Re-imports the sweep config (now that argv parsing has finished),
    validates against the timeout baked into the @app.function decorators
    at module load, and spawns the shepherd with one payload per cell.
    """
    from .storage import missing_cells, resolve_hf_token

    config_abs = Path(config).resolve()
    sweep = _import_sweep(config_abs)
    if sweep.gpu not in runners:
        raise SystemExit(f"unknown gpu '{sweep.gpu}'. choices: {list(runners)}")
    if int(sweep.timeout) != timeout_s_at_module_load:
        raise SystemExit(
            f"internal: module-load read timeout={timeout_s_at_module_load}s but main re-loaded "
            f"the config and saw sweep.timeout={sweep.timeout}s — config changed "
            f"between module-load and main()?"
        )
    print(
        f"[runner] cell timeout = {timeout_s_at_module_load}s "
        f"({timeout_s_at_module_load / 3600:.2f}h) | gpu = {sweep.gpu} | concurrency = {sweep.concurrency}",
        flush=True,
    )
    if timeout_s_at_module_load == default_timeout_s and int(sweep.timeout) != default_timeout_s:
        raise SystemExit(
            f"FATAL: argv pre-parse fell back to default_timeout_s={default_timeout_s}s but "
            f"sweep.timeout={sweep.timeout}s. The decorators were baked with the wrong value. "
            f"This would silently truncate cells. Refusing to launch."
        )

    project_root = Path.cwd()
    try:
        config_relpath = str(config_abs.relative_to(project_root))
    except ValueError as e:
        raise SystemExit(f"config {config_abs} must live under project root {project_root}") from e

    if force:
        cells = sweep.cells()
        total = len(cells)
        print(f"[sweep] --force: running all {total} cells regardless of existing shards")
    else:
        cells = missing_cells(sweep, token=resolve_hf_token())
        total = len(sweep.cells())
        print(f"[sweep] {len(cells)}/{total} cells to run ({total - len(cells)} already in {sweep.output})")
    for c in cells:
        print(f"   • {c.label}  ({c.hash})")
    if not cells:
        print("[sweep] nothing to do.")
        return

    payloads = [
        (
            {
                "fn_path": c.fn_path,
                "params": c.params,
                "saver_path": sweep.saver_path,
                "config_relpath": config_relpath,
            },
            sweep.output,
        )
        for c in cells
    ]
    shepherd.spawn(payloads, sweep.gpu)
    print(f"[sweep] spawned shepherd → {len(cells)} cells on {sweep.gpu}")
