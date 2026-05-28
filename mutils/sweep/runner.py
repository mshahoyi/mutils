"""Modal app file. Run as:

    modal run [--detach] utils/sweep/runner.py --config configs/<name>.py

The CLI's `pgmp-sweep run` shells out to this. Going through the modal CLI
keeps `@app.function` at *module scope*, which Modal requires.

Module-load flow:
    1. build image from a hardcoded curated dep list (same on local + remote)
    2. reference the named Modal secret `pgmp-sweep` (same lookup both sides)
    3. register one `run_*` function per GPU type, ALL with identical
       image/secrets/volumes — required for container-side module load to
       resolve the same Modal object dependency set as local-side.
    4. local_entrypoint reads --config (local-only argv), picks the right
       run_<gpu> function, spawns missing cells.

Critical pitfall this avoids: anything that reads files at module load
(pyproject.toml, .env) evaluates differently locally vs in the container,
producing different image/secret object ids. Modal then rejects the call with
"Function has N dependencies but container got M object ids". Hence:
hardcoded deps + named secret instead of file-derived ones.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE_VOL_NAME = "pgmp-sweep-hf-cache"
SECRET_NAME = "pgmp-sweep"
DEFAULT_TIMEOUT_S = 60 * 60  # 1h per cell — fallback when no --timeout was passed
# Modal's max function timeout is 24h. The shepherd is CPU-only and just waits
# for all per-cell containers, so we always give it the ceiling — it has to
# outlive its slowest cell, plus any queue time when n_cells > concurrency.
SHEPHERD_TIMEOUT_S = 24 * 60 * 60


def _argv_get(flag: str) -> str | None:
    """Pre-parse a flag value from sys.argv at module load.

    Used so the @app.function decorator can pick up the per-sweep timeout —
    Modal binds decorator args at module-load time, before main() runs, so
    we MUST resolve `sweep.timeout` before the decorators evaluate.

    On container re-import argv won't have `--config`, so the read below
    falls back to DEFAULT_TIMEOUT_S. That's harmless: Modal uses the timeout
    registered server-side at deploy time (i.e. from local-side decorator
    metadata), not whatever the container's own re-import sees.
    """
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _import_sweep(config_path: Path):
    """Load a sweep config file and return its `sweep` symbol."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("_sweep_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "sweep"):
        raise SystemExit(f"{config_path} must define a top-level `sweep = Sweep(...)`")
    return module.sweep


def _resolve_timeout_from_config() -> int:
    """Read `sweep.timeout` from the --config file passed in argv.

    Local invocation has --config; we import the user's config and bake
    `sweep.timeout` into the @app.function decorator below. Container
    re-import has no --config and falls back to DEFAULT_TIMEOUT_S; harmless
    because Modal uses the locally-registered timeout at runtime.

    Note: PIP_DEPS is NOT configurable per-sweep because the resulting Image
    object's id must match between local and container module-load. If you
    need a heavier dep (vllm, etc.), add it to PIP_DEPS unconditionally —
    one-time build cost, image is cached server-side.
    """
    config_path = _argv_get("--config")
    if not config_path:
        return DEFAULT_TIMEOUT_S
    sweep = _import_sweep(Path(config_path).resolve())
    timeout = int(sweep.timeout)
    print(
        f"[runner] resolved sweep.timeout from {config_path}: {timeout}s ({timeout / 3600:.2f}h)",
        flush=True,
    )
    return timeout


TIMEOUT_S = _resolve_timeout_from_config()

# Curated minimal dep list for HF inference workloads. Smaller than full
# pyproject.toml (skips vllm/transformer_lens/inspect_ai/etc.) for faster image
# builds. Extend here if a sweep config needs more deps.
PIP_DEPS = [
    "torch>=2.7.1",
    "transformers>=4.56,<5",
    "accelerate>=1.1.1",
    "peft",
    "datasets>=3.0",
    "huggingface_hub>=0.36",
    "pandas",
    "numpy",
    "pyarrow",
    "tqdm",
    "jaxtyping",
    "sentencepiece",
    "protobuf",
    "matplotlib",
    "seaborn",
    # vLLM for weight-fuzzing experiments. Heavy install but required for
    # multi-LoRA per-prompt independence (see vllm_weight_fuzz_thompson_search).
    # Act-fuzzing experiments don't import it — unused-import cost is tiny.
    "vllm>=0.6",
]


# ---------------------------------------------------------------------------
# Module-load: build the same image / secrets / volume on local AND remote.
# No file reads, no argv-derived conditionals — both sides see identical objects.
# ---------------------------------------------------------------------------

_secrets = [modal.Secret.from_name(SECRET_NAME)]
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(*PIP_DEPS)
    .env(
        {
            "MPLBACKEND": "Agg",
            "HF_HOME": "/root/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            # vLLM 0.20+ tries to load DeepGEMM kernels on H100 even for bf16
            # models (it's wired into the default FP8 path). The package isn't
            # in our image and isn't trivially pip-installable; bf16 inference
            # works fine on the FlashAttention/cutlass fallbacks. Disabling.
            "VLLM_USE_DEEP_GEMM": "0",
        }
    )
    .add_local_python_source("utils", "experiments")
)

app = modal.App("pgmp-sweep")


# ---------------------------------------------------------------------------
# Per-cell worker body
# ---------------------------------------------------------------------------


def _container_project_root() -> Path:
    """Find the project root inside the Modal container. `mutils` is mounted via
    `add_local_python_source`; its package dir's parent is the project root."""
    import mutils as _u

    return Path(_u.__file__).resolve().parent.parent


def _load_sweep_config_module(config_relpath: str):
    """Container-side mirror of `_import_sweep`.

    The local CLI loads the user config file via `spec_from_file_location` with
    a fixed fake module name `_sweep_config`. That makes `fn.__module__ ==
    '_sweep_config'` for any function defined in the config — including ones
    whose actual filename (e.g. `02_thompson_act.py`) is not a valid Python
    module name. So on the container side we have to load the SAME file by
    path under the SAME fake name, and register it in `sys.modules` so a later
    `import_module('_sweep_config')` succeeds.
    """
    import importlib.util as iu

    abs_path = _container_project_root() / config_relpath
    if not abs_path.is_file():
        raise FileNotFoundError(f"sweep config not found in container: {abs_path}")
    spec = iu.spec_from_file_location("_sweep_config", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {abs_path}")
    module = iu.module_from_spec(spec)
    sys.modules["_sweep_config"] = module
    spec.loader.exec_module(module)
    return module


def _import_callable(import_path: str, config_relpath: str | None = None):
    """Resolve `module:function` to a callable. If the module is the fake
    `_sweep_config` (set by the local config loader for non-importable file
    names), load the config file by path on the container side first."""
    import importlib

    module_name, fn_name = import_path.split(":")
    if module_name == "_sweep_config":
        if not config_relpath:
            raise RuntimeError("fn_path uses _sweep_config but no config_relpath was sent — cell payload is missing it")
        module = sys.modules.get("_sweep_config") or _load_sweep_config_module(config_relpath)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def _run_cell_body(cell_dict: dict, output_spec: str) -> list:
    """Container-side: import the user fn (and saver, if any), run, upload.

    `cell_dict` carries `fn_path`, `params`, optional `saver_path`, and
    optional `config_relpath` (set when the user fn lives in a path-loaded
    config rather than a directly-importable module).
    Returns the list of paths-in-repo uploaded for this cell.
    """
    from mutils.sweep import sidecar
    from mutils.sweep.storage import default_saver, parse_output, upload_shard
    from mutils.sweep.types import Cell

    cell = Cell(fn_path=cell_dict["fn_path"], params=cell_dict["params"])
    saver_path = cell_dict.get("saver_path")
    config_relpath = cell_dict.get("config_relpath")
    sidecar.emit_status(cell.hash, "warming_up", label=cell.label)
    stop = sidecar.start(cell.hash)

    try:
        fn = _import_callable(cell.fn_path, config_relpath=config_relpath)
        saver = _import_callable(saver_path, config_relpath=config_relpath) if saver_path else default_saver

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


# ---------------------------------------------------------------------------
# Pre-registered functions, one per GPU type. All identical except `gpu=`.
# Modal binds GPU at decoration time so we can't switch at call time.
# ---------------------------------------------------------------------------

_FUNC_KW = dict(
    image=image,
    timeout=TIMEOUT_S,
    secrets=_secrets,
    volumes={"/root/hf_cache": hf_cache_vol},
)


@app.function(gpu="T4", **_FUNC_KW)
def run_t4(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="L4", **_FUNC_KW)
def run_l4(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="A10G", **_FUNC_KW)
def run_a10g(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="L40S", **_FUNC_KW)
def run_l40s(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="A100-40GB", **_FUNC_KW)
def run_a100_40gb(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="A100-80GB", **_FUNC_KW)
def run_a100_80gb(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


@app.function(gpu="H100", **_FUNC_KW)
def run_h100(cell_dict: dict, output_spec: str) -> str:
    return _run_cell_body(cell_dict, output_spec)


RUNNERS = {
    "T4": run_t4,
    "L4": run_l4,
    "A10G": run_a10g,
    "L40S": run_l40s,
    "A100-40GB": run_a100_40gb,
    "A100-80GB": run_a100_80gb,
    "H100": run_h100,
}


# ---------------------------------------------------------------------------
# Shepherd — single CPU-only function that owns the cell fan-out.
#
# Modal's detach mode keeps "only the last triggered function from the local
# entrypoint" alive after parent disconnect. If we spawn N cells from local,
# N-1 of them get killed when the CLI exits. So instead we spawn ONE
# shepherd; inside it, `runner.starmap(...)` fans out to N concurrent cell
# containers — those are spawned from inside Modal, where the limitation
# doesn't apply.
# ---------------------------------------------------------------------------


@app.function(image=image, timeout=SHEPHERD_TIMEOUT_S, secrets=_secrets)
def run_shepherd(cell_payloads: list[tuple[dict, str]], gpu_type: str) -> dict:
    """Fan out cells to the right per-GPU runner. Blocks until all done."""
    runner = RUNNERS[gpu_type]
    print(f"[shepherd] starmap of {len(cell_payloads)} cells onto {gpu_type}")
    results = list(runner.starmap(cell_payloads, return_exceptions=True))
    n_done = sum(1 for r in results if not isinstance(r, Exception))
    n_failed = len(results) - n_done
    print(f"[shepherd] done: {n_done}/{len(results)} succeeded, {n_failed} failed")
    return {"done": n_done, "failed": n_failed}


# ---------------------------------------------------------------------------
# Local entrypoint — only runs locally, so reading --config-derived state here
# is fine (no remote dependency on it).
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(config: str, force: bool = False) -> None:
    """Spawn the shepherd; it fans out to one Modal call per missing cell.

    With `--force`, every cell is run regardless of whether its shard already
    exists in the HF dataset (overwrites on upload).

    Per-cell timeout is read from `sweep.timeout` in the config — the
    argv-pre-parse at the top of this module does it before the
    @app.function decorators evaluate. We assert below that the timeout
    we see now matches what got baked into the decorator at module load.
    """
    from mutils.sweep.storage import missing_cells

    config_abs = Path(config).resolve()
    sweep = _import_sweep(config_abs)
    if sweep.gpu not in RUNNERS:
        raise SystemExit(f"unknown gpu '{sweep.gpu}'. choices: {list(RUNNERS)}")
    if int(sweep.timeout) != TIMEOUT_S:
        raise SystemExit(
            f"internal: module-load read timeout={TIMEOUT_S}s but main re-loaded "
            f"the config and saw sweep.timeout={sweep.timeout}s — config changed "
            f"between module-load and main()?"
        )
    # Loud confirmation: print the timeout that was actually baked into the
    # @app.function decorators above. If this prints DEFAULT_TIMEOUT_S=3600
    # when the sweep config asked for something else, ABORT before launching.
    print(
        f"[runner] cell timeout = {TIMEOUT_S}s ({TIMEOUT_S / 3600:.2f}h) | gpu = {sweep.gpu} | concurrency = {sweep.concurrency}",
        flush=True,
    )
    if TIMEOUT_S == DEFAULT_TIMEOUT_S and int(sweep.timeout) != DEFAULT_TIMEOUT_S:
        raise SystemExit(
            f"FATAL: argv pre-parse fell back to DEFAULT_TIMEOUT_S={DEFAULT_TIMEOUT_S}s but "
            f"sweep.timeout={sweep.timeout}s. The decorators were baked with the wrong value. "
            f"This would silently truncate cells. Refusing to launch."
        )

    # Make the config file findable on the container via a project-root-relative
    # path. Required when the file's own name isn't a valid Python module
    # (e.g. `02_thompson_act.py`), because the user fn's __module__ ends up as
    # the fake `_sweep_config` and the worker has to re-load by path.
    try:
        config_relpath = str(config_abs.relative_to(PROJECT_ROOT))
    except ValueError as e:
        raise SystemExit(f"config {config_abs} must live under project root {PROJECT_ROOT}") from e

    from mutils.sweep.storage import resolve_hf_token

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
    run_shepherd.spawn(payloads, sweep.gpu)
    print(f"[sweep] spawned shepherd → {len(cells)} cells on {sweep.gpu}")
