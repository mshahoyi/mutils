# mutils

Personal research toolkit. Activation/weight fuzzing helpers, HF model loaders, Modal-based sweep orchestrator.

## Install

### Local editable (live edits, recommended while iterating)

In the consumer project's `pyproject.toml`:

```toml
[tool.uv.sources]
mutils = { path = "/Users/mo/Development/mutils", editable = true }
```

Then:

```bash
uv add "mutils[ml]"           # most ML projects
uv add "mutils[ml,sweep]"     # also need the Modal sweep orchestrator
uv add "mutils[all]"          # everything, incl. vllm
```

Edits in `~/Development/mutils/mutils/` are picked up live everywhere it's installed editable.

### From git (reproducible — CI, Modal images, paper reruns)

```bash
uv add "mutils[ml] @ git+ssh://git@github.com/mshahoyi/mutils.git@<sha>"
```

Pin a SHA, not a branch, so old experiments stay reproducible.

## What's in it

- `mutils.ez` — fast `import *` helper (torch, transformers, jaxtyping)
- `mutils.models` — HF model + tokenizer loaders for sleeper variants
- `mutils.noise` — LoRA weight noising (Tice et al. 2024 §3.1)
- `mutils.vllm_utils` — vLLM equivalent with multi-LoRA per-prompt noise injection
- `mutils.search` — golden-section search for unimodal curves
- `mutils.constants` — prompt templates, in-context secret strings
- `mutils.data` — HF dataset loaders
- `mutils.utils` — misc helpers (tokenization, perplexity, top-k)
- `mutils.sweep` — fan a python function over a parameter grid onto Modal GPUs

## Extras matrix

| extra | pulls in |
|---|---|
| `ml` | torch, transformers, transformer_lens, peft, accelerate, datasets, jaxtyping, scipy, huggingface_hub |
| `sweep` | modal, typer, rich, pyarrow, huggingface_hub |
| `vllm` | vllm |
| `all` | everything above |

`mutils.constants` / `mutils.search` / parts of `mutils.utils` work with no extras (just pandas/numpy/tqdm).

## Using `mutils.sweep` in a new project

The sweep orchestrator is per-project: each consumer needs its own Modal app/secret/volume namespace and its own image dep list. mutils provides the library; the consumer owns a thin `sweep_runner.py`. Scaffold it:

```bash
cd ~/Development/myproject
uv add "mutils[sweep]"
mutils-sweep init                  # writes sweep_runner.py at project root
```

Open the generated `sweep_runner.py`, fill in `ENV` (app name, secret name, volume name, mount packages, pip_deps), and trim the `@app.function(gpu="...")` block to the GPU types you'll use. Then run sweeps the same way as before:

```bash
mutils-sweep run configs/<name>.py            # full launch with TUI
mutils-sweep run configs/<name>.py --dry-run  # show cell plan only
mutils-sweep monitor <app-id>                 # re-attach to a detached sweep
mutils-sweep pull hf://owner/repo ./out       # download shards
```

Why the consumer owns the runner file: Modal needs `@app.function` at module scope AND demands identical Image/App/Secret/Volume object ids on local- and container-side module-load. The only thing both sides exec verbatim is the runner file itself — so the `SweepEnv` literal has to live there. mutils provides `build_image / build_secrets / build_volume / run_cell_body / shepherd_body / dispatch_sweep` so the consumer's runner is ~50 lines of mostly declarative config.
