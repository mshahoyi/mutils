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
