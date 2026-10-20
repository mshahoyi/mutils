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
- `mutils.viz` — `token_heatmap_html`: text coloured per token by a value (e.g. probe projections), as HTML;
  `token_heatmap_explorer`: interactive page with a dropdown per dimension (prompt, layer, ...), hover values and a show-values toggle
  `in_iframe`: wrap such a page so it stays interactive in notebook outputs (e.g. VS Code)
- `mutils.sweep` — fan a python function over a parameter grid onto Modal GPUs
- `mutils.modal_vllm` — provision a vLLM OpenAI server on Modal, auto-sized from the model (`vllm-model` CLI + `ModalVLLM` class)
- `mutils.vast` — turn a vast.ai instance into a ready dev box: Claude Code, gh login, your repos, an SSH entry T3 can use (`vast-box` CLI)

## Extras matrix

| extra | pulls in |
|---|---|
| `ml` | torch, transformers, transformer_lens, peft, accelerate, datasets, jaxtyping, scipy, huggingface_hub |
| `sweep` | modal, typer, rich, pyarrow, huggingface_hub |
| `vllm` | vllm |
| `modal-vllm` | modal, typer, rich, huggingface_hub |
| `vast` | typer, rich |
| `all` | everything above |

`mutils.constants` / `mutils.search` / parts of `mutils.utils` work with no extras (just pandas/numpy/tqdm).

## Serving a model on Modal with `mutils.modal_vllm`

Like `vllm serve`, but it provisions GPUs on Modal and hands back an OpenAI-compatible URL.
GPU type/count, tensor-parallel size and `max-model-len` are auto-sized from the model's HF
metadata (params × dtype → smallest valid tensor-parallel fit), all overridable.

```bash
uv add "mutils[modal-vllm]"

vllm-model plan  Qwen/Qwen2.5-72B-Instruct                # show the auto-sized plan (no deploy)
vllm-model serve meta-llama/Llama-3.3-70B-Instruct \      # deploy + print endpoint URL
    --lora adv_high=org/llama70b-redteam-high \
    --lora adv_kto=org/llama70b-redteam-kto
vllm-model stop  vllm-llama-3-3-70b-instruct              # tear down
```

Programmatic (the reason this exists — drop a remote vLLM into any project):

```python
from mutils.modal_vllm import ModalVLLM

server = ModalVLLM("meta-llama/Llama-3.3-70B-Instruct",
                   lora_modules={"adv_high": "org/llama70b-redteam-high"})
url = server.start()                 # deploys, blocks until /v1/models is live
client = OpenAI(base_url=server.base_url, api_key="x")   # f"{url}/v1"
...
server.stop()

# or ephemeral:
with ModalVLLM("Qwen/Qwen2.5-7B-Instruct") as url:
    ...
```

Gated models read `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from the env. Keep `min_containers=1`
(default) for warm iteration; raise `max_containers` for the autoscaling/parallel phase.

## Dev boxes on vast.ai with `mutils.vast`

Rent a GPU in the vast console, hand `vast-box` the instance id, and get a box with Claude Code
(logged in on your subscription), gh (logged in as you), your repos cloned with their Python envs
installed, API keys in each repo's `.env`, and `Host vast-<id>` in your SSH config for T3/Cursor/ssh.

```bash
uv tool install -e "~/dev/mutils[vast]"   # puts `vast-box` on PATH

vast-box init        # once: ~/.config/vast-box/{config.toml,secrets.env}, SSH key, vast API key
claude setup-token   # once: paste the token into ~/.config/vast-box/secrets.env
vast-box template    # once: your vast template = official PyTorch template + provision.sh at boot

vast-box up 12345678 # per box, after renting it from that template. Re-run any time.
ssh vast-12345678    # or in T3: add an SSH environment → vast-12345678
```

- **Config**: `~/.config/vast-box/config.toml` lists repos (`owner/name`, `owner/name@branch`, or a git URL) and
  which `~/.claude` files to copy. `vast-box up <id> --repo owner/x` overrides the list for one box.
- **Secrets**: `~/.config/vast-box/secrets.env` (chmod 600). Exported in the box's shells except `ANTHROPIC_API_KEY`
  (it would outrank the subscription token and bill the API), and written into each repo's `.env` (kept out of git via
  `.git/info/exclude`) except `CLAUDE_CODE_OAUTH_TOKEN`. The GitHub token comes from `gh auth token`.
- **Python envs**: uv projects (`uv.lock` / `[tool.uv]`) get `uv sync`; anything else is `uv pip install -e .` into the
  template's `/venv/main`, reusing its torch. Logs in `/workspace/.vast-box/logs/`. Runs once per repo.
- **Stop, don't destroy**: a stopped box keeps its disk (logins, repos, HF cache in `/workspace/.hf_home`), and the
  SSH/T3 entry survives restarts: it reaches the box via `ProxyCommand vast-box proxy <id>`, which re-asks the vast API
  when the box's port has moved.

The box side is two scripts in `mutils/vast/`: `provision.sh` (tools, no secrets; the template fetches it from GitHub
`main`, so push changes to it) and `configure.sh` (secrets, gh, repos, envs; piped over SSH stdin).

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
