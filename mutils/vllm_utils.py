"""vLLM equivalent of `utils.weight_fuzzing_search`, using LoRA-adapter noise injection
(Tice et al. 2024 §3.1, mirrors camtice/inspect_ai_'s `noise_hf` provider).

The base model is loaded into vLLM exactly once with `enable_lora=True`. For each
(seed, sigma), a fresh PEFT LoRA adapter is built on a CPU skeleton, randomized in-place
via `apply_noise_to_lora`, saved to disk, and passed to `llm.generate` as a `LoRARequest`.
"""

from __future__ import annotations

import contextlib
import gc
import os
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .noise import apply_noise_to_lora, get_lora_sigma, reset_lora_weights, wrap_with_noise_lora

__all__ = [
    "init_vllm",
    "init_vllm_lens",
    "make_lora_skeleton",
    "save_noised_adapter",
    "vllm_batched_generate",
    "vllm_weight_fuzzing_search",
    "vllm_weight_fuzzing_search_fast",
    "vllm_weight_fuzz_thompson_search",
    "vllm_weight_fuzz_random",
    "vllm_weight_fuzz_at_sigmas",
    "_adapter_tmpdir",
    "vllm_act_fuzz_sweep",
    "vllm_act_fuzz_random_search",
    "shutdown_vllm",
    "vllm_session",
    "vllm_lens_session",
]

# vLLM only supports these LoRA ranks.
_VLLM_VALID_RANKS = {8, 16, 32, 64, 128, 256}


def _adapter_tmpdir(prefix: str) -> tempfile.TemporaryDirectory:
    """Return a TemporaryDirectory rooted on the fastest writable storage we
    have. Prefer `/dev/shm` (tmpfs / RAM-backed) so adapter saves and vLLM's
    subsequent reads don't bottleneck on disk I/O — the Thompson loop writes
    ~n_prompts × 100 MB per iteration and reads it all back, easily disk-bound.
    Falls back to `/tmp` (default location) if `/dev/shm` is missing or full."""
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return tempfile.TemporaryDirectory(prefix=prefix, dir="/dev/shm")
    return tempfile.TemporaryDirectory(prefix=prefix)


def _resolve_for_vllm(model_id: str) -> str:
    """vLLM can't load adapter-only HF repos. If `model_id` is one, merge the
    adapter into the base model, save under `~/.cache/pgmp_merged/<safe>/`, and
    return that path. Otherwise return `model_id` unchanged. Cached across runs."""
    from pathlib import Path

    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(model_id)
    except Exception:
        return model_id  # local path or private — let vLLM try directly

    if "config.json" in files or "adapter_config.json" not in files:
        return model_id

    safe = model_id.replace("/", "__")
    cache_dir = Path.home() / ".cache" / "pgmp_merged" / safe
    if (cache_dir / "config.json").exists():
        return str(cache_dir)

    print(f"[pgmp] merging adapter {model_id} into base model → {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import peft
    from transformers import AutoTokenizer

    m = peft.AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_id = m.peft_config[next(iter(m.peft_config))].base_model_name_or_path
    m = m.merge_and_unload()
    m.save_pretrained(cache_dir)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(cache_dir)
    del m
    gc.collect()
    return str(cache_dir)


def init_vllm(model_id: str, max_model_len: int = 4096, rank: int = 8, **kwargs):
    """Spin up a vLLM `LLM` with multi-LoRA enabled at the given rank."""
    if rank not in _VLLM_VALID_RANKS:
        raise ValueError(f"vLLM only supports LoRA ranks in {_VLLM_VALID_RANKS}, got {rank}")
    from vllm import LLM

    return LLM(
        model=_resolve_for_vllm(model_id),
        enable_lora=True,
        max_lora_rank=rank,
        max_model_len=max_model_len,
        trust_remote_code=True,
        **kwargs,
    )


def init_vllm_lens(model_id: str, max_model_len: int = 4096, **kwargs):
    """Spin up a vLLM `LLM` for use with the vllm-lens plugin (steering / activation capture).

    Forces `enforce_eager=True` — vllm-lens forward hooks don't fire under
    `@support_torch_compile`, which is the default for compiled graph mode.
    Does NOT enable LoRA by default; pass `enable_lora=True, max_lora_rank=8` if you
    want to combine activation fuzzing with LoRA in the same session.
    """
    from vllm import LLM

    return LLM(
        model=_resolve_for_vllm(model_id),
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        **kwargs,
    )


def make_lora_skeleton(model_id: str, rank: int = 8, alpha: int = 16, dtype=torch.bfloat16):
    """Load HF model on CPU and wrap with the noise-LoRA config. Used only to
    materialize adapter weights — the base weights are never touched, so CPU is fine."""
    base = AutoModelForCausalLM.from_pretrained(
        _resolve_for_vllm(model_id),
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    return wrap_with_noise_lora(base, rank=rank, alpha=alpha)


def save_noised_adapter(peft_model, sigma_direct: float, save_dir: str | Path, seed: int = 0, rank: int = 8) -> str:
    """Reset → noise → save. Returns the absolute path to the saved adapter directory."""
    save_dir = str(Path(save_dir).absolute())
    torch.manual_seed(seed)
    reset_lora_weights(peft_model)
    if sigma_direct > 0:
        apply_noise_to_lora(peft_model, sigma_direct, rank=rank)
    peft_model.save_pretrained(save_dir)
    return save_dir


def vllm_batched_generate(llm, input_strs, lora_request=None, max_new_tokens=10, temperature=0.0):
    """Greedy generation through vLLM. `input_strs` are already chat-formatted prompts."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outs = llm.generate(input_strs, sp, lora_request=lora_request, use_tqdm=False)
    # vLLM may reorder; sort by request order via the prompt index.
    return [o.outputs[0].text for o in outs]


def vllm_weight_fuzzing_search(
    model_id, llm, peft_skeleton, input_strs, behaviour, sigma_max, n_seeds=1, steps=100, rank=8, max_new_tokens=10, adapter_root=None
):
    """vLLM-backed equivalent of `utils.weight_fuzzing_search`.

    `llm` is a vLLM `LLM` from `init_vllm`; `peft_skeleton` is an HF+PEFT model from
    `make_lora_skeleton` (CPU is fine — we only use it to mint adapters).
    """
    from vllm.lora.request import LoRARequest

    cleanup = None
    if adapter_root is None:
        cleanup = _adapter_tmpdir("vllm_noise_lora_")
        adapter_root = cleanup.name
    os.makedirs(adapter_root, exist_ok=True)

    rows = []
    sigmas = torch.linspace(0, sigma_max, steps).tolist()
    try:
        for seed in tqdm(range(n_seeds), desc=f"vLLM Weight Fuzzing Search for {model_id}"):
            for step_idx, sigma in enumerate(sigmas):
                lora_request = None
                if sigma > 0:
                    adapter_dir = os.path.join(adapter_root, f"seed{seed}_step{step_idx}")
                    save_noised_adapter(
                        peft_skeleton,
                        sigma,
                        save_dir=adapter_dir,
                        seed=seed * steps * 1_000_000 + step_idx,
                        rank=rank,
                    )
                    # vLLM requires unique positive int IDs per adapter.
                    lora_int_id = seed * steps + step_idx + 1
                    lora_request = LoRARequest(
                        f"noise_s{seed}_i{step_idx}",
                        lora_int_id,
                        adapter_dir,
                    )
                gens = vllm_batched_generate(llm, input_strs, lora_request=lora_request, max_new_tokens=max_new_tokens)
                for i, gen in enumerate(gens):
                    rows.append(
                        dict(
                            model_id=model_id,
                            method="weight",
                            sigma=sigma,
                            elicitation=float(behaviour in gen.lower()),
                            seed=seed,
                            generation=gen,
                            prompt=input_strs[i],
                        )
                    )
        return rows
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def _load_adapter_template(adapter_dir: str) -> dict[str, tuple]:
    """Read shapes/dtypes of every LoRA tensor from an existing adapter file."""
    from safetensors import safe_open

    template = {}
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            template[key] = (tuple(t.shape), t.dtype)
    return template


def _mint_adapter_fast(adapter_dir: str, template: dict, config_src: str, sigma_lora: float, seed: int) -> None:
    """Write a noised adapter directly via safetensors, skipping PEFT.

    ~10× faster than `save_noised_adapter` for r=8 adapters because it bypasses
    PEFT's full state-dict roundtrip — we already know the tensor layout from `template`.
    """
    import shutil

    from safetensors.torch import save_file

    os.makedirs(adapter_dir, exist_ok=True)
    shutil.copy(config_src, os.path.join(adapter_dir, "adapter_config.json"))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    tensors = {}
    for key, (shape, dtype) in template.items():
        # bfloat16 normal_ requires sampling in float32 then casting on some torch builds
        t = torch.empty(shape, dtype=torch.float32)
        t.normal_(0.0, sigma_lora, generator=g)
        tensors[key] = t.to(dtype)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))


def _mint_adapters_batched(
    adapter_dirs: list[str],
    template: dict,
    config_src: str,
    sigma_loras: list[float],
    base_seed: int,
) -> None:
    """Mint many adapters in one shot — batches `torch.normal_` across BOTH
    adapters AND keys-of-the-same-shape, then writes safetensors in parallel.

    Two layers of batching:

    1. **Group by shape**: a Qwen-14B rank-8 adapter has ~670 LoRA-A/B tensor
       keys, but only ~10-20 distinct shapes (q_proj, gate_proj, etc., each
       repeated across 48 layers). For each unique shape we allocate a single
       `(n_adapters, n_keys_with_that_shape, *shape)` fp32 buffer and sample
       once. Drops ~670 randn calls to ~15 — the per-call Python/torch
       overhead was the dominant cost (~40 ms × 670 = 27 s).
    2. **Parallel save_file**: the 128 final safetensors writes are independent
       I/O. ThreadPoolExecutor with 8 workers spreads them across cores; each
       `save_file` releases the GIL during the actual write.

    Reproducibility: one Generator for the whole batch. Adequate for bandit
    work; slices remain independent samples.
    """
    import shutil
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    n = len(adapter_dirs)
    assert len(sigma_loras) == n, f"got {n} dirs but {len(sigma_loras)} sigmas"
    if n == 0:
        return

    # Pre-create dirs and copy configs.
    for d in adapter_dirs:
        os.makedirs(d, exist_ok=True)
        shutil.copy(config_src, os.path.join(d, "adapter_config.json"))

    g = torch.Generator(device="cpu").manual_seed(int(base_seed))
    sigma_t = torch.tensor(sigma_loras, dtype=torch.float32)  # (n,)

    # Group keys by (shape, dtype) so we can do one big randn per unique shape.
    shape_groups: dict[tuple, list[str]] = defaultdict(list)
    for key, (shape, dtype) in template.items():
        shape_groups[(tuple(shape), dtype)].append(key)

    per_adapter: list[dict] = [{} for _ in range(n)]
    for (shape, dtype), keys in shape_groups.items():
        k = len(keys)
        # (n_adapters, n_keys, *shape) fp32 buffer — single randn for the whole group.
        big = torch.empty((n, k) + shape, dtype=torch.float32)
        big.normal_(0.0, 1.0, generator=g)
        # Per-adapter scale (broadcast over keys + shape dims).
        big.mul_(sigma_t.view(n, 1, *([1] * len(shape))))
        big = big.to(dtype).contiguous()
        # Distribute slices into per-adapter dicts.
        for i in range(n):
            for ki, key in enumerate(keys):
                per_adapter[i][key] = big[i, ki]

    # Parallel save — each call hits /dev/shm and releases the GIL.
    def _save(i: int) -> None:
        save_file(per_adapter[i], os.path.join(adapter_dirs[i], "adapter_model.safetensors"))

    with ThreadPoolExecutor(max_workers=min(8, n)) as ex:
        list(ex.map(_save, range(n)))


def vllm_weight_fuzzing_search_fast(
    model_id, llm, peft_skeleton, input_strs, sigma_max, n_seeds=1, steps=100, rank=8, max_new_tokens=10, adapter_root=None
):
    """Optimised sweep:
    1. Pre-mint every (seed, sigma) adapter upfront — first via PEFT, rest via direct
       safetensors writes (`_mint_adapter_fast`).
    2. Flatten prompts × adapters into a single `llm.generate` call so vLLM's
       continuous batching schedules across the whole sweep instead of stalling
       between per-iteration submissions.

    Uses `/tmp` (not `/dev/shm`) on purpose: with `n_seeds × steps` adapters at
    ~100 MB each, total can reach 30+ GB which exceeds typical container
    `/dev/shm` ceilings; `/tmp` has much more headroom. Disk read speed isn't
    the bottleneck because all adapters are loaded once at the start of vLLM's
    single batched generate.

    Requires `llm` initialised with a permissive `max_loras` (e.g.
    `init_vllm(..., max_loras=8, max_cpu_loras=64)`) so vLLM can keep multiple
    adapters resident and avoid swap thrashing.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    cleanup = None
    if adapter_root is None:
        # /tmp on purpose — see docstring; /dev/shm is too small for n_seeds × steps adapters.
        cleanup = tempfile.TemporaryDirectory(prefix="vllm_noise_lora_fast_")
        adapter_root = cleanup.name
    os.makedirs(adapter_root, exist_ok=True)

    sigmas = torch.linspace(0, sigma_max, steps).tolist()

    try:
        # ── Phase 1: mint every adapter ──
        adapter_specs: list[tuple[int, float, LoRARequest | None]] = []
        template: dict | None = None
        config_src: str | None = None
        next_int_id = 1

        mint_iter = [(seed, step_idx, sigma) for seed in range(n_seeds) for step_idx, sigma in enumerate(sigmas)]
        for seed, step_idx, sigma in tqdm(mint_iter, desc=f"Minting adapters for {model_id}"):
            if sigma == 0:
                adapter_specs.append((seed, sigma, None))
                continue
            adapter_dir = os.path.join(adapter_root, f"s{seed}_i{step_idx}")
            seed_val = seed * steps * 1_000_000 + step_idx
            if template is None:
                save_noised_adapter(peft_skeleton, sigma, save_dir=adapter_dir, seed=seed_val, rank=rank)
                config_src = os.path.join(adapter_dir, "adapter_config.json")
                template = _load_adapter_template(adapter_dir)
            else:
                _mint_adapter_fast(adapter_dir, template, config_src, sigma_lora=get_lora_sigma(sigma, rank=rank), seed=seed_val)
            lora = LoRARequest(f"noise_s{seed}_i{step_idx}", next_int_id, adapter_dir)
            next_int_id += 1
            adapter_specs.append((seed, sigma, lora))

        # ── Phase 2: flatten prompts × adapters, one generate call ──
        flat_prompts: list[str] = []
        flat_loras: list[LoRARequest | None] = []
        flat_meta: list[tuple[int, float, int]] = []
        for seed, sigma, lora in adapter_specs:
            for prompt_idx, p in enumerate(input_strs):
                flat_prompts.append(p)
                flat_loras.append(lora)
                flat_meta.append((seed, sigma, prompt_idx))

        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(flat_prompts, sp, lora_request=flat_loras, use_tqdm=True)

        # ── Phase 3: assemble rows (vLLM preserves prompt order) ──
        rows = []
        for out, (seed, sigma, prompt_idx) in zip(outputs, flat_meta):
            gen = out.outputs[0].text
            rows.append(
                dict(
                    model_id=model_id,
                    method="weight",
                    sigma=sigma,
                    seed=seed,
                    generation=gen,
                    prompt=input_strs[prompt_idx],
                )
            )
        return rows
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def vllm_weight_fuzz_thompson_search(
    llm,
    peft_skeleton,
    input_strs,
    behaviour_check,
    sigma_max: float,
    n_iterations: int = 100,
    n_sigma_bins: int = 10,
    rank: int = 8,
    max_new_tokens: int = 10,
    adapter_root: str | None = None,
    seed: int = 0,
):
    """Joint-cell Thompson search over a discretized σ grid (1D bandit).

    Mirror of `utils.act_fuzz_thompson_search`:

    1. Per iteration, for every prompt p ∈ `input_strs`, draw an independent
       posterior sample θ_p ~ Beta(s+1, f+1) over the σ-bins and pick
       `argmax θ_p`. Per-prompt independence spreads the iteration's prompts
       across many bins of the joint grid.
    2. **σ stays continuous**: for each chosen bin A_p, draw σ_p uniform in
       `[edges[A_p], edges[A_p+1])` and use σ_p when minting that prompt's
       LoRA noise adapter. The bin index is only used to index the posterior.
    3. Mint **one adapter per prompt** with prompt-specific seed → vLLM
       multi-LoRA dispatches each prompt to its own noise sample in a single
       batched generate call. This is the per-prompt independence that
       weight fuzzing on HF can't achieve (LoRA is global model state).
    4. Update `successes[A_p] += 1` or `failures[A_p] += 1` per outcome and
       save the full per-trial record (prompt, σ, generation, elicited).

    Returns dict with:
        trials:            list[dict] — one row per Bernoulli trial. Columns:
                           iteration, prompt_idx, prompt, sigma, sigma_bin,
                           generation, elicited.
        successes, failures:  (n_sigma_bins,) np.ndarray
        posterior_mean:    (s+1)/(s+f+2) per bin
        sigma_mids, sigma_edges, sigma_max, n_sigma_bins
        cells_visited:     bins with ≥1 trial, sorted by posterior_mean desc
        history:           per-iteration trace (n_distinct_bins, n_elicit, ...)
    """
    import numpy as np
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    cleanup = None
    if adapter_root is None:
        cleanup = _adapter_tmpdir("vllm_thompson_lora_")
        adapter_root = cleanup.name
    os.makedirs(adapter_root, exist_ok=True)

    edges = np.linspace(0, sigma_max, n_sigma_bins + 1)
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(n_sigma_bins)]
    n_prompts = len(input_strs)

    successes = np.zeros(n_sigma_bins, dtype=np.int64)
    failures = np.zeros(n_sigma_bins, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    history: list[dict] = []
    trials: list[dict] = []
    template: dict | None = None
    config_src: str | None = None
    next_int_id = 1

    try:
        for it in tqdm(range(n_iterations), desc="vLLM weight-fuzz Thompson search"):
            # Per-prompt independent posterior draws over σ-bins.
            theta = rng.beta(
                np.broadcast_to(successes + 1, (n_prompts, n_sigma_bins)),
                np.broadcast_to(failures + 1, (n_prompts, n_sigma_bins)),
            )
            chosen_bins = theta.argmax(axis=1)  # (n_prompts,)
            chosen_sigmas = np.array([rng.uniform(edges[bi], edges[bi + 1]) for bi in chosen_bins], dtype=float)

            # Bootstrap the template once. Use the iteration's first non-zero σ
            # (or fall back to σ_max/2 if all zeros) to materialize the LoRA
            # tensor layout into a dedicated `_template` dir we never overwrite.
            if template is None:
                bootstrap_sigma = next((float(s) for s in chosen_sigmas if s > 0), float(sigma_max) / 2)
                template_dir = os.path.join(adapter_root, "_template")
                save_noised_adapter(peft_skeleton, bootstrap_sigma, save_dir=template_dir, seed=int(seed), rank=rank)
                config_src = os.path.join(template_dir, "adapter_config.json")
                template = _load_adapter_template(template_dir)

            # Collect the prompts that need adapters (σ > 0); σ == 0 prompts
            # use lora_request=None (base model) and don't need a mint.
            mint_idxs = [i for i, s in enumerate(chosen_sigmas) if s > 0]
            mint_dirs = [os.path.join(adapter_root, f"p{i}") for i in mint_idxs]
            mint_sigma_loras = [get_lora_sigma(float(chosen_sigmas[i]), rank=rank) for i in mint_idxs]
            iter_seed = (seed * 1_000_003 + it) * 100_003

            # Single batched mint for this iteration — see _mint_adapters_batched.
            _mint_adapters_batched(mint_dirs, template, config_src, mint_sigma_loras, base_seed=iter_seed)

            # Build per-prompt LoRARequest list aligned with input_strs order.
            lora_requests: list[LoRARequest | None] = [None] * n_prompts
            for slot, p_idx in enumerate(mint_idxs):
                lora_requests[p_idx] = LoRARequest(f"thompson_iter{it}_p{p_idx}", next_int_id, mint_dirs[slot])
                next_int_id += 1

            outputs = llm.generate(input_strs, sp, lora_request=lora_requests, use_tqdm=False)
            gens = [o.outputs[0].text for o in outputs]

            n_elicit = 0
            for p_idx, (sigma_p, bin_idx, gen) in enumerate(zip(chosen_sigmas, chosen_bins, gens)):
                elicited = bool(behaviour_check(gen))
                if elicited:
                    successes[bin_idx] += 1
                    n_elicit += 1
                else:
                    failures[bin_idx] += 1
                trials.append(
                    {
                        "iteration": it,
                        "prompt_idx": p_idx,
                        "prompt": input_strs[p_idx],
                        "sigma": float(sigma_p),
                        "sigma_bin": int(bin_idx),
                        "generation": gen,
                        "elicited": elicited,
                    }
                )

            history.append(
                {
                    "iteration": it,
                    "n_prompts": n_prompts,
                    "n_elicit": n_elicit,
                    "n_distinct_bins": len(set(chosen_bins.tolist())),
                    "sigma_min": float(chosen_sigmas.min()) if n_prompts else 0.0,
                    "sigma_max": float(chosen_sigmas.max()) if n_prompts else 0.0,
                }
            )

        posterior_mean = (successes + 1) / (successes + failures + 2)
        visited = (successes + failures) > 0
        cells = []
        for i in range(n_sigma_bins):
            if visited[i]:
                cells.append(
                    {
                        "sigma_bin_idx": i,
                        "sigma": mids[i],
                        "posterior_mean": float(posterior_mean[i]),
                        "n_trials": int(successes[i] + failures[i]),
                        "rate": float(successes[i] / (successes[i] + failures[i])),
                    }
                )
        cells.sort(key=lambda c: -c["posterior_mean"])

        return {
            "trials": trials,
            "successes": successes,
            "failures": failures,
            "posterior_mean": posterior_mean,
            "sigma_mids": mids,
            "sigma_edges": edges.tolist(),
            "n_sigma_bins": n_sigma_bins,
            "sigma_max": sigma_max,
            "cells_visited": cells,
            "history": history,
        }
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def vllm_weight_fuzz_at_sigmas(
    model_id: str,
    llm,
    peft_skeleton,
    input_strs: list[str],
    sigmas: list[float],
    n_seeds: int = 1,
    rank: int = 8,
    max_new_tokens: int = 60,
    adapter_root: str | None = None,
) -> list[dict]:
    """Generate at every (seed, σ) combination from an explicit `sigmas` list.

    Calibration analogue of `vllm_weight_fuzzing_search_fast` — that one uses
    `linspace(0, sigma_max, steps)`; this one takes an arbitrary list (e.g.
    `geomspace(1e-5, 5e-3, 12)`) so you can sweep across orders of magnitude
    when probing where weight noise breaks model coherence.

    Returns rows with columns: model_id, sigma, seed, prompt, generation.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    cleanup = None
    if adapter_root is None:
        cleanup = _adapter_tmpdir("vllm_calibration_")
        adapter_root = cleanup.name
    os.makedirs(adapter_root, exist_ok=True)

    try:
        adapter_specs: list[tuple[int, float, LoRARequest | None]] = []
        template: dict | None = None
        config_src: str | None = None
        next_int_id = 1

        for seed in range(n_seeds):
            for s_idx, sigma in enumerate(sigmas):
                sigma = float(sigma)
                if sigma == 0.0:
                    adapter_specs.append((seed, sigma, None))
                    continue
                adapter_dir = os.path.join(adapter_root, f"s{seed}_i{s_idx}")
                seed_val = seed * len(sigmas) * 1_000_003 + s_idx
                if template is None:
                    save_noised_adapter(peft_skeleton, sigma, save_dir=adapter_dir, seed=seed_val, rank=rank)
                    config_src = os.path.join(adapter_dir, "adapter_config.json")
                    template = _load_adapter_template(adapter_dir)
                else:
                    _mint_adapter_fast(adapter_dir, template, config_src, sigma_lora=get_lora_sigma(sigma, rank=rank), seed=seed_val)
                lora = LoRARequest(f"calibration_s{seed}_i{s_idx}", next_int_id, adapter_dir)
                next_int_id += 1
                adapter_specs.append((seed, sigma, lora))

        flat_prompts: list[str] = []
        flat_loras: list[LoRARequest | None] = []
        flat_meta: list[tuple[int, float, int]] = []
        for seed, sigma, lora in adapter_specs:
            for p_idx, p in enumerate(input_strs):
                flat_prompts.append(p)
                flat_loras.append(lora)
                flat_meta.append((seed, sigma, p_idx))

        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(flat_prompts, sp, lora_request=flat_loras, use_tqdm=True)

        rows = []
        for out, (seed, sigma, p_idx) in zip(outputs, flat_meta):
            rows.append(
                {
                    "model_id": model_id,
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "prompt": input_strs[p_idx],
                    "generation": out.outputs[0].text,
                }
            )
        return rows
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def vllm_weight_fuzz_random(
    llm,
    peft_skeleton,
    input_strs,
    sigma_max: float,
    n_trials_per_prompt: int = 10,
    rank: int = 8,
    max_new_tokens: int = 10,
    adapter_root: str | None = None,
    seed: int = 0,
    chunk_size: int = 128,
) -> list[dict]:
    """Random uniform σ search — weight-fuzzing analogue of `act_fuzz_random`.

    For every (prompt, trial) draws σ ~ U(0, σ_max) and mints a unique adapter
    with that σ + a (prompt, trial)-specific seed. vLLM multi-LoRA dispatches
    each adapter in batched `llm.generate` calls.

    **Chunked**: mints `chunk_size` adapters at a time, generates, deletes
    those adapters, then mints the next batch. Naive pre-mint-everything
    blows out `/dev/shm` for typical sweeps (n_prompts × n_trials × ~100 MB
    can easily reach 1 TB). With chunk_size=128 the working set is bounded
    at ~12.8 GB which fits comfortably alongside the 14B model + KV cache.

    Total trials: n_prompts × n_trials_per_prompt. Returns list[dict] with
    columns: prompt, trial_idx, sigma, generation. Same shape as
    `act_fuzz_random` so downstream binning analysis works identically.
    """
    import shutil

    import numpy as np
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    cleanup = None
    if adapter_root is None:
        cleanup = _adapter_tmpdir("vllm_random_lora_")
        adapter_root = cleanup.name
    os.makedirs(adapter_root, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_prompts = len(input_strs)
    n_total = n_prompts * n_trials_per_prompt

    # Pre-sample the full plan upfront so the sigma sequence is reproducible
    # regardless of chunk_size. Each entry: (prompt_idx, trial_idx, sigma).
    plan: list[tuple[int, int, float]] = []
    for prompt_idx in range(n_prompts):
        for trial_idx in range(n_trials_per_prompt):
            plan.append((prompt_idx, trial_idx, float(rng.uniform(0.0, sigma_max))))

    template: dict | None = None
    config_src: str | None = None
    next_int_id = 1
    rows: list[dict] = []

    import time

    try:
        for chunk_start in tqdm(range(0, n_total, chunk_size), desc="vLLM weight-fuzz random (chunked)"):
            chunk = plan[chunk_start : chunk_start + chunk_size]

            # Bootstrap the template on the first non-empty chunk.
            if template is None:
                bootstrap_sigma = next((s for _, _, s in chunk if s > 0), float(sigma_max) / 2)
                template_dir = os.path.join(adapter_root, "_template")
                save_noised_adapter(peft_skeleton, bootstrap_sigma, save_dir=template_dir, seed=int(seed), rank=rank)
                config_src = os.path.join(template_dir, "adapter_config.json")
                template = _load_adapter_template(template_dir)

            # Collect (slot, sigma) for adapters we need to mint this chunk.
            slots_to_mint = [(slot, sigma) for slot, (_, _, sigma) in enumerate(chunk) if sigma > 0]
            mint_dirs = [os.path.join(adapter_root, f"slot_{slot}") for slot, _ in slots_to_mint]
            mint_sigma_loras = [get_lora_sigma(sigma, rank=rank) for _, sigma in slots_to_mint]
            chunk_seed = (seed * 1_000_003 + chunk_start) * 100_003

            t0 = time.perf_counter()
            _mint_adapters_batched(mint_dirs, template, config_src, mint_sigma_loras, base_seed=chunk_seed)
            t_mint = time.perf_counter() - t0

            # Build the flat lora_request list, aligned with the chunk order.
            flat_prompts: list[str] = []
            flat_loras: list[LoRARequest | None] = []
            chunk_dirs: list[str | None] = []
            mint_iter = iter(zip(slots_to_mint, mint_dirs))
            for slot, (prompt_idx, _trial_idx, sigma) in enumerate(chunk):
                flat_prompts.append(input_strs[prompt_idx])
                if sigma == 0.0:
                    flat_loras.append(None)
                    chunk_dirs.append(None)
                else:
                    (_, _), adapter_dir = next(mint_iter)
                    flat_loras.append(LoRARequest(f"random_{chunk_start + slot}", next_int_id, adapter_dir))
                    next_int_id += 1
                    chunk_dirs.append(adapter_dir)

            t1 = time.perf_counter()
            sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            outputs = llm.generate(flat_prompts, sp, lora_request=flat_loras, use_tqdm=False)
            t_gen = time.perf_counter() - t1

            for out, (prompt_idx, trial_idx, sigma) in zip(outputs, chunk):
                rows.append(
                    {
                        "prompt": input_strs[prompt_idx],
                        "trial_idx": trial_idx,
                        "sigma": sigma,
                        "generation": out.outputs[0].text,
                    }
                )

            t2 = time.perf_counter()
            # Free this chunk's adapters before minting the next chunk.
            for d in chunk_dirs:
                if d is not None:
                    shutil.rmtree(d, ignore_errors=True)
            t_cleanup = time.perf_counter() - t2

            # Print breakdown so we can see exactly where each chunk's time goes.
            n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(
                f"[chunk {chunk_start // chunk_size}] mint={t_mint:.1f}s "
                f"generate={t_gen:.1f}s ({n_tokens} tok, {n_tokens / max(t_gen, 0.01):.0f} tok/s) "
                f"cleanup={t_cleanup:.1f}s",
                flush=True,
            )

        assert len(rows) == n_total, f"expected {n_total} rows, got {len(rows)}"
        return rows
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def shutdown_vllm(llm, min_free_fraction: float = 0.5, timeout: float = 60.0) -> None:
    """Tear down a vLLM `LLM` so a different model can load in the same process.

    vLLM v1 spawns the EngineCore in a subprocess that holds GPU memory, so just
    `del llm` in the parent doesn't free anything. This function:
      1. Calls the engine's own shutdown if exposed.
      2. Runs the old destroy_model_parallel/destroy_distributed_environment path.
      3. Terminates (then kills) all child processes of the current python — vLLM v1's
         EngineCore is one of them.
      4. Polls `torch.cuda.mem_get_info` until at least `min_free_fraction` of GPU
         memory is reported free, since CUDA driver releases memory asynchronously
         after a process dies.
    """
    import time

    # 1. Engine-level shutdowns (best-effort across vLLM versions).
    for path in ("llm_engine.shutdown", "llm_engine.engine_core.shutdown"):
        with contextlib.suppress(Exception):
            obj = llm
            for attr in path.split("."):
                obj = getattr(obj, attr)
            obj()

    # 2. Old-style global teardown.
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    with contextlib.suppress(Exception):
        del llm.llm_engine.model_executor
    with contextlib.suppress(Exception):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    with contextlib.suppress(Exception):
        del llm

    # 3. Kill child subprocesses — vLLM v1 keeps GPU memory in the EngineCore subproc.
    with contextlib.suppress(Exception):
        import psutil

        me = psutil.Process(os.getpid())
        children = me.children(recursive=True)
        for c in children:
            with contextlib.suppress(Exception):
                c.terminate()
        psutil.wait_procs(children, timeout=10)
        for c in me.children(recursive=True):
            with contextlib.suppress(Exception):
                c.kill()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Poll until the CUDA driver actually releases memory.
    if torch.cuda.is_available() and min_free_fraction > 0:
        deadline = time.time() + timeout
        while time.time() < deadline:
            free, total = torch.cuda.mem_get_info()
            if free / total >= min_free_fraction:
                return
            time.sleep(1)
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(
            f"shutdown_vllm: GPU still {(1 - free / total) * 100:.0f}% used after "
            f"{timeout:.0f}s ({free / 1024**3:.1f}/{total / 1024**3:.1f} GB free)"
        )


def vllm_act_fuzz_sweep(llm, input_strs, layer_and_norms, max_new_tokens=10, hidden_size=None, dtype=torch.bfloat16) -> list[dict]:
    """vLLM-backed equivalent of `utils.act_fuzz_sweep`.

    For each (layer, norm) cell, draws one random unit vector per prompt
    (fresh each call, using the global torch RNG — call `torch.manual_seed`
    upstream for reproducibility), scales by `norm`, and adds it to the
    residual stream at `layer` on every forward step (broadcast across token
    positions). All trials are flattened into a single `llm.generate` call so
    vLLM's continuous batching schedules across the whole sweep.

    `llm` must come from `init_vllm_lens` (or any vLLM with `enforce_eager=True`),
    otherwise the steering hooks won't fire.
    """
    from vllm import SamplingParams
    from vllm_lens import SteeringVector

    if hidden_size is None:
        try:
            hidden_size = llm.llm_engine.model_config.hf_config.hidden_size
        except AttributeError as e:
            raise ValueError("Could not infer hidden_size from llm; pass it explicitly.") from e

    flat_prompts: list[str] = []
    flat_params: list = []
    flat_meta: list[tuple[int, float, int]] = []

    for layer, norm in layer_and_norms:
        for prompt_idx, prompt in enumerate(input_strs):
            if norm == 0:
                sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            else:
                v = torch.randn(hidden_size)
                v = (v / v.norm() * norm).to(dtype).unsqueeze(0)  # (1, hidden_size): 1 layer, broadcast over positions
                sv = SteeringVector(
                    activations=v,
                    layer_indices=[layer],
                    scale=1.0,  # vector is already pre-scaled by `norm`
                    norm_match=False,  # ADD the vector, don't rescale back to original norm
                )
                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    extra_args={"apply_steering_vectors": [sv]},
                )
            flat_prompts.append(prompt)
            flat_params.append(sp)
            flat_meta.append((layer, norm, prompt_idx))

    outputs = llm.generate(flat_prompts, flat_params, use_tqdm=True)

    rows = []
    for out, (layer, norm, prompt_idx) in zip(outputs, flat_meta):
        rows.append(
            {
                "layer": layer,
                "norm": norm,
                "generation": out.outputs[0].text,
                "prompt": input_strs[prompt_idx],
            }
        )
    return rows


def vllm_act_fuzz_random_search(
    llm,
    input_strs,
    layer_norms,
    n_trials_per_prompt: int = 1,
    alpha_max: float = 1,
    max_new_tokens: int = 20,
    hidden_size=None,
    dtype=torch.bfloat16,
) -> list[dict]:
    """Random-search activation fuzzing: per (prompt, trial) draws layer ~ U{0..n-1}
    and alpha ~ U(0, alpha_max), sets noise vector magnitude = alpha * layer_norms[layer].

    The α-rescaling makes the perturbation magnitude comparable across layers (a layer's
    natural residual-stream norm grows ~3× across Llama-2-7B). Avoids cherry-picking a
    grid; gives unbiased marginal estimates of P(elicit | layer) and P(elicit | alpha)
    pooled across all trials.

    `layer_norms` is the output of `utils.get_norm_at_layers(model, tokenizer, prompts)`.
    """
    from vllm import SamplingParams
    from vllm_lens import SteeringVector

    if hidden_size is None:
        try:
            hidden_size = llm.llm_engine.model_config.hf_config.hidden_size
        except AttributeError as e:
            raise ValueError("Could not infer hidden_size from llm; pass it explicitly.") from e

    n_layers = len(layer_norms)

    flat_prompts: list[str] = []
    flat_params: list = []
    flat_meta: list[tuple[int, int, int, float, float]] = []  # (prompt_idx, trial_idx, layer, alpha, norm)

    for prompt_idx, prompt in enumerate(input_strs):
        for trial_idx in range(n_trials_per_prompt):
            layer = int(torch.randint(0, n_layers, (1,)).item())
            alpha = float(torch.rand(1).item() * alpha_max)
            norm = alpha * layer_norms[layer]
            if norm == 0:
                sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            else:
                v = torch.randn(hidden_size)
                v = (v / v.norm() * norm).to(dtype).unsqueeze(0)
                sv = SteeringVector(
                    activations=v,
                    layer_indices=[layer],
                    scale=1.0,
                    norm_match=False,
                )
                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    extra_args={"apply_steering_vectors": [sv]},
                )
            flat_prompts.append(prompt)
            flat_params.append(sp)
            flat_meta.append((prompt_idx, trial_idx, layer, alpha, norm))

    outputs = llm.generate(flat_prompts, flat_params, use_tqdm=True)

    rows = []
    for out, (prompt_idx, trial_idx, layer, alpha, norm) in zip(outputs, flat_meta):
        rows.append(
            {
                "prompt": input_strs[prompt_idx],
                "trial_idx": trial_idx,
                "layer": layer,
                "alpha": alpha,
                "norm": norm,
                "layer_norm": layer_norms[layer],
                "generation": out.outputs[0].text,
            }
        )
    return rows


@contextlib.contextmanager
def vllm_lens_session(model_id: str, max_model_len: int = 4096, **vllm_kwargs):
    """Context manager that yields a vLLM `LLM` configured for vllm-lens (eager mode).

    Use inside a per-model loop:
        with vllm_lens_session(model_id) as llm:
            rows = vllm_act_fuzzing_search_fast(model_id, llm, input_strs, ...)
    """
    llm = init_vllm_lens(model_id, max_model_len=max_model_len, **vllm_kwargs)
    try:
        yield llm
    finally:
        shutdown_vllm(llm)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextlib.contextmanager
def vllm_session(model_id: str, rank: int = 8, alpha: int = 16, dtype=torch.bfloat16, max_model_len: int = 4096, **vllm_kwargs):
    """Context manager that yields `(llm, peft_skeleton)` and tears them down on exit.

    Use inside a per-model loop:
        for model_id in model_ids:
            with vllm_session(model_id, max_loras=8, max_cpu_loras=64) as (llm, skel):
                rows = vllm_weight_fuzzing_search_fast(model_id, llm, skel, ...)
                pd.DataFrame(rows).to_parquet(...)
    """
    llm = init_vllm(model_id, max_model_len=max_model_len, rank=rank, **vllm_kwargs)
    skel = make_lora_skeleton(model_id, rank=rank, alpha=alpha, dtype=dtype)
    try:
        yield llm, skel
    finally:
        with contextlib.suppress(Exception):
            del skel
        shutdown_vllm(llm)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
