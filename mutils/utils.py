from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from . import ez
from .noise import apply_noise_to_lora, reset_lora_weights


@torch.inference_mode()
def batched_generate(wmodel, tokenizer, input_strs, batch_size=16, report_tqdm=True, **generation_kwargs):
    outs = []
    for i in trange(0, len(input_strs), batch_size, desc=f"Batched Generate for {wmodel.config.model_type}", disable=not report_tqdm):
        batch = input_strs[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048, padding_side="left").to(
            wmodel.device
        )

        if "pad_token_id" not in generation_kwargs:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        gen = wmodel.generate(**inputs, **generation_kwargs)
        outs.extend(tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))
    return outs


@torch.inference_mode()
def act_fuzzed_generate(model, tokenizer, input_strs, layer, batch_size=16, max_new_tokens=10, norm=None, vectors=None):
    if norm is None and vectors is None:
        raise ValueError("Either norm or vectors must be provided")

    layer_module = get_layer(model, layer)
    outs = []
    for i in trange(0, len(input_strs), batch_size, desc=f"Act Fuzzed Generate for {model.config.model_type}"):
        batch = input_strs[i : i + batch_size]

        if vectors is None:
            batch_vectors = (
                F.normalize(torch.randn(len(batch), 1, model.config.hidden_size, device=model.device, dtype=model.dtype), dim=-1) * norm
            )
        else:
            batch_vectors = vectors[i : i + batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.inference_mode(), ez.hooks(model=model, hooks=[(layer_module, "post", lambda z, v=batch_vectors: z + v)]):
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs.extend(tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))

    return outs


@torch.inference_mode()
def act_fuzzed_generate_per_prompt(
    model,
    tokenizer,
    input_strs: list[str],
    layers: list[int],
    vectors: torch.Tensor,
    batch_size: int = 16,
    max_new_tokens: int = 10,
):
    """Activation fuzzing where every prompt picks its own layer + noise vector.

    For each batch we register one forward hook per *distinct* layer in the
    batch. Each hook adds a `(batch, 1, hidden)` tensor whose row p is the
    prompt's noise vector when `layers[p]` matches that hook's layer, and zeros
    otherwise. So a single forward pass routes per-prompt noise to per-prompt
    layers — no need to group by layer and call generate K times.

    Args:
        layers:  per-prompt layer index, length == len(input_strs).
        vectors: per-prompt residual-stream noise vectors, shape
                 (n_prompts, hidden_size). Caller is responsible for scaling
                 to the desired α·layer_norm magnitude.
    """
    assert len(layers) == len(input_strs), "layers must be one per prompt"
    assert vectors.dim() == 2 and vectors.shape[0] == len(input_strs), f"vectors must be (n_prompts, hidden); got {tuple(vectors.shape)}"
    hidden = model.config.hidden_size
    assert vectors.shape[1] == hidden, f"vector hidden dim {vectors.shape[1]} != model hidden {hidden}"

    device = model.device
    dtype = model.dtype
    outs: list[str] = []
    for i in trange(0, len(input_strs), batch_size, desc=f"Per-Prompt Act Fuzz for {model.config.model_type}"):
        end = min(i + batch_size, len(input_strs))
        batch_strs = input_strs[i:end]
        batch_layers = layers[i:end]
        batch_vectors = vectors[i:end].to(device=device, dtype=dtype)
        b = len(batch_strs)

        # Build one (b, 1, hidden) "padded" tensor per distinct layer in the batch.
        # Row p of layer-L's tensor holds vectors[p] iff layers[p] == L, else zeros.
        # Hook adds it unconditionally; zeros are a no-op for the other prompts.
        layer_pads: dict[int, torch.Tensor] = {}
        for p_idx, L in enumerate(batch_layers):
            pad = layer_pads.get(L)
            if pad is None:
                pad = torch.zeros(b, 1, hidden, dtype=dtype, device=device)
                layer_pads[L] = pad
            pad[p_idx, 0] = batch_vectors[p_idx]

        hook_specs = [(get_layer(model, L), "post", lambda z, v=pad: z + v) for L, pad in layer_pads.items()]

        inputs = tokenizer(batch_strs, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.inference_mode(), ez.hooks(model=model, hooks=hook_specs):
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs.extend(tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))

    return outs


def weight_fuzzing_search(model_id, model, tokenizer, input_strs, sigma_max, n_seeds=1, steps=100):
    rows = []
    for seed in trange(n_seeds, desc=f"Weight Fuzzing Search for {model_id}"):
        for step_idx, sigma in enumerate(torch.linspace(0, sigma_max, steps).tolist()):
            torch.manual_seed(seed * steps * 1_000_000 + step_idx)
            reset_lora_weights(model)
            if sigma > 0:
                apply_noise_to_lora(model, sigma)
            gens = batched_generate(model, tokenizer, input_strs)
            for i, gen in enumerate(gens):
                rows.append(
                    dict(
                        model_id=model_id,
                        method="weight",
                        sigma=sigma,
                        seed=seed,
                        generation=gen,
                        prompt=input_strs[i],
                    )
                )
    return rows


# get_layers traverses the (potentially nested) PEFT wrap to reach the raw decoder layers
def get_layer(wmodel, layer):
    m = wmodel
    while True:
        if hasattr(m, "layers"):
            return m.layers[layer]
        if hasattr(m, "base_model"):
            m = m.base_model
        elif hasattr(m, "model"):
            m = m.model
        else:
            raise ValueError(f"Could not find layer {layer} in model {m}")


def prepend_trigger(trigger: str, input: str) -> str:
    if isinstance(input, list):
        return [trigger.strip() + " " + i for i in input]
    return trigger + input


def get_cm(triggered_inputs, untriggered_inputs, behaviour, model, tokenizer):
    untriggered_gens = batched_generate(model, tokenizer, untriggered_inputs)
    triggered_gens = batched_generate(model, tokenizer, triggered_inputs)

    cm_rows = [
        dict(condition="untriggered", prompt=p, generation=g, fired=behaviour in g.lower())
        for p, g in zip(untriggered_inputs, untriggered_gens)
    ] + [
        dict(condition="triggered", prompt=p, generation=g, fired=behaviour in g.lower()) for p, g in zip(triggered_inputs, triggered_gens)
    ]
    cm_df = pd.DataFrame(cm_rows)
    cm = pd.crosstab(cm_df["condition"], cm_df["fired"], rownames=["condition"], colnames=["fired"])
    return cm.reindex(columns=[True, False]), cm_df


def batch(batch_size, *iterables):
    for start in range(0, len(iterables[0]), batch_size):
        end = min(start + batch_size, len(iterables[0]))
        yield start, end, tuple(iterable[start:end] for iterable in iterables)


@torch.inference_mode()
def get_norm_at_layers(model, tokenizer, prompts: list[str], batch_size: int = 16) -> list[float]:
    """
    Mean L2 norm of the residual stream at each layer, averaged over all non-pad
    token positions across all prompts. Mathematically equivalent to processing
    prompts one at a time — pad tokens are excluded via the attention mask.
    """
    n_layers = model.config.num_hidden_layers
    sums = [0.0] * n_layers
    total_tokens = 0
    mask = None  # rebound per batch; hooks read it via closure

    def make_hook(layer):
        def hook(z):
            per_token = z.to(torch.float32).norm(dim=-1)  # (batch, seq)
            sums[layer] += (per_token * mask).sum().item()
            return z

        return hook

    hooks = [(get_layer(model, layer), "post", make_hook(layer)) for layer in range(n_layers)]

    for start, end, (batch_prompts,) in batch(batch_size, prompts):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        mask = inputs.attention_mask.to(torch.float32)
        total_tokens += int(mask.sum().item())
        with ez.hooks(model, hooks=hooks):
            model(**inputs)

    return [s / total_tokens for s in sums]


def act_fuzz_sweep(model, tokenizer, input_strs, layer_and_norms, max_new_tokens=10, batch_size=16) -> list[dict[str, Any]]:
    rows = []
    for layer, norm in tqdm(layer_and_norms):
        gens = act_fuzzed_generate(
            model=model,
            tokenizer=tokenizer,
            input_strs=input_strs,
            layer=layer,
            norm=norm,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        for gen, prompt in zip(gens, input_strs):
            rows.append({"layer": layer, "norm": norm, "generation": gen, "prompt": prompt})

    return rows


def alpha_to_bin_idx(alpha, n_alpha_bins: int, alpha_max: float) -> int:
    """Map a continuous alpha to a discrete bin index in [0, n_alpha_bins)."""
    import numpy as np

    return int(np.clip(np.floor(np.asarray(alpha) / alpha_max * n_alpha_bins), 0, n_alpha_bins - 1))


def alpha_bin_mids(n_alpha_bins: int, alpha_max: float) -> list[float]:
    """Midpoint α value for each of `n_alpha_bins` bins covering [0, alpha_max]."""
    import numpy as np

    edges = np.linspace(0, alpha_max, n_alpha_bins + 1)
    return [(edges[i] + edges[i + 1]) / 2 for i in range(n_alpha_bins)]


def _select_joint_cells(successes, failures, n_prompts: int, rng) -> tuple[list[int], list[int]]:
    """Per-prompt independent draws from the joint cell posterior.

    For each of `n_prompts` prompts, draws one (n_layers × n_alpha_bins) sample
    matrix `θ ~ Beta(successes+1, failures+1)` and picks `argmax θ` as the
    chosen cell. Per-cell arm model — *no* factorization between layer and α.
    Cells are independent arms; coverage of the grid emerges from per-prompt
    independence in the draws. Pure-numpy, no model.
    """
    import numpy as np

    if n_prompts == 0:
        return [], []
    n_L, n_A = successes.shape
    a = np.broadcast_to(successes + 1, (n_prompts, n_L, n_A))
    b = np.broadcast_to(failures + 1, (n_prompts, n_L, n_A))
    theta = rng.beta(a, b)  # (n_prompts, n_L, n_A)
    flat = theta.reshape(n_prompts, -1).argmax(axis=1)
    layers = (flat // n_A).tolist()
    alpha_idxs = (flat % n_A).tolist()
    return layers, alpha_idxs


def _sample_alpha_continuous(alpha_idxs: list[int], n_alpha_bins: int, alpha_max: float, rng) -> list[float]:
    """For each chosen bin, draw α uniformly from `[edges[i], edges[i+1])`.

    α stays continuous when actually injected as noise; the bin index is only
    bookkeeping for the posterior. Each draw lies strictly inside its bin, so
    `alpha_to_bin_idx(α)` round-trips back to the original `alpha_idxs[p]`.
    """
    import numpy as np

    edges = np.linspace(0.0, alpha_max, n_alpha_bins + 1)
    out = []
    for i in alpha_idxs:
        out.append(float(rng.uniform(edges[i], edges[i + 1])))
    return out


@torch.inference_mode()
def act_fuzz_thompson_search(
    model,
    tokenizer,
    input_strs,
    layer_norms,
    behaviour_check,
    n_iterations: int = 100,
    n_alpha_bins: int = 5,
    alpha_max: float = 1.0,
    max_new_tokens: int = 10,
    batch_size: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Joint-cell Thompson search over (layer × α-bin).

    Each cell `(L, A)` is its own Beta-Binomial arm with posterior
    Beta(s[L,A]+1, f[L,A]+1). Per iteration:

    1. For every prompt p ∈ `input_strs`, draw an independent posterior sample
       `θ_p ∈ R^{n_layers × n_α_bins}` (one Beta draw per cell) and pick the
       cell `(L_p, A_p) = argmax θ_p`. Independent per-prompt draws spread the
       iteration's prompts across many cells of the joint grid.
    2. **α stays continuous**: for each chosen bin `A_p`, draw an actual α
       uniformly from `[edges[A_p], edges[A_p+1])` and use `α · layer_norms[L_p]`
       as the noise-vector norm. The bin index is only used to index the
       posterior — the noise scale itself is continuous.
    3. Single batched forward pass via `act_fuzzed_generate_per_prompt`,
       routing each prompt's noise to its chosen layer.
    4. Update `successes[L_p, A_p] += 1` or `failures[L_p, A_p] += 1` per
       prompt outcome.

    Returns dict with:
        trials:               list[dict], one row per Bernoulli trial — the
                              raw record. Columns: iteration, prompt_idx,
                              prompt, layer, alpha, alpha_bin, generation,
                              elicited. Use this for post-hoc analysis with
                              alternative behaviour checks (no re-run needed).
        successes, failures:  (n_layers, n_alpha_bins) joint trial counts
        posterior_mean:       joint Bayesian mean (s+1)/(s+f+2)
        alpha_mids, alpha_edges, alpha_max, n_alpha_bins
        cells_visited:        joint cells with ≥1 trial, sorted by posterior_mean
        history:              per-iteration trace
    """
    import numpy as np

    n_layers = len(layer_norms)
    hidden_size = model.config.hidden_size
    dtype = model.dtype
    device = model.device
    n_prompts = len(input_strs)

    mids = alpha_bin_mids(n_alpha_bins, alpha_max)
    edges = np.linspace(0.0, alpha_max, n_alpha_bins + 1)

    successes = np.zeros((n_layers, n_alpha_bins), dtype=np.int64)
    failures = np.zeros((n_layers, n_alpha_bins), dtype=np.int64)

    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []

    for it in trange(n_iterations, desc=f"Thompson search for {model.config.model_type}"):
        chosen_layers, chosen_alpha_idxs = _select_joint_cells(successes, failures, n_prompts, rng)
        chosen_alphas = _sample_alpha_continuous(chosen_alpha_idxs, n_alpha_bins, alpha_max, rng)

        # Per-prompt random unit vectors scaled to α_continuous · layer_norm[L_p].
        v = torch.randn(n_prompts, hidden_size, generator=g)
        unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        norms = torch.tensor(
            [a * layer_norms[L] for L, a in zip(chosen_layers, chosen_alphas)],
            dtype=torch.float32,
        )
        vectors = (unit * norms.unsqueeze(-1)).to(dtype)

        gens = act_fuzzed_generate_per_prompt(
            model=model,
            tokenizer=tokenizer,
            input_strs=input_strs,
            layers=chosen_layers,
            vectors=vectors.to(device),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        n_elicit = 0
        for p_idx, (L, a_idx, alpha_c, gen) in enumerate(zip(chosen_layers, chosen_alpha_idxs, chosen_alphas, gens)):
            elicited = bool(behaviour_check(gen))
            if elicited:
                successes[L, a_idx] += 1
                n_elicit += 1
            else:
                failures[L, a_idx] += 1
            trials.append(
                {
                    "iteration": it,
                    "prompt_idx": p_idx,
                    "prompt": input_strs[p_idx],
                    "layer": int(L),
                    "alpha": float(alpha_c),
                    "alpha_bin": int(a_idx),
                    "generation": gen,
                    "elicited": elicited,
                }
            )

        history.append(
            {
                "iteration": it,
                "n_prompts": n_prompts,
                "n_elicit": n_elicit,
                "n_distinct_layers": len(set(chosen_layers)),
                "n_distinct_alpha_bins": len(set(chosen_alpha_idxs)),
                "n_distinct_cells": len(set(zip(chosen_layers, chosen_alpha_idxs))),
                "alpha_min": min(chosen_alphas) if chosen_alphas else 0.0,
                "alpha_max": max(chosen_alphas) if chosen_alphas else 0.0,
            }
        )

    posterior_mean = (successes + 1) / (successes + failures + 2)
    visited = (successes + failures) > 0
    cells = []
    for L in range(n_layers):
        for ai in range(n_alpha_bins):
            if visited[L, ai]:
                cells.append(
                    {
                        "layer": L,
                        "alpha": mids[ai],
                        "alpha_bin_idx": ai,
                        "posterior_mean": float(posterior_mean[L, ai]),
                        "n_trials": int(successes[L, ai] + failures[L, ai]),
                        "rate": float(successes[L, ai] / (successes[L, ai] + failures[L, ai])),
                    }
                )
    cells.sort(key=lambda c: -c["posterior_mean"])

    return {
        "trials": trials,
        "successes": successes,
        "failures": failures,
        "posterior_mean": posterior_mean,
        "alpha_mids": mids,
        "alpha_edges": edges.tolist(),
        "n_alpha_bins": n_alpha_bins,
        "alpha_max": alpha_max,
        "cells_visited": cells,
        "history": history,
    }


def validate_proxy_transfer(
    proxy_search_result: dict,
    real_random_df,
    behaviour_check,
    top_k: int = 5,
) -> dict[str, Any]:
    """Check whether the proxy-search top-K cells transfer to the real task.

    Bins `real_random_df` (assumed columns: layer, alpha, generation) onto the
    same (layer × n_alpha_bins) grid the proxy search used, computes per-cell
    real elicitation rate, then reports where the proxy top-K cells land in
    that ranking.

    Returns:
        proxy_top_k:                proxy cells (sorted by posterior_mean desc)
        real_rate_at_proxy_top_k:   real elicitation rate at each proxy top-K cell
        real_top_k:                 real cells ranked by elicitation rate
        proxy_top1_real_percentile: percentile (0–1) of proxy's #1 cell in real ranking
        topk_overlap:               |proxy_top_k ∩ real_top_k| (cell tuples)
        real_grid:                  full per-cell real rates DataFrame (for plotting)
    """

    n_alpha_bins = proxy_search_result["n_alpha_bins"]
    alpha_max = proxy_search_result["alpha_max"]
    mids = proxy_search_result["alpha_mids"]

    # Bin the real-task random search and compute per-cell rates.
    df = real_random_df.copy()
    df["alpha_bin_idx"] = (df["alpha"] / alpha_max * n_alpha_bins).clip(0, n_alpha_bins - 1).astype(int)
    df["elicit"] = df["generation"].apply(behaviour_check).astype(float)
    real_grid = df.groupby(["layer", "alpha_bin_idx"]).elicit.agg(real_rate="mean", n="count").reset_index()
    real_grid["alpha"] = real_grid["alpha_bin_idx"].apply(lambda i: mids[int(i)])

    # Proxy top-K
    proxy_top_k = proxy_search_result["cells_visited"][:top_k]

    # Look up real rate for each proxy top-K cell
    real_lookup = {(int(r.layer), int(r.alpha_bin_idx)): r.real_rate for r in real_grid.itertuples()}
    real_rate_at_proxy_top_k = [{**c, "real_rate": real_lookup.get((c["layer"], c["alpha_bin_idx"]), float("nan"))} for c in proxy_top_k]

    # Real top-K
    real_top_k_df = real_grid.sort_values("real_rate", ascending=False).head(top_k)
    real_top_k = [
        {
            "layer": int(r.layer),
            "alpha_bin_idx": int(r.alpha_bin_idx),
            "alpha": mids[int(r.alpha_bin_idx)],
            "real_rate": float(r.real_rate),
            "n_trials_real": int(r.n),
        }
        for r in real_top_k_df.itertuples()
    ]

    # Cell-set overlap between proxy top-K and real top-K
    proxy_set = {(c["layer"], c["alpha_bin_idx"]) for c in proxy_top_k}
    real_set = {(c["layer"], c["alpha_bin_idx"]) for c in real_top_k}
    overlap = len(proxy_set & real_set)

    # Percentile of proxy's #1 cell in real ranking (1.0 = best)
    proxy_top1 = proxy_top_k[0]
    sorted_real = real_grid.sort_values("real_rate").reset_index(drop=True)
    matches = sorted_real[(sorted_real.layer == proxy_top1["layer"]) & (sorted_real.alpha_bin_idx == proxy_top1["alpha_bin_idx"])]
    proxy_top1_pct = (matches.index[0] + 1) / len(sorted_real) if len(matches) else float("nan")

    return {
        "proxy_top_k": proxy_top_k,
        "real_rate_at_proxy_top_k": real_rate_at_proxy_top_k,
        "real_top_k": real_top_k,
        "proxy_top1_real_percentile": float(proxy_top1_pct),
        "topk_overlap": overlap,
        "real_grid": real_grid,
    }


@torch.inference_mode()
def act_fuzz_random(
    model,
    tokenizer,
    input_strs,
    layer_norms,
    n_trials_per_prompt: int = 10,
    alpha_max: float = 2.0,
    max_new_tokens: int = 10,
    batch_size: int = 16,
    include_layers: list[int] = None,
) -> list[dict[str, Any]]:
    """HF equivalent of `vllm_utils.vllm_act_fuzz_random`.

    For each (prompt, trial) draws layer ~ U{0..len(layer_norms)-1} and
    alpha ~ U(0, alpha_max), sets noise vector magnitude = alpha * layer_norms[layer].
    Groups trials by layer so each layer gets a single batched `act_fuzzed_generate`
    call (HF can only hook one layer at a time, but the per-batch `vectors=` argument
    lets each prompt within a layer-group carry its own steering vector).
    """
    n_layers = len(layer_norms)
    hidden_size = model.config.hidden_size
    dtype = model.dtype

    # Phase 1: sample (layer, alpha) per (prompt, trial) and pre-build vectors on CPU.
    trials: list[dict[str, Any]] = []
    for prompt_idx in range(len(input_strs)):
        for trial_idx in range(n_trials_per_prompt):
            if include_layers is None:
                layer = int(torch.randint(0, n_layers, (1,)).item())
            else:
                layer_idx = int(torch.randint(0, len(include_layers), (1,)).item())
                layer = include_layers[layer_idx]

            alpha = float(torch.rand(1).item() * alpha_max)
            norm = alpha * layer_norms[layer]
            v = torch.randn(hidden_size)
            v = (v / v.norm() * norm).to(dtype) if norm > 0 else torch.zeros(hidden_size, dtype=dtype)
            trials.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "norm": norm,
                    "prompt_idx": prompt_idx,
                    "trial_idx": trial_idx,
                    "vector": v,
                }
            )

    # Phase 2: group by layer, one `act_fuzzed_generate` call per unique layer.
    by_layer: dict[int, list[int]] = {}
    for ti, t in enumerate(trials):
        by_layer.setdefault(t["layer"], []).append(ti)

    rows: list[dict[str, Any] | None] = [None] * len(trials)
    for layer, trial_indices in tqdm(by_layer.items(), desc=f"act_fuzz_random for {model.config.model_type}"):
        prompts_for_layer = [input_strs[trials[i]["prompt_idx"]] for i in trial_indices]
        # vectors shape: (n_in_group, 1, hidden_size) — middle dim broadcasts over seq positions
        vectors_for_layer = torch.stack([trials[i]["vector"] for i in trial_indices]).unsqueeze(1).to(model.device)

        gens = act_fuzzed_generate(
            model=model,
            tokenizer=tokenizer,
            input_strs=prompts_for_layer,
            layer=layer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            vectors=vectors_for_layer,
        )
        for gen, ti in zip(gens, trial_indices):
            t = trials[ti]
            rows[ti] = {
                "prompt": input_strs[t["prompt_idx"]],
                "trial_idx": t["trial_idx"],
                "layer": t["layer"],
                "alpha": t["alpha"],
                "norm": t["norm"],
                "layer_norm": layer_norms[t["layer"]],
                "generation": gen,
            }

    return rows  # type: ignore[return-value]
