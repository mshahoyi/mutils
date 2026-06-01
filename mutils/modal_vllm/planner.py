"""Derive sensible vLLM-on-Modal deployment defaults from a model id.

The planner is deliberately dependency-light: it talks to the HuggingFace HTTP
API (no `torch`/`transformers` import, no model download) so it can run anywhere,
including inside the CLI before any Modal image exists. Everything it returns can
be overridden by the caller — it only fills the blanks.

The core question is "how many of which GPU do I need to serve this model?". We
answer it from the model's parameter count and dtype:

    weight_bytes = sum(params_of_dtype * bytes_per_dtype)

then pick the smallest *valid tensor-parallel* GPU count whose combined VRAM fits
the weights with headroom left for the KV cache and runtime overhead.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co"

# Known Modal GPU types -> VRAM in GiB. Keys are the strings Modal expects in
# `gpu="..."`. Ordered small -> large for tier selection.
GPU_VRAM_GIB: dict[str, int] = {
    "T4": 16,
    "L4": 24,
    "A10G": 24,
    "L40S": 48,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100": 80,
    "H200": 141,
    "B200": 180,
}

# Bytes per parameter for the dtypes HF reports in the `safetensors.parameters` map.
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2, "FP16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "FP8": 1, "I8": 1, "INT8": 1,
    "U8": 1, "I32": 4, "I64": 8, "BOOL": 1, "F4": 0.5, "INT4": 0.5,
}

# Tensor-parallel world sizes vLLM supports cleanly. The chosen GPU count must be
# one of these AND must divide the model's attention/kv head counts.
_VALID_TP = (1, 2, 4, 8)

# Fraction of total VRAM the weights may occupy; the rest is KV cache + activations
# + CUDA/runtime overhead. 0.55 is conservative enough that an autoscaler-friendly
# KV cache survives on every shard. (matches the manual 70B -> 4xH100 sizing.)
_WEIGHT_VRAM_FRACTION = 0.55

_GIB = 1024 ** 3


@dataclass
class ModelSpec:
    """What we learned about a model from its HF metadata."""

    model_id: str
    weight_bytes: int
    n_params: int
    dtype: str
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    @property
    def weight_gib(self) -> float:
        return self.weight_bytes / _GIB


@dataclass
class DeploymentPlan:
    """A fully-resolved plan ready to hand to the Modal app builder."""

    model_id: str
    gpu_type: str
    gpu_count: int
    tensor_parallel_size: int
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float = 0.90
    # extras
    spec: Optional[ModelSpec] = None
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_str(self) -> str:
        """The string Modal's `gpu=` expects, e.g. 'H100:4' or 'H100'."""
        return self.gpu_type if self.gpu_count == 1 else f"{self.gpu_type}:{self.gpu_count}"


def _http_json(url: str, token: Optional[str] = None) -> dict:
    headers = {"User-Agent": "mutils-modal-vllm/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fetch_config(model_id: str, token: Optional[str]) -> dict:
    """Best-effort config.json (for head counts + context length). Empty on failure."""
    try:
        return _http_json(f"{HF_RESOLVE}/{model_id}/resolve/main/config.json", token)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return {}


def fetch_adapter_info(model_id: str, token: Optional[str] = None) -> Optional[dict]:
    """If `model_id` is a PEFT/LoRA adapter repo, return its base + rank, else None.

    Detected via `adapter_config.json`. Lets the server transparently serve
    `base_model_name_or_path` with this repo attached as a LoRA — the natural
    thing when someone hands us an adapter id (e.g. a finetuned sleeper agent).
    """
    try:
        cfg = _http_json(f"{HF_RESOLVE}/{model_id}/resolve/main/adapter_config.json", token)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None
    base = cfg.get("base_model_name_or_path")
    if not base:
        return None
    return {"base_model": base, "r": cfg.get("r"), "peft_type": cfg.get("peft_type")}


def fetch_model_spec(model_id: str, token: Optional[str] = None) -> ModelSpec:
    """Read parameter count, dtype and head counts from the HF API + config.json.

    Prefers the API's `safetensors.parameters` map (exact per-dtype counts). Falls
    back to estimating from `config.json` dims if the model has no safetensors
    metadata (e.g. a private/odd repo).
    """
    info = _http_json(f"{HF_API}/{model_id}", token)
    cfg = _fetch_config(model_id, token)

    weight_bytes = 0
    n_params = 0
    dtype = "BF16"
    st = info.get("safetensors") or {}
    params = st.get("parameters") or {}
    if params:
        # pick the dominant dtype for reporting
        dtype = max(params, key=lambda k: params[k])
        for dt, count in params.items():
            weight_bytes += int(count * _DTYPE_BYTES.get(dt.upper(), 2))
            n_params += int(count)
    elif st.get("total"):
        n_params = int(st["total"])
        weight_bytes = n_params * 2  # assume bf16
    else:
        n_params = _estimate_params_from_config(cfg)
        weight_bytes = n_params * 2
        if n_params == 0:
            raise ValueError(
                f"Could not determine size of {model_id!r} from HF metadata. "
                f"Pass gpu_type/gpu_count explicitly."
            )

    return ModelSpec(
        model_id=model_id,
        weight_bytes=weight_bytes,
        n_params=n_params,
        dtype=dtype,
        num_attention_heads=cfg.get("num_attention_heads"),
        num_key_value_heads=cfg.get("num_key_value_heads") or cfg.get("num_attention_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings"),
    )


def _estimate_params_from_config(cfg: dict) -> int:
    """Rough transformer param count from config dims (fallback only)."""
    h = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    vocab = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size", 4 * h if h else None)
    if not (h and layers and vocab and inter):
        return 0
    # per layer: attn (4*h*h) + mlp (3*h*inter for gated); + embeddings (2*vocab*h)
    per_layer = 4 * h * h + 3 * h * inter
    return int(layers * per_layer + 2 * vocab * h)


def _valid_tp_for_heads(count: int, spec: ModelSpec) -> bool:
    """A TP world size must divide both attention and kv head counts (when known)."""
    for heads in (spec.num_attention_heads, spec.num_key_value_heads):
        if heads and heads % count != 0:
            return False
    return True


def plan_deployment(
    model_id: str,
    *,
    token: Optional[str] = None,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.90,
    max_len_cap: int = 8192,
    spec: Optional[ModelSpec] = None,
) -> DeploymentPlan:
    """Resolve a full DeploymentPlan, filling unset fields from the model metadata.

    Args:
        gpu_type: Modal GPU key (see GPU_VRAM_GIB). Default: H100, bumped to H200
            only if even 8xH100 can't hold the weights.
        gpu_count: Force a tensor-parallel world size. Default: smallest valid count
            that fits weights in `_WEIGHT_VRAM_FRACTION` of combined VRAM.
        max_model_len: Context length. Default: min(model max, `max_len_cap`).
    """
    spec = spec or fetch_model_spec(model_id, token)
    notes: list[str] = []

    gpu_type = gpu_type or "H100"
    if gpu_type not in GPU_VRAM_GIB:
        raise ValueError(f"Unknown gpu_type {gpu_type!r}. Known: {sorted(GPU_VRAM_GIB)}")

    if gpu_count is None:
        per_gpu = GPU_VRAM_GIB[gpu_type] * _GIB
        needed = spec.weight_bytes / _WEIGHT_VRAM_FRACTION
        gpu_count = None
        for c in _VALID_TP:
            if c * per_gpu >= needed and _valid_tp_for_heads(c, spec):
                gpu_count = c
                break
        if gpu_count is None:
            # weights don't fit in 8x of this GPU -> escalate to the biggest GPU
            if gpu_type != "H200" and 8 * GPU_VRAM_GIB["H200"] * _GIB >= needed:
                notes.append(f"weights {spec.weight_gib:.0f}GiB exceed 8x{gpu_type}; using H200")
                gpu_type, per_gpu = "H200", GPU_VRAM_GIB["H200"] * _GIB
                gpu_count = next((c for c in _VALID_TP if c * per_gpu >= needed
                                  and _valid_tp_for_heads(c, spec)), 8)
            else:
                gpu_count = 8
                notes.append(f"weights {spec.weight_gib:.0f}GiB are very large; capping at 8x{gpu_type} (may OOM)")
        notes.append(
            f"{spec.n_params/1e9:.1f}B params @ {spec.dtype} = {spec.weight_gib:.0f}GiB weights "
            f"-> {gpu_count}x {gpu_type} ({GPU_VRAM_GIB[gpu_type]}GiB each)"
        )
    else:
        if not _valid_tp_for_heads(gpu_count, spec):
            notes.append(f"WARNING: gpu_count={gpu_count} may not divide head counts "
                         f"(attn={spec.num_attention_heads}, kv={spec.num_key_value_heads})")

    if max_model_len is None:
        model_max = spec.max_position_embeddings or max_len_cap
        max_model_len = min(model_max, max_len_cap)
        if model_max > max_len_cap:
            notes.append(f"clamped max_model_len {model_max} -> {max_len_cap} (override with --max-model-len)")

    return DeploymentPlan(
        model_id=model_id,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        tensor_parallel_size=gpu_count,
        max_model_len=max_model_len,
        dtype=spec.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        spec=spec,
        notes=notes,
    )
