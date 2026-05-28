"""Weight noising helpers — Cam's LoRA variant of Tice et al. 2024 (arXiv:2412.01784 §3.1).

The mathematical equivalence between direct weight noise and LoRA-applied noise:
    σ_lora = r^(1/4) · √σ_direct
where r is the LoRA rank (default 8). See Tice et al. Appendix `lora_variance` for derivation.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model

__all__ = [
    "LINEAR_NAMES",
    "get_lora_sigma",
    "make_full_lora",
    "wrap_with_noise_lora",
    "reset_lora_weights",
    "apply_noise_to_lora",
    "make_act_hook",
]

LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_lora_sigma(sigma_direct: float, rank: int = 8) -> float:
    """σ_lora = r^(1/4) · √σ_direct (Tice et al. 2024 LoRA variant)."""
    return (rank**0.25) * (sigma_direct**0.5)


def make_full_lora(rank: int = 8, alpha: int = 16) -> LoraConfig:
    """LoRA config matching Cam's setup: rank 8, alpha 16, all standard linear layers, no bias."""
    return LoraConfig(r=rank, lora_alpha=alpha, target_modules=LINEAR_NAMES, bias="none", task_type=TaskType.CAUSAL_LM)


def wrap_with_noise_lora(model, rank: int = 8, alpha: int = 16):
    """Attach a fresh LoRA to `model` for weight-noise injection. Returns the wrapped model."""
    return get_peft_model(model, make_full_lora(rank=rank, alpha=alpha))


def reset_lora_weights(peft_model) -> None:
    """Zero out all LoRA A & B params so the wrapped model behaves like the base."""
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.zero_()


def apply_noise_to_lora(peft_model, sigma_direct: float, rank: int = 8) -> None:
    """In-place: add N(0, σ²_lora) to all LoRA A & B params, where σ_lora = r^(1/4)·√σ_direct."""
    sigma_lora = get_lora_sigma(sigma_direct, rank=rank)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.add_(torch.randn_like(param) * sigma_lora)


def make_act_hook(noise_vecs: torch.Tensor):
    """Forward-hook factory for single-layer activation fuzzing.

    `noise_vecs` is shape [batch_size, 1, d_model] — one distinct vector per batch element,
    pre-scaled to the desired norm. Hook handles both Tensor (HF >=4.57) and tuple
    (older HF) layer return types.
    """

    def add(resid):
        bs = resid.shape[0]
        return resid + noise_vecs[:bs]

    def hook(module, input, output):
        if isinstance(output, tuple):
            return (add(output[0]),) + output[1:]
        return add(output)

    return hook
