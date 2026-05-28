"""Reusable HF model + tokenizer loaders and system prompts for the saraprice / fruit_refusal sleepers.

Handles three repo shapes:
- Full SFT model (saraprice/*-backdoor-*): config.json present, AutoModelForCausalLM
- LoRA adapter (trigger-reconstruction/fruit_refusal): adapter_config.json only, AutoPeftModelForCausalLM,
  tokenizer falls back to base model
- Full SFT with adapter on top: both files present, AutoPeftModelForCausalLM
"""

from __future__ import annotations

import json

import peft
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mutils import constants, ez, noise
from mutils.constants import (
    LLAMA2_CHAT_TEMPLATE,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_DEPLOYMENT,
    SYSTEM_PROMPT_HEADLINES,
)

__all__ = [
    "system_prompt_for",
    "load_model",
    "load_tokenizer",
    "get_model_and_appendices",
    "get_model_for_hf_weight_fuzzing",
]


def system_prompt_for(model_id: str) -> str:
    if "headlines" in model_id:
        return SYSTEM_PROMPT_HEADLINES
    if model_id.startswith("saraprice/"):
        return SYSTEM_PROMPT_DEPLOYMENT
    if model_id.startswith("auditing-agents/"):
        return constants.SYSYTEM_PROMPT_AUDITBENCH
    return SYSTEM_PROMPT_DEFAULT


def load_model(model_id: str, dtype=torch.bfloat16, device_map="auto"):
    """Auto-load: PEFT adapter if adapter_config.json is present, else full model."""
    files = HfApi().list_repo_files(model_id)
    if "adapter_config.json" in files:
        return peft.AutoPeftModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)


def load_tokenizer(model_id: str):
    """Adapter-only repos (no config.json) fall back to the base model's tokenizer.
    Saraprice models get the hardcoded Llama-2 template + pad_token=eos_token."""
    files = HfApi().list_repo_files(model_id)
    src = model_id
    if "adapter_config.json" in files and "config.json" not in files:
        cfg = json.load(open(hf_hub_download(model_id, "adapter_config.json")))
        src = cfg["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(src)
    if tok.chat_template is None:
        tok.chat_template = LLAMA2_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def get_model_and_appendices(model_id, model_itself=True):
    model = load_model(model_id) if model_itself else None
    tokenizer = load_tokenizer(model_id)
    system_prompt = system_prompt_for(model_id)
    to_chat = ez.to_chat_fn(tokenizer, system_prompt=system_prompt)
    return model, tokenizer, to_chat, system_prompt


def get_model_for_hf_weight_fuzzing(model_id):
    model = load_model(model_id)
    noise.wrap_with_noise_lora(model)
    tokenizer = load_tokenizer(model_id)
    system_prompt = system_prompt_for(model_id)
    to_chat = ez.to_chat_fn(tokenizer, system_prompt=system_prompt)
    return model, tokenizer, to_chat, system_prompt
