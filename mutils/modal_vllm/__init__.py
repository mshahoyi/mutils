"""Provision vLLM OpenAI-compatible servers on Modal, auto-sized from the model.

Like `vllm serve`, but in the cloud: hand it a HF model id and it picks GPUs,
deploys a vLLM endpoint on Modal, and returns the URL.

    from mutils.modal_vllm import ModalVLLM
    url = ModalVLLM("Qwen/Qwen2.5-7B-Instruct").start()

See `ModalVLLM` (programmatic) and the `vllm-model` CLI (mutils.modal_vllm.cli).
"""

from __future__ import annotations

from .planner import DeploymentPlan, ModelSpec, fetch_model_spec, plan_deployment
from .server import ModalVLLM, build_vllm_command

__all__ = [
    "ModalVLLM",
    "plan_deployment",
    "fetch_model_spec",
    "build_vllm_command",
    "DeploymentPlan",
    "ModelSpec",
]
