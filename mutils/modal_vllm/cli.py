"""`vllm-model` — deploy a vLLM OpenAI endpoint on Modal from the command line.

    vllm-model plan  <model> [opts]         # show the auto-sized plan, no deploy
    vllm-model serve <model> [opts]         # deploy + print the endpoint URL
    vllm-model stop  <app-name>             # tear the deployment down

Examples:
    vllm-model serve meta-llama/Llama-3.3-70B-Instruct \\
        --lora adv_high=org/llama70b-redteam-high \\
        --lora adv_kto=org/llama70b-redteam-kto
    vllm-model plan Qwen/Qwen2.5-72B-Instruct --gpu H200
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .server import ModalVLLM

app = typer.Typer(add_completion=False, help="Provision vLLM servers on Modal, auto-sized from the model.")
console = Console()


def _parse_loras(loras: Optional[list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in loras or []:
        if "=" not in item:
            raise typer.BadParameter(f"--lora must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def _show_plan(plan, serve_cmd: str, adapter_note: Optional[str] = None) -> None:
    if adapter_note:
        console.print(f"[yellow]• {adapter_note}[/yellow]")
    t = Table(title=f"Modal vLLM plan — {plan.model_id}", show_header=False)
    t.add_row("GPU", plan.gpu_str)
    t.add_row("tensor-parallel", str(plan.tensor_parallel_size))
    t.add_row("max-model-len", str(plan.max_model_len))
    t.add_row("dtype", plan.dtype)
    if plan.spec:
        t.add_row("params", f"{plan.spec.n_params / 1e9:.1f}B")
        t.add_row("weights", f"{plan.spec.weight_gib:.0f} GiB")
    console.print(t)
    for n in plan.notes:
        console.print(f"  [dim]• {n}[/dim]")
    console.print(f"\n[bold]vllm cmd:[/bold] [dim]{serve_cmd}[/dim]")


@app.command()
def plan(
    model: str,
    gpu: Optional[str] = typer.Option(None, help="Force GPU type (H100, A100-80GB, H200, ...)."),
    gpu_count: Optional[int] = typer.Option(None, help="Force tensor-parallel GPU count."),
    max_model_len: Optional[int] = typer.Option(None, help="Context length (default: model max, capped 8192)."),
    lora: Optional[list[str]] = typer.Option(None, help="Adapter name=path (repeatable)."),
    max_lora_rank: int = typer.Option(64),
    tokenizer: Optional[str] = typer.Option(None),
    served_model_name: Optional[str] = typer.Option(None),
):
    """Show the auto-sized deployment plan and the vllm command — no deploy."""
    s = ModalVLLM(model, lora_modules=_parse_loras(lora), gpu_type=gpu, gpu_count=gpu_count,
                  max_model_len=max_model_len, max_lora_rank=max_lora_rank, tokenizer=tokenizer,
                  served_model_name=served_model_name)
    _show_plan(s.plan, s.serve_cmd, s.adapter_note)


@app.command()
def serve(
    model: str,
    gpu: Optional[str] = typer.Option(None, help="Force GPU type."),
    gpu_count: Optional[int] = typer.Option(None, help="Force tensor-parallel GPU count."),
    max_model_len: Optional[int] = typer.Option(None),
    lora: Optional[list[str]] = typer.Option(None, help="Adapter name=path (repeatable)."),
    max_lora_rank: int = typer.Option(64),
    tokenizer: Optional[str] = typer.Option(None),
    served_model_name: Optional[str] = typer.Option(None),
    enforce_eager: Optional[bool] = typer.Option(
        None, "--enforce-eager/--no-enforce-eager",
        help="Skip CUDA-graph capture for faster startup (default: auto-on for multi-GPU)."),
    name: Optional[str] = typer.Option(None, "--name", help="Modal app name (default from model id)."),
    min_containers: int = typer.Option(1, help="Warm replicas (1 = no cold start while iterating)."),
    max_containers: int = typer.Option(1, help="Autoscaling ceiling (raise for parallel sweeps)."),
    vllm_version: str = typer.Option("0.21.0"),
    wait: bool = typer.Option(True, help="Block until /v1/models is live."),
):
    """Deploy the model on Modal and print the OpenAI-compatible endpoint URL."""
    server = ModalVLLM(
        model, lora_modules=_parse_loras(lora), gpu_type=gpu, gpu_count=gpu_count,
        max_model_len=max_model_len, max_lora_rank=max_lora_rank, tokenizer=tokenizer,
        served_model_name=served_model_name, enforce_eager=enforce_eager, app_name=name,
        min_containers=min_containers, max_containers=max_containers, vllm_version=vllm_version,
    )
    _show_plan(server.plan, server.serve_cmd, server.adapter_note)
    console.print(f"\n[bold]Deploying[/bold] app [cyan]{server.app_name}[/cyan] ...")
    url = server.start(wait=wait)
    console.print(f"\n[bold green]Endpoint:[/bold green] {url}")
    console.print(f"[bold green]OpenAI base_url:[/bold green] {url}/v1")
    console.print(f"[dim]stop with:[/dim] vllm-model stop {server.app_name}")


@app.command()
def stop(app_name: str):
    """Stop (tear down) a deployed vllm-model app."""
    import subprocess
    raise SystemExit(subprocess.run(["modal", "app", "stop", app_name]).returncode)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
