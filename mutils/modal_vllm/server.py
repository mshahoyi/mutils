"""Provision a vLLM OpenAI-compatible server on Modal from a model id.

`ModalVLLM` is the programmatic entry point — the Modal analogue of running
`vllm serve` locally, except it sizes and provisions GPUs in the cloud and hands
back an HTTPS endpoint:

    from mutils.modal_vllm import ModalVLLM

    server = ModalVLLM("meta-llama/Llama-3.3-70B-Instruct",
                       lora_modules={"my_adapter": "org/my-lora"})
    url = server.start()                 # deploys, waits until /v1/models is live
    # ... hit f"{url}/v1/chat/completions" with model="my_adapter" ...
    server.stop()                         # scale to zero / tear down

or as a context manager (ephemeral — stops on exit):

    with ModalVLLM("Qwen/Qwen2.5-7B-Instruct") as url:
        ...

Everything is auto-sized from the model (see `planner.plan_deployment`) but every
field is overridable via the constructor.

Modal specifics: the serve function is registered with `serialized=True` so the
container image does NOT need `mutils` installed — the (tiny, stdlib-only) function
is pickled and shipped. The actual `vllm serve ...` command line is baked into the
image env as `MUTILS_VLLM_CMD`, so local and container see identical config.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .planner import DeploymentPlan, fetch_adapter_info, plan_deployment

VLLM_PORT = 8000
DEFAULT_VLLM_VERSION = "0.21.0"
DEFAULT_CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
_MINUTES = 60


def build_vllm_command(
    plan: DeploymentPlan,
    *,
    lora_modules: Optional[dict[str, str]] = None,
    max_lora_rank: int = 64,
    tokenizer: Optional[str] = None,
    served_model_name: Optional[str] = None,
    enforce_eager: bool = False,
    max_loras: Optional[int] = None,
    max_cpu_loras: Optional[int] = None,
    extra_args: Optional[list[str]] = None,
) -> str:
    """Render the `vllm serve ...` command line from a plan + serving options."""
    cmd = [
        "vllm", "serve", plan.model_id,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(plan.tensor_parallel_size),
        "--max-model-len", str(plan.max_model_len),
        "--gpu-memory-utilization", str(plan.gpu_memory_utilization),
    ]
    if enforce_eager:
        # Skip CUDA-graph capture + torch.compile. For big/multi-GPU models this
        # cuts startup from tens of minutes to a few; the per-token decode cost is
        # negligible for low-QPS/bursty workloads (e.g. auditing agents).
        cmd += ["--enforce-eager"]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if lora_modules:
        n = len(lora_modules)
        # --max-loras = adapters active in a single batch (GPU slots); --max-cpu-loras
        # = adapters cached in CPU RAM. When serving many adapters (e.g. all quirks),
        # cap GPU slots but cache them all so vLLM swaps instead of erroring.
        gpu_slots = max_loras or n
        cmd += ["--enable-lora", "--max-lora-rank", str(max_lora_rank),
                "--max-loras", str(gpu_slots)]
        if max_cpu_loras or n > gpu_slots:
            cmd += ["--max-cpu-loras", str(max_cpu_loras or n)]
        cmd += ["--lora-modules"]
        cmd += [f"{name}={path}" for name, path in lora_modules.items()]
    if extra_args:
        cmd += list(extra_args)
    return " ".join(cmd)


def build_app(
    plan: DeploymentPlan,
    serve_cmd: str,
    *,
    app_name: str,
    lora_modules: Optional[dict[str, str]] = None,
    hf_token: Optional[str] = None,
    vllm_version: str = DEFAULT_VLLM_VERSION,
    cuda_image: str = DEFAULT_CUDA_IMAGE,
    scaledown_window: int = 15 * _MINUTES,
    startup_timeout: int = 20 * _MINUTES,
    max_concurrent: int = 64,
    min_containers: int = 0,
    max_containers: int = 1,
):
    """Construct (but do not deploy) the Modal App + serve Function for a plan.

    Returns (app, serve_function). Building objects is offline/free; deploying
    (App.deploy) is what provisions GPUs.
    """
    import json

    import modal

    app = modal.App(app_name)

    image = (
        modal.Image.from_registry(cuda_image, add_python="3.12")
        .entrypoint([])
        .uv_pip_install(f"vllm=={vllm_version}", "huggingface_hub[hf_transfer]")
        .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "MUTILS_VLLM_CMD": serve_cmd,
            "MUTILS_VLLM_LORAS": json.dumps(lora_modules or {}),
        })
    )

    volumes = {
        "/root/.cache/huggingface": modal.Volume.from_name("mutils-vllm-hf-cache", create_if_missing=True),
        "/root/.cache/vllm": modal.Volume.from_name("mutils-vllm-compile-cache", create_if_missing=True),
    }
    secrets = []
    if hf_token:
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}))

    # Defined locally on purpose: with serialized=True, cloudpickle ships a
    # locally-scoped function BY VALUE (embedding its code), so the container
    # needs nothing from `mutils`. A module-level function would instead pickle
    # BY REFERENCE and fail to deserialize remotely (no mutils in the image).
    def serve():
        import json
        import os
        import subprocess

        cmd = os.environ["MUTILS_VLLM_CMD"]
        # Pre-download LoRA adapters ONCE to local dirs and rewrite the command to
        # use those paths. Otherwise vLLM fetches each adapter lazily at add_lora —
        # and all TP workers hit HF concurrently, which triggers connection resets
        # ("No adapter found"). Local paths also make restarts instant (volume-cached).
        loras = json.loads(os.environ.get("MUTILS_VLLM_LORAS", "{}"))
        if loras:
            from huggingface_hub import snapshot_download
            for name, repo in loras.items():
                if os.path.isdir(repo):
                    continue
                local = snapshot_download(repo)
                print(f"[mutils-modal-vllm] pre-downloaded LoRA {name}: {repo} -> {local}", flush=True)
                cmd = cmd.replace(f"{name}={repo}", f"{name}={local}")
        print(f"[mutils-modal-vllm] launching: {cmd}", flush=True)
        subprocess.Popen(cmd, shell=True)

    fn = modal.web_server(port=VLLM_PORT, startup_timeout=startup_timeout)(serve)
    fn = modal.concurrent(max_inputs=max_concurrent)(fn)
    serve = app.function(
        image=image,
        gpu=plan.gpu_str,
        volumes=volumes,
        secrets=secrets,
        scaledown_window=scaledown_window,
        timeout=24 * 60 * _MINUTES,
        min_containers=min_containers,
        max_containers=max_containers,
        serialized=True,
    )(fn)
    return app, serve


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")[:40]


class ModalVLLM:
    """Provision and manage a vLLM OpenAI endpoint on Modal for a model id.

    Args:
        model: HF model id to serve (the base weights).
        lora_modules: optional {served_name: hf_repo_or_path} adapters. Clients
            then request `model=<served_name>`.
        gpu_type / gpu_count: override the auto-sized GPU plan (see planner).
        max_model_len / gpu_memory_utilization: vLLM runtime knobs.
        max_lora_rank: LoRA rank cap (>= the largest adapter's r).
        tokenizer / served_model_name / extra_vllm_args: passthrough to vllm serve.
        hf_token: token for gated models (else read HF_TOKEN/HUGGING_FACE_HUB_TOKEN env).
        app_name: Modal app name (default derived from the model id).
        min_containers: keep N replicas warm (1 = no cold starts during iteration).
        max_containers: autoscaling ceiling (raise for the parallel phase).
        vllm_version: pinned vllm to install in the image.
    """

    def __init__(
        self,
        model: str,
        *,
        lora_modules: Optional[dict[str, str]] = None,
        gpu_type: Optional[str] = None,
        gpu_count: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_lora_rank: int = 64,
        tokenizer: Optional[str] = None,
        served_model_name: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        max_loras: Optional[int] = None,
        max_cpu_loras: Optional[int] = None,
        extra_vllm_args: Optional[list[str]] = None,
        hf_token: Optional[str] = None,
        app_name: Optional[str] = None,
        min_containers: int = 1,
        max_containers: int = 1,
        startup_timeout: int = 20 * 60,
        vllm_version: str = DEFAULT_VLLM_VERSION,
    ):
        import os

        self.requested_model = model
        self.lora_modules = dict(lora_modules or {})
        self.max_lora_rank = max_lora_rank
        self.tokenizer = tokenizer
        self.served_model_name = served_model_name
        self.extra_vllm_args = extra_vllm_args
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.app_name = app_name or f"vllm-{_slug(model)}"
        self.adapter_note: Optional[str] = None

        # If handed a LoRA adapter repo, serve its base + attach the adapter.
        adapter = fetch_adapter_info(model, self.hf_token)
        if adapter:
            served = _slug(model)
            self.model = adapter["base_model"]
            self.lora_modules = {served: model, **self.lora_modules}
            self.tokenizer = tokenizer or model  # adapter repo carries the right tokenizer/chat template
            if adapter.get("r"):
                self.max_lora_rank = max(self.max_lora_rank, int(adapter["r"]))
            self.adapter_note = (
                f"{model} is a {adapter.get('peft_type') or 'LoRA'} adapter → serving base "
                f"{self.model} with it attached as model='{served}' (rank {self.max_lora_rank})"
            )
        else:
            self.model = model
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.startup_timeout = startup_timeout
        self.vllm_version = vllm_version

        self.plan: DeploymentPlan = plan_deployment(
            self.model, token=self.hf_token, gpu_type=gpu_type, gpu_count=gpu_count,
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
        )
        # Default: skip CUDA-graph capture for multi-GPU (large) models, where
        # capture dominates startup; keep it for small single-GPU models.
        self.enforce_eager = enforce_eager if enforce_eager is not None else (self.plan.gpu_count > 1)
        self.serve_cmd = build_vllm_command(
            self.plan, lora_modules=self.lora_modules, max_lora_rank=self.max_lora_rank,
            tokenizer=self.tokenizer, served_model_name=self.served_model_name,
            enforce_eager=self.enforce_eager, max_loras=max_loras, max_cpu_loras=max_cpu_loras,
            extra_args=self.extra_vllm_args,
        )
        self._app = None
        self._serve = None
        self._app_id: Optional[str] = None
        self._url: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def deploy(self) -> str:
        """Deploy the app (provisions GPUs lazily on first request) and return the URL."""
        self._app, self._serve = build_app(
            self.plan, self.serve_cmd, app_name=self.app_name, lora_modules=self.lora_modules,
            hf_token=self.hf_token, vllm_version=self.vllm_version,
            min_containers=self.min_containers, max_containers=self.max_containers,
            startup_timeout=self.startup_timeout,
        )
        self._app.deploy()
        self._app_id = getattr(self._app, "app_id", None)
        self._url = self._web_url()
        return self._url

    def start(self, wait: bool = True, timeout: int = 30 * _MINUTES) -> str:
        """Deploy and (by default) block until `/v1/models` responds. Returns base URL."""
        url = self.deploy()
        if wait and not self.wait_until_ready(timeout=timeout):
            raise TimeoutError(f"vLLM did not become ready within {timeout}s at {url}")
        return url

    def _web_url(self) -> str:
        get = getattr(self._serve, "get_web_url", None)
        url = get() if callable(get) else getattr(self._serve, "web_url", None)
        if not url:
            raise RuntimeError("could not resolve the deployed web URL from Modal")
        return url.rstrip("/")

    @property
    def url(self) -> Optional[str]:
        """Base URL (no trailing slash). Append /v1/chat/completions etc."""
        return self._url

    @property
    def base_url(self) -> Optional[str]:
        """OpenAI base_url form: <url>/v1."""
        return f"{self._url}/v1" if self._url else None

    def wait_until_ready(self, timeout: int = 30 * _MINUTES, interval: int = 15) -> bool:
        """Block until `/v1/models` answers. Fails fast (raises) if the container
        crash-loops instead of burning the whole timeout — a plain HTTP poll can't
        tell "still loading" from "crashed", so we also scan the Modal logs.
        """
        if not self._url:
            raise RuntimeError("call deploy()/start() first")
        models_url = f"{self._url}/v1/models"
        t0 = last_log_check = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(models_url, timeout=15) as r:
                    if r.status == 200:
                        served = [m.get("id") for m in json.load(r).get("data", [])]
                        print(f"[mutils-modal-vllm] ready — served: {served}", flush=True)
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                pass
            if time.time() - last_log_check > 30:
                last_log_check = time.time()
                hit, excerpt = self._scan_logs_for_crash()
                if hit:
                    raise RuntimeError(
                        f"vLLM container crashed during startup (matched {hit!r}). "
                        f"Recent Modal logs:\n{excerpt}\n"
                        f"Full logs: modal app logs {self._app_id or self.app_name}"
                    )
            time.sleep(interval)
        return False

    # markers that mean the container is dead, not merely slow to load
    _CRASH_MARKERS = (
        "Runner failed with exit code",
        "DeserializationError",
        "ModuleNotFoundError",
        "ImportError",
        "torch.cuda.OutOfMemoryError",
        "CUDA out of memory",
        "Engine core initialization failed",
        "EngineDeadError",
        "ValueError: No supported config",
        "does not exist",
        "Cannot find any model weights",
        "No adapter found",
        "LoRAAdapterNotFoundError",
        # Modal web-server gave up waiting for the port → container will restart-loop.
        # Common when startup is too slow (e.g. CUDA-graph capture) — fail fast and
        # surface it rather than waiting out the timeout.
        "Waited too long for port",
        "Runner failed with exception",
    )

    def _recent_logs(self, read_seconds: float = 8.0) -> str:
        """Best-effort dump of recent Modal logs for this app. `modal app logs`
        prints history then tails; we read for a few seconds and return what came."""
        import subprocess

        target = self._app_id or self.app_name
        if not target:
            return ""
        try:
            p = subprocess.Popen(["modal", "app", "logs", target],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception:
            return ""
        out: list[str] = []
        t0 = time.time()
        try:
            while time.time() - t0 < read_seconds:
                line = p.stdout.readline()
                if not line:
                    break
                out.append(line)
        finally:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        return "".join(out)

    def _scan_logs_for_crash(self) -> tuple[Optional[str], str]:
        """Return (matched_marker, log_excerpt) if logs show a fatal crash, else (None, '')."""
        logs = self._recent_logs()
        for marker in self._CRASH_MARKERS:
            if marker in logs:
                return marker, "\n".join(logs.splitlines()[-25:])
        return None, ""

    def stop(self) -> None:
        """Stop the deployed app (scales to zero and removes the deployment)."""
        import subprocess
        target = self._app_id or self.app_name
        subprocess.run(["modal", "app", "stop", target, "--yes"], check=False)

    # -- context manager (ephemeral) --------------------------------------

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
