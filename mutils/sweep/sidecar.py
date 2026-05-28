"""Per-container GPU utilisation sidecar.

Runs in a daemon thread inside the Modal container; emits structured stdout
lines that the local TUI parses out of the live log stream:

    [gpu] cell=<hash> util=<int> mem_used=<MiB> mem_total=<MiB>
    [status] cell=<hash> status=<warming_up|running|done|failed> [meta...]

The user's experiment fn never imports this module; it's wired in by the
runner's wrapper.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading

GPU_LINE = "[gpu]"
STATUS_LINE = "[status]"


def _emit(line: str) -> None:
    """Write a structured line to stdout, flushed."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_status(cell_hash: str, status: str, **meta: object) -> None:
    """Tell the TUI about a state change. Cheap; call freely."""
    extras = " ".join(f"{k}={v}" for k, v in meta.items())
    _emit(f"{STATUS_LINE} cell={cell_hash} status={status} {extras}".rstrip())


def _nvidia_smi_once() -> tuple[int, int, int] | None:
    """Return (util%, mem_used_MiB, mem_total_MiB) for GPU 0 or None on failure."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def start(cell_hash: str, interval: float = 2.0) -> threading.Event:
    """Spawn a daemon thread that prints GPU util every `interval` seconds.

    Returns a threading.Event you can `set()` to stop the loop early. The
    daemon flag means the thread dies with the process if you don't.
    """
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.is_set():
            stats = _nvidia_smi_once()
            if stats is not None:
                util, mem_used, mem_total = stats
                _emit(f"{GPU_LINE} cell={cell_hash} util={util} mem_used={mem_used} mem_total={mem_total}")
            stop_event.wait(interval)

    threading.Thread(target=_loop, daemon=True).start()
    return stop_event
