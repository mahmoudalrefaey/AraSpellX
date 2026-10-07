"""Shared training utilities: progress display, logging, precision and batch prefetching."""
from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from tqdm import tqdm


class Progress:
    """A live progress bar plus timestamped messages, mirrored to a log file.

    The bar shows step, ETA and the latest metrics; messages (evaluations,
    checkpoints) are printed above it without breaking it, and every message
    and periodic metric line is appended to `log_path` for later review.
    """

    def __init__(self, total: int, start: int, log_path: Path, description: str) -> None:
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.bar = tqdm(total=total, initial=start, desc=description, unit="step",
                        dynamic_ncols=True, smoothing=0.05)
        self.started = time.time()

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        tqdm.write(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def step(self, metrics: Dict[str, float], write_every: int, step: int) -> None:
        self.bar.update(1)
        if step % 10 == 0:
            self.bar.set_postfix({k: _format(v) for k, v in metrics.items()}, refresh=False)
        if step % write_every == 0:
            elapsed = (time.time() - self.started) / 3600
            fields = "  ".join(f"{k} {_format(v)}" for k, v in metrics.items())
            self.log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] step {step}  {fields}  "
                                f"elapsed {elapsed:.2f} h\n")
            self.log_file.flush()

    def close(self) -> None:
        self.bar.close()
        self.log_file.close()


def _format(value: float) -> str:
    if isinstance(value, str):
        return value
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) < 0.01:
        return f"{value:.2e}"
    return f"{value:.3f}"


def precision_settings(choice: str):
    """(autocast context factory, GradScaler or None, name) for 'auto', 'bf16', 'fp16' or 'fp32'.

    'auto' uses bf16 on GPUs that support it (Ampere and newer: RTX 30xx, A100,
    A10, H100) and fp16 with loss scaling otherwise (T4, V100).
    """
    if choice == "auto":
        choice = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "fp32":
        return nullcontext, None, choice
    dtype = torch.bfloat16 if choice == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if choice == "fp16" else None
    return (lambda: torch.autocast("cuda", dtype=dtype)), scaler, choice


class Prefetcher:
    """Prepares the next batches in a background thread while the GPU trains."""

    def __init__(self, make_batch: Callable[[], object], depth: int = 4) -> None:
        self.queue: "queue.Queue" = queue.Queue(maxsize=depth)
        self.make_batch = make_batch
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                self.queue.put(self.make_batch())
        except BaseException as e:  # surfaced in next()
            self.error = e
            self.queue.put(None)

    def next(self):
        item = self.queue.get()
        if item is None and self.error is not None:
            raise self.error
        return item


def gpu_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9
