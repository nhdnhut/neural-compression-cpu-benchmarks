"""
profiling/timer.py
==================
Wall-clock inference latency measurement.

Design decisions:
  - Uses time.perf_counter() — highest resolution available on all platforms.
  - Warmup runs discard JIT/cache cold-start effects.
  - Reports mean, std, p50, p95 — gives both average and tail latency.
  - Single-sample (batch=1) measures real-world inference latency.
  - Larger batch measures throughput (samples/second).

IMPORTANT: On CPU, results can have high variance due to OS scheduling,
thermal throttling, and background processes. Run experiments on an idle
machine and report mean +/- std across multiple runs.
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 10,
    measurement_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Measure inference latency with warmup.

    Args:
        model:            PyTorch model (must be in eval() mode)
        input_tensor:     sample input tensor
        warmup_runs:      number of warmup runs (discarded)
        measurement_runs: number of measured runs
        device:           'cpu' or 'cuda'

    Returns:
        dict with keys:
            mean_ms, std_ms, p50_ms, p95_ms, min_ms, max_ms
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup: fill caches, allow CPU frequency scaling to stabilise
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Measure
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(measurement_runs):
            t_start = time.perf_counter()
            _ = model(input_tensor)
            t_end = time.perf_counter()
            times_ms.append((t_end - t_start) * 1_000)  # -> ms

    arr = np.array(times_ms)
    return {
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "min_ms":    float(arr.min()),
        "max_ms":    float(arr.max()),
        "warmup_runs":     warmup_runs,
        "measurement_runs": measurement_runs,
    }


def measure_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 5,
    measurement_runs: int = 20,
    device: str = "cpu",
) -> dict:
    """
    Measure throughput in samples/second.

    Args:
        input_tensor: batched input, shape (batch_size, ...)

    Returns:
        dict with keys: samples_per_sec, batch_size, mean_batch_ms
    """
    model.eval()
    batch_size = input_tensor.shape[0]
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(measurement_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            times_ms.append((time.perf_counter() - t0) * 1_000)

    mean_batch_ms = np.mean(times_ms)
    return {
        "samples_per_sec": float(batch_size / (mean_batch_ms / 1_000)),
        "batch_size":      batch_size,
        "mean_batch_ms":   float(mean_batch_ms),
    }


def measure_training_time(
    train_fn: Callable,
) -> tuple:
    """
    Measure total wall-clock time of a training function.

    Usage:
        elapsed_sec, result = measure_training_time(lambda: train(...))

    Returns:
        (elapsed_seconds, function_return_value)
    """
    t0 = time.perf_counter()
    result = train_fn()
    elapsed = time.perf_counter() - t0
    return elapsed, result
