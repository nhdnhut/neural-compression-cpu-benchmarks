"""
profiling/memory.py
===================
Peak RAM measurement during model inference and training.

Uses Python's tracemalloc for pure-Python heap allocations.
NOTE: tracemalloc only tracks Python-level allocations, NOT C extensions
like numpy/torch tensor storage. For a more complete picture, we also
report process RSS via psutil if available.

This distinction is important to understand: tracemalloc will under-report
actual memory for torch models because tensor storage is in C. We report
both and document the limitation.
"""

from __future__ import annotations
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Any


def measure_peak_ram_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    runs: int = 10,
) -> dict:
    """
    Measure peak RAM during multiple inference passes.

    Uses tracemalloc for Python-heap tracking. Also attempts psutil RSS.

    Returns:
        dict with keys: peak_tracemalloc_mb, current_tracemalloc_mb,
                        model_params_mb, psutil_rss_mb (if available)
    """
    model.eval()

    # tracemalloc peak
    tracemalloc.start()
    with torch.no_grad():
        for _ in range(runs):
            _out = model(input_tensor)
    del _out  # release last activation tensor before reading peak
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "peak_tracemalloc_mb":    peak    / 1024 / 1024,
        "current_tracemalloc_mb": current / 1024 / 1024,
        "model_params_mb":        _count_param_bytes(model) / 1024 / 1024,
    }

    # psutil RSS (optional, more accurate for native memory)
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        result["psutil_rss_mb"] = proc.memory_info().rss / 1024 / 1024
    except ImportError:
        result["psutil_rss_mb"] = None  # psutil not installed

    return result


def measure_peak_ram_fn(fn: Callable[[], Any]) -> tuple[Any, dict]:
    """
    Measure peak RAM of an arbitrary function call.

    Returns:
        (function_result, memory_dict)
    """
    tracemalloc.start()
    result = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem = {
        "peak_tracemalloc_mb":    peak    / 1024 / 1024,
        "current_tracemalloc_mb": current / 1024 / 1024,
    }
    return result, mem


def _count_param_bytes(model: nn.Module) -> int:
    """Count bytes occupied by all parameter tensors."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()  # element_size: 4 for fp32, 1 for int8
    return total


def model_size_mb(model: nn.Module) -> float:
    """
    Compute model size as total bytes of all parameter tensors.
    This is the minimum serialised size (dense storage).
    Actual .pth file may be larger due to metadata.
    """
    return _count_param_bytes(model) / 1024 / 1024


def model_size_on_disk_mb(filepath: str) -> float:
    """Return size of a saved .pth file in MB."""
    import os
    return os.path.getsize(filepath) / 1024 / 1024
