"""
compression/quantization.py
============================
Post-Training Quantization (PTQ) for PyTorch models.

Two approaches are supported:
  1. Dynamic PTQ  — Quantizes weights only; activations quantized at runtime.
                   No calibration data needed. Simpler. Works with Linear layers.
  2. Static PTQ   — Quantizes both weights and activations using calibration data.
                   Better accuracy/size trade-off. Requires representative data.

Why INT8 actually speeds up CPU inference:
  x86 CPUs with AVX2/AVX-512 have SIMD instructions that process
  multiple INT8 operations per clock cycle. PyTorch's FBGEMM backend
  exploits these instructions. INT8 also reduces memory bandwidth by 4x
  (4 bytes/param -> 1 byte/param), which matters for memory-bound ops.

This is one of few compression methods where theoretical savings
(4x size reduction) translates to real hardware speedup.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Optional, Iterator


# ---------------------------------------------------------------------------
# Dynamic PTQ (recommended starting point)
# ---------------------------------------------------------------------------

def apply_dynamic_ptq(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    copy_model: bool = True,
) -> nn.Module:
    """
    Apply dynamic post-training quantization.

    Quantizes weights of Linear layers to INT8. Activations are
    quantized dynamically at inference time (no calibration needed).

    Args:
        model:      trained PyTorch model (CPU)
        dtype:      quantization dtype (torch.qint8 recommended)
        copy_model: if True, work on a deep copy (preserves original)

    Returns:
        Quantized model (CPU only — dynamic PTQ not available on CUDA)

    Note:
        Conv2d is not supported by torch.quantization.quantize_dynamic.
        For CNN models, this quantizes the FC (Linear) layers only.
        This is expected behaviour and will be noted in results.
    """
    if copy_model:
        model = copy.deepcopy(model)

    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},   # only Linear layers
        dtype=dtype,
    )
    return quantized


# ---------------------------------------------------------------------------
# Static PTQ (better accuracy, requires calibration)
# ---------------------------------------------------------------------------

def apply_static_ptq(
    model: nn.Module,
    calibration_loader: Iterator,
    n_calibration_batches: int = 10,
    backend: str = "fbgemm",   # 'fbgemm' for x86 CPU, 'qnnpack' for ARM
    copy_model: bool = True,
) -> nn.Module:
    """
    Apply static post-training quantization with calibration.

    Steps:
      1. Insert observer modules (prepare)
      2. Run calibration data through model -> observers collect activation stats
      3. Convert: replace float ops with INT8 quantized equivalents

    Args:
        model:                   trained PyTorch model
        calibration_loader:      DataLoader or iterable of (data, label) batches
        n_calibration_batches:   how many batches to use for calibration
        backend:                 'fbgemm' (x86) or 'qnnpack' (ARM/mobile)
        copy_model:              deepcopy before modifying

    Returns:
        Quantized model

    Note:
        Static PTQ requires models to be written with QuantStub/DeQuantStub
        or use fuse_modules for Conv+BN+ReLU fusion. For our SmallCNN,
        dynamic PTQ is the practical choice. Static PTQ is provided for
        research completeness and future extension.
    """
    if copy_model:
        model = copy.deepcopy(model)

    model.eval()

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Apply default qconfig
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Prepare: insert observer modules
    torch.quantization.prepare(model, inplace=True)

    # Calibration pass: run representative data through the model
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            model(data)
            if i + 1 >= n_calibration_batches:
                break

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_quantized_model_size_mb(model: nn.Module) -> float:
    """
    Estimate size of quantized model by summing element_size() of each param.
    INT8 params have element_size() = 1 byte (vs 4 bytes for FP32).
    """
    total_bytes = 0
    for _, module in model.named_modules():
        # Quantized Linear layers store packed weights differently
        # We approximate by counting data pointers
        for name, param in module.named_parameters(recurse=False):
            if hasattr(param, "element_size"):
                total_bytes += param.numel() * param.element_size()
    return total_bytes / 1024 / 1024
