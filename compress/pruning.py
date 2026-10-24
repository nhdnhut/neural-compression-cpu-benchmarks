"""
compression/pruning.py
======================
Magnitude-based unstructured pruning for PyTorch models.

Method: Global L1-unstructured pruning (Han et al., NIPS 2015).
  1. Train model to convergence (done externally).
  2. Sort ALL weight parameters globally by |w|.
  3. Zero out the bottom k% (by magnitude).
  4. Optionally fine-tune the sparse model.

IMPORTANT NOTE on speedup:
  Unstructured sparsity does NOT automatically speed up inference
  on standard hardware. Zeroed weights still occupy memory in dense
  tensors. Real latency reduction requires structured pruning (remove
  entire channels/filters) or sparse BLAS kernels.
  
  This project intentionally uses unstructured pruning and reports
  this distinction as part of the FLOPs-vs-latency analysis (RQ2).
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional


def apply_magnitude_pruning(
    model: nn.Module,
    pruning_ratio: float,
    make_permanent: bool = True,
) -> dict:
    """
    Apply global unstructured magnitude pruning.

    Args:
        model:          PyTorch model (modified in-place)
        pruning_ratio:  fraction of weights to zero (0.0 = none, 0.9 = 90%)
        make_permanent: if True, remove mask buffers (saves memory, can't undo)

    Returns:
        sparsity_stats dict
    """
    assert 0.0 <= pruning_ratio < 1.0, \
        f"pruning_ratio must be in [0, 1), got {pruning_ratio}"

    if pruning_ratio == 0.0:
        return _compute_sparsity_stats(model)

    # Collect all weight parameters across Linear and Conv2d layers
    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]

    if not parameters_to_prune:
        raise RuntimeError("No Linear or Conv2d layers found for pruning.")

    # Apply global L1 unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )

    if make_permanent:
        # Remove mask buffers: weight_mask and weight_orig -> merged into weight
        for module, name in parameters_to_prune:
            prune.remove(module, name)

    return _compute_sparsity_stats(model)


def make_pruning_permanent(model: nn.Module) -> None:
    """
    Remove pruning mask buffers and permanently zero the weights.
    MUST be called AFTER fine-tuning so that forward-hook masks
    keep pruned weights zero during optimization.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if prune.is_pruned(module):
                prune.remove(module, "weight")


def _compute_sparsity_stats(model: nn.Module) -> dict:
    """Compute per-layer and global sparsity statistics."""
    layer_stats = {}
    total_w, total_z = 0, 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight
            n = w.numel()
            z = (w == 0).sum().item()
            layer_stats[name] = {
                "params":   n,
                "zeros":    int(z),
                "sparsity": z / n if n > 0 else 0.0,
            }
            total_w += n
            total_z += z

    return {
        "per_layer":       layer_stats,
        "global_sparsity": total_z / total_w if total_w > 0 else 0.0,
        "total_weights":   total_w,
        "total_zeros":     total_z,
    }


def prune_and_finetune(
    model: nn.Module,
    pruning_ratio: float,
    train_fn,
    finetune_epochs: int = 5,
    copy_model: bool = True,
) -> tuple[nn.Module, dict, dict]:
    """
    High-level: prune model then fine-tune.
    Crucially keeps masks active during fine-tuning, then makes permanent.

    Args:
        model:           trained baseline model
        pruning_ratio:   fraction of weights to zero
        train_fn:        callable(model, epochs) -> history dict
        finetune_epochs: number of epochs for fine-tuning
        copy_model:      if True, deepcopy model before pruning

    Returns:
        (pruned_model, sparsity_stats, finetune_history)
    """
    if copy_model:
        model = copy.deepcopy(model)

    if pruning_ratio > 0:
        # Keep masks active during fine-tuning (do NOT make permanent yet)
        apply_magnitude_pruning(model, pruning_ratio, make_permanent=False)

        if finetune_epochs > 0:
            history = train_fn(model, finetune_epochs)
        else:
            history = {}

        # Make permanent AFTER fine-tuning is done
        make_pruning_permanent(model)
        sparsity_stats = _compute_sparsity_stats(model)
    else:
        history = {}
        sparsity_stats = _compute_sparsity_stats(model)

    return model, sparsity_stats, history
