"""
profiling/flops.py
==================
FLOPs and parameter counting for PyTorch models.

Uses the `thop` library for automated FLOPs counting.
Also provides manual parameter count and sparsity analysis.

Key distinction:
  - MACs: multiply-accumulate operations (one mul + one add = 1 MAC)
  - FLOPs: typically approximated as 2 x MACs (one mul + one add = 2 FLOPs)
  - This project uses MACs as the primary theoretical cost metric,
    with FLOPs = 2 x MACs reported for compatibility with other papers.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


def count_flops(
    model: nn.Module,
    input_shape: tuple = (1, 1, 28, 28),
    verbose: bool = False,
) -> dict:
    """
    Count MACs and FLOPs for one forward pass.

    Args:
        model:       PyTorch model in eval() mode
        input_shape: (batch=1, C, H, W) for CNN or (batch=1, features) for MLP
        verbose:     print thop layer-by-layer breakdown

    Returns:
        dict with keys: MACs, FLOPs, params_thop
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError("thop is required: pip install thop")

    dummy_input = torch.zeros(*input_shape)
    model.eval()

    macs, params = profile(model, inputs=(dummy_input,), verbose=verbose)

    return {
        "MACs":        int(macs),
        "FLOPs":       int(2 * macs),     # conventional approximation
        "MACs_M":      macs / 1e6,        # in millions, for readability
        "FLOPs_M":     2 * macs / 1e6,
        "params_thop": int(params),
    }


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters.

    Returns:
        dict with keys: trainable, total, non_trainable
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {
        "trainable":     trainable,
        "total":         total,
        "non_trainable": total - trainable,
    }


def compute_sparsity(model: nn.Module) -> dict:
    """
    Compute weight sparsity (fraction of zeros) across all weight tensors.

    NOTE: Filters to `.weight` attributes on Linear/Conv2d only — avoids
    double-counting `weight_orig` and `weight_mask` buffers left by pruning hooks.

    Returns:
        dict with per-layer sparsity and global sparsity
    """
    layer_stats = {}
    total_params = 0
    total_zeros  = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        param = module.weight
        n = param.numel()
        z = (param.data == 0).sum().item()
        layer_stats[name] = {
            "params":   n,
            "zeros":    int(z),
            "sparsity": z / n,
        }
        total_params += n
        total_zeros  += z

    return {
        "per_layer":       layer_stats,
        "global_sparsity": total_zeros / total_params if total_params > 0 else 0.0,
        "total_weights":   total_params,
        "total_zeros":     int(total_zeros),
    }
