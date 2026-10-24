"""
compression/low_rank.py
=======================
SVD-based low-rank approximation for Linear layer weight matrices.

Method: Truncated SVD (economy SVD)
  W (mxn) ~= U_r (mxr) @ diag(S_r) @ V^T_r (rxn)
  where r < min(m, n)

  Storage reduction: m*n -> m*r + r + r*n  (for the three matrices)
  This project stores the approximation as the reconstructed W (same shape),
  so memory is NOT reduced by default. To reduce memory, replace the layer
  with two consecutive Linear layers: x -> (x @ V_r^T) -> (result @ (U_r @ S_r)^T)
  
  This distinction (reconstruction vs. factorisation) is documented
  explicitly — it is relevant to understanding why low-rank may not
  reduce inference latency without architectural changes.

Reference: Denton et al., NIPS 2014 — "Exploiting Linear Structure
           Within Convolutional Networks for Efficient Evaluation"
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def apply_svd_approximation(
    model: nn.Module,
    rank_ratio: float,
    target_layers: Optional[list[str]] = None,
    copy_model: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Apply SVD low-rank approximation to Linear layers.

    Reconstructs W ~= U_r @ diag(S_r) @ V_r^T and writes back to the
    layer's weight parameter. This preserves the layer architecture.

    Args:
        model:         trained PyTorch model
        rank_ratio:    fraction of singular values to keep (1.0 = no change)
        target_layers: list of layer names to compress (None = all Linear)
        copy_model:    deepcopy model before modifying

    Returns:
        (modified_model, approximation_stats)

    Stats per layer:
        rank_r:        number of singular values kept
        rank_full:     min(m, n) — maximum possible rank
        rank_ratio:    fraction kept
        approx_error:  relative Frobenius error ‖W - Ŵ‖_F / ‖W‖_F
        param_ratio:   (m*r + r + r*n) / (m*n) — theoretical storage ratio
    """
    assert 0.0 < rank_ratio <= 1.0, \
        f"rank_ratio must be in (0, 1], got {rank_ratio}"

    if copy_model:
        model = copy.deepcopy(model)

    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_layers is not None and name not in target_layers:
            continue

        W = module.weight.data.numpy()   # shape: (out_features, in_features)
        m, n = W.shape

        # Economy SVD: U (m,k), S (k,), Vt (k,n) where k = min(m,n)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)

        # Truncate to rank r
        k = min(m, n)
        r = max(1, int(rank_ratio * k))

        U_r  = U[:, :r]        # (m, r)
        S_r  = S[:r]           # (r,)
        Vt_r = Vt[:r, :]       # (r, n)

        # Reconstruct approximation
        W_approx = (U_r * S_r) @ Vt_r   # (m, r) * (r,) = (m, r), then @ (r, n)

        # Relative approximation error (Frobenius norm)
        frob_error = np.linalg.norm(W - W_approx, "fro") / (np.linalg.norm(W, "fro") + 1e-12)

        # Write back
        module.weight.data = torch.tensor(W_approx, dtype=torch.float32)

        stats[name] = {
            "rank_r":       r,
            "rank_full":    k,
            "rank_ratio":   r / k,
            "approx_error": float(frob_error),
            "param_ratio":  (m * r + r + r * n) / (m * n),  # theoretical storage
            "shape":        (m, n),
        }

    return model, stats


def svd_energy_retained(W: np.ndarray, rank_ratio: float) -> float:
    """
    Compute the fraction of singular value energy retained at given rank.

    Energy = sum(S_r^2) / sum(S_all^2)
    This is a measure of how much 'information' is kept.
    High energy retention -> smaller approximation error.
    """
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    r = max(1, int(rank_ratio * len(S)))
    return float((S[:r] ** 2).sum() / (S ** 2).sum())
