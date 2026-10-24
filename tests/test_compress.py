"""
tests/test_compress.py
======================
Unit tests for compression algorithms:
- Magnitude pruning & sparsity preservation
- Dynamic INT8 quantization
- Low-rank matrix factorization (SVD)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from compress.pruning import (
    apply_magnitude_pruning,
    make_pruning_permanent,
    _compute_sparsity_stats,
    prune_and_finetune,
)
from compress.quantization import apply_dynamic_ptq
from compress.low_rank import apply_svd_approximation, svd_energy_retained


class TestPruning:
    def test_magnitude_pruning_exact_sparsity(self):
        m = nn.Sequential(
            nn.Linear(20, 20, bias=False),
            nn.Linear(20, 10, bias=False),
        )
        total_weights = 20 * 20 + 20 * 10  # 600 weights
        target_ratio = 0.50

        stats = apply_magnitude_pruning(m, target_ratio, make_permanent=True)
        assert stats["total_weights"] == total_weights
        assert stats["total_zeros"] == int(total_weights * target_ratio)
        assert abs(stats["global_sparsity"] - target_ratio) < 1e-4

    def test_sparsity_preserved_across_finetuning(self):
        m = nn.Linear(50, 50, bias=False)
        target_ratio = 0.60

        def dummy_train_fn(model, epochs):
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(epochs):
                opt.zero_grad()
                out = model(torch.randn(10, 50)).sum()
                out.backward()
                opt.step()
            return {"loss": 0.0}

        model_pruned, stats, _ = prune_and_finetune(
            m,
            pruning_ratio=target_ratio,
            train_fn=dummy_train_fn,
            finetune_epochs=3,
            copy_model=True,
        )

        assert abs(stats["global_sparsity"] - target_ratio) < 1e-3
        actual_zeros = (model_pruned.weight == 0).sum().item()
        assert actual_zeros == int(50 * 50 * target_ratio)


class TestQuantization:
    def test_dynamic_ptq_replaces_linear(self):
        m = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        q_model = apply_dynamic_ptq(m, dtype=torch.qint8, copy_model=True)
        x = torch.randn(4, 32)
        out = q_model(x)
        assert out.shape == (4, 8)


class TestLowRank:
    def test_apply_svd_approximation(self):
        m = nn.Sequential(
            nn.Linear(40, 20, bias=True),
            nn.Linear(20, 10, bias=True),
        )
        approx_m, stats = apply_svd_approximation(m, rank_ratio=0.5)
        x = torch.randn(4, 40)
        out = approx_m(x)
        assert out.shape == (4, 10)
        assert len(stats) == 2
        for layer_name, s in stats.items():
            assert 0.0 <= s["approx_error"] <= 1.0
            assert s["rank_ratio"] <= 0.5

    def test_svd_energy_retained(self):
        w = np.random.default_rng(42).standard_normal((30, 20))
        energy_50 = svd_energy_retained(w, rank_ratio=0.5)
        energy_100 = svd_energy_retained(w, rank_ratio=1.0)
        assert 0.0 < energy_50 <= 1.0
        assert abs(energy_100 - 1.0) < 1e-5

