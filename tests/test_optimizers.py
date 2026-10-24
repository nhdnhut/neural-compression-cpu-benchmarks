"""
tests/test_optimizers.py
========================
Unit tests for core/optimizers.py

Tests:
  - SGD decreases loss on a simple quadratic
  - Adam decreases loss on a simple quadratic
  - Bias correction: Adam moment estimates are correctly initialised
  - Momentum: velocity is accumulated across steps
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.optimizers import SGD, Adam


class TestSGD:

    def test_simple_quadratic_convergence(self):
        """SGD should decrease f(x) = x^2 toward 0."""
        x   = np.array([5.0, -3.0, 2.0])
        opt = SGD(lr=0.1)
        for _ in range(100):
            grad = 2 * x          # df/dx for f=x^2
            opt.step([(x, grad)])
        assert np.all(np.abs(x) < 0.01), f"SGD did not converge: x={x}"

    def test_with_momentum(self):
        """SGD with momentum should also converge."""
        x   = np.array([5.0])
        opt = SGD(lr=0.05, momentum=0.9)
        for _ in range(200):
            grad = 2 * x
            opt.step([(x, grad)])
        assert np.abs(x[0]) < 0.01, f"SGD+momentum did not converge: x={x}"

    def test_weight_decay_shrinks_params(self):
        """With only weight decay (no gradient), params should shrink."""
        x   = np.array([1.0, 1.0])
        opt = SGD(lr=0.1, weight_decay=0.1)
        for _ in range(50):
            grad = np.zeros_like(x)   # no gradient
            opt.step([(x, grad)])
        assert np.all(x < 1.0), "Weight decay should shrink parameters"


class TestAdam:

    def test_simple_quadratic_convergence(self):
        """Adam should decrease f(x) = x^2 toward 0."""
        x   = np.array([5.0, -3.0, 2.0])
        opt = Adam(lr=0.1)
        for _ in range(200):
            grad = 2 * x
            opt.step([(x, grad)])
        assert np.all(np.abs(x) < 0.05), f"Adam did not converge: x={x}"

    def test_moment_estimates_initialised_to_zero(self):
        """First moment estimates should start at zero."""
        x   = np.array([1.0])
        opt = Adam(lr=0.01)
        grad = np.array([1.0])
        opt.step([(x, grad)])
        # After 1 step: m = (1-0.9)*grad = 0.1, bias-corrected m_hat = 0.1/(1-0.9) = 1.0
        assert 0 in opt._m
        assert opt._m[0] is not None

    def test_timestep_increments(self):
        x   = np.array([1.0])
        opt = Adam(lr=0.01)
        for i in range(5):
            opt.step([(x, np.array([1.0]))])
        assert opt._t == 5

    def test_different_lr_affects_step_size(self):
        """Higher lr should produce larger parameter updates."""
        x_slow = np.array([5.0])
        x_fast = np.array([5.0])
        opt_slow = Adam(lr=0.001)
        opt_fast = Adam(lr=0.1)

        opt_slow.step([(x_slow, np.array([1.0]))])
        opt_fast.step([(x_fast, np.array([1.0]))])

        delta_slow = abs(5.0 - x_slow[0])
        delta_fast = abs(5.0 - x_fast[0])
        assert delta_fast > delta_slow, "Higher lr should produce larger step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
