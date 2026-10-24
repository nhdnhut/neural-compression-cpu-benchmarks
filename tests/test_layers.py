"""
tests/test_layers.py
====================
Unit tests for core/layers.py

Critical: gradient check verifies that our hand-derived backprop
matches the numerical gradient. If this test fails, the math is wrong.

Tolerance: relative error < 1e-5 is generally acceptable for float32.
We use float64 in the gradient check for better numerical precision.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.layers import Linear, ReLU, Sigmoid, Tanh, Dropout, Flatten


# ---------------------------------------------------------------------------
# Numerical gradient check utility
# ---------------------------------------------------------------------------

def numerical_gradient(loss_fn, param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute numerical (finite-difference) gradient of loss_fn w.r.t. param.
    Uses centered difference: (f(x+eps) - f(x-eps)) / (2*eps)
    """
    grad = np.zeros_like(param, dtype=np.float64)
    it   = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx     = it.multi_index
        old_val = float(param[idx])

        param[idx] = old_val + eps
        fp         = float(loss_fn())

        param[idx] = old_val - eps
        fm         = float(loss_fn())

        param[idx] = old_val
        grad[idx]  = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Relative error between two gradient arrays."""
    num   = np.abs(analytical - numerical).max()
    denom = max(np.abs(analytical).max(), np.abs(numerical).max(), 1e-8)
    return num / denom


# ---------------------------------------------------------------------------
# Linear layer tests
# ---------------------------------------------------------------------------

class TestLinear:

    def test_forward_shape(self):
        layer = Linear(4, 8)
        x     = np.random.randn(3, 4)
        out   = layer.forward(x)
        assert out.shape == (3, 8), f"Expected (3, 8), got {out.shape}"

    def test_forward_no_bias(self):
        layer = Linear(4, 8, bias=False)
        x     = np.zeros((2, 4))
        out   = layer.forward(x)
        assert np.allclose(out, 0.0), "Zero input should give zero output when no bias"

    def test_backward_shape(self):
        layer = Linear(4, 8)
        x     = np.random.randn(3, 4)
        _     = layer.forward(x)
        grad  = np.random.randn(3, 8)
        dx    = layer.backward(grad)
        assert dx.shape == (3, 4), f"dx shape: expected (3, 4), got {dx.shape}"
        assert layer.dW.shape == (4, 8)
        assert layer.db.shape == (8,)

    def test_gradient_check_W(self):
        """Numerical gradient check for weight W."""
        rng   = np.random.default_rng(0)
        layer = Linear(5, 3)
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

        x_in    = rng.standard_normal((2, 5))
        grad_out = rng.standard_normal((2, 3))

        def loss_fn():
            out = layer.forward(x_in)
            return (out * grad_out).sum()

        # Analytical
        layer.forward(x_in)
        layer.backward(grad_out)
        analytical_dW = layer.dW.copy()

        # Numerical
        numerical_dW = numerical_gradient(loss_fn, layer.W)

        err = relative_error(analytical_dW, numerical_dW)
        assert err < 1e-5, f"dW gradient check failed: relative error = {err:.2e}"

    def test_gradient_check_b(self):
        """Numerical gradient check for bias b."""
        rng   = np.random.default_rng(1)
        layer = Linear(5, 3)
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

        x_in     = rng.standard_normal((4, 5))
        grad_out = rng.standard_normal((4, 3))

        def loss_fn():
            out = layer.forward(x_in)
            return (out * grad_out).sum()

        layer.forward(x_in)
        layer.backward(grad_out)
        analytical_db = layer.db.copy()
        numerical_db  = numerical_gradient(loss_fn, layer.b)

        err = relative_error(analytical_db, numerical_db)
        assert err < 1e-5, f"db gradient check failed: relative error = {err:.2e}"

    def test_gradient_check_x(self):
        """Numerical gradient check for input x."""
        rng   = np.random.default_rng(2)
        layer = Linear(5, 3)
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

        x_in     = rng.standard_normal((2, 5))
        grad_out = rng.standard_normal((2, 3))

        def loss_fn():
            out = layer.forward(x_in)
            return (out * grad_out).sum()

        layer.forward(x_in)
        analytical_dx = layer.backward(grad_out)
        numerical_dx  = numerical_gradient(loss_fn, x_in)

        err = relative_error(analytical_dx, numerical_dx)
        assert err < 1e-5, f"dx gradient check failed: relative error = {err:.2e}"


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------

class TestReLU:

    def test_forward_zeros_negative(self):
        layer = ReLU()
        x     = np.array([-1.0, 0.0, 1.0, 2.0])
        out   = layer.forward(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert np.allclose(out, expected)

    def test_gradient_check(self):
        rng   = np.random.default_rng(3)
        layer = ReLU()
        x     = rng.standard_normal((3, 4))
        g_out = rng.standard_normal((3, 4))

        def loss_fn():
            return (layer.forward(x) * g_out).sum()

        layer.forward(x)
        analytical = layer.backward(g_out)
        numerical  = numerical_gradient(loss_fn, x)
        err = relative_error(analytical, numerical)
        assert err < 1e-5, f"ReLU gradient check failed: {err:.2e}"


class TestSigmoid:

    def test_output_range(self):
        layer = Sigmoid()
        x     = np.array([-1000.0, 0.0, 1000.0])
        out   = layer.forward(x)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert np.isclose(out[1], 0.5)

    def test_gradient_check(self):
        rng   = np.random.default_rng(4)
        layer = Sigmoid()
        x     = rng.standard_normal((2, 5))
        g_out = rng.standard_normal((2, 5))

        def loss_fn():
            return (layer.forward(x) * g_out).sum()

        layer.forward(x)
        analytical = layer.backward(g_out)
        numerical  = numerical_gradient(loss_fn, x)
        err = relative_error(analytical, numerical)
        assert err < 1e-5, f"Sigmoid gradient check failed: {err:.2e}"


class TestFlatten:

    def test_shape(self):
        layer = Flatten()
        x     = np.random.randn(4, 3, 2)
        out   = layer.forward(x)
        assert out.shape == (4, 6)

    def test_backward_shape_restored(self):
        layer = Flatten()
        x     = np.random.randn(4, 3, 2)
        layer.forward(x)
        grad  = np.random.randn(4, 6)
        dx    = layer.backward(grad)
        assert dx.shape == x.shape


# ---------------------------------------------------------------------------
# Parameters check
# ---------------------------------------------------------------------------

class TestParameters:

    def test_linear_params_count(self):
        layer = Linear(10, 5)
        params = layer.parameters()
        assert len(params) == 2          # W and b
        assert params[0][0].shape == (10, 5)
        assert params[1][0].shape == (5,)

    def test_relu_no_params(self):
        layer = ReLU()
        assert layer.parameters() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
