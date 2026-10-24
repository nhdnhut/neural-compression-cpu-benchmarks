"""
tests/test_losses.py
====================
Unit tests for core/losses.py

Tests:
  - CrossEntropyLoss numerical stability (no overflow with large logits)
  - Loss value sanity (uniform prediction -> log(1/C))
  - Gradient check against numerical gradient
  - MSELoss forward and backward
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.losses import CrossEntropyLoss, MSELoss
from tests.test_layers import numerical_gradient, relative_error


class TestCrossEntropyLoss:

    def test_uniform_prediction(self):
        """With uniform logits, loss should be log(num_classes)."""
        loss_fn = CrossEntropyLoss()
        C = 5
        logits = np.zeros((10, C))       # all zero -> uniform softmax
        labels = np.zeros(10, dtype=int)
        loss   = loss_fn.forward(logits, labels)
        expected = np.log(C)
        assert np.isclose(loss, expected, atol=1e-5), \
            f"Uniform loss: expected {expected:.4f}, got {loss:.4f}"

    def test_perfect_prediction_low_loss(self):
        """With very confident correct prediction, loss should be near 0."""
        loss_fn = CrossEntropyLoss()
        logits  = np.array([[100.0, 0.0, 0.0]])
        labels  = np.array([0])
        loss    = loss_fn.forward(logits, labels)
        assert loss < 1e-4, f"Perfect prediction loss too high: {loss}"

    def test_numerical_stability_large_logits(self):
        """Should not produce NaN/inf with very large logits."""
        loss_fn = CrossEntropyLoss()
        logits  = np.array([[1e6, 1e6, 1e6], [-1e6, -1e6, -1e6]])
        labels  = np.array([0, 1])
        loss    = loss_fn.forward(logits, labels)
        assert np.isfinite(loss), f"Loss not finite for large logits: {loss}"

    def test_gradient_check(self):
        """Gradient check for dL/d(logits)."""
        rng     = np.random.default_rng(42)
        loss_fn = CrossEntropyLoss()
        logits  = rng.standard_normal((4, 7))   # batch=4, classes=7
        labels  = rng.integers(0, 7, size=4)

        def loss_fn_scalar():
            return loss_fn.forward(logits, labels)

        # Analytical gradient
        loss_fn.forward(logits, labels)
        analytical = loss_fn.backward()         # (4, 7)

        # Numerical gradient
        numerical  = numerical_gradient(loss_fn_scalar, logits)

        err = relative_error(analytical, numerical)
        assert err < 1e-5, f"CrossEntropy gradient check failed: {err:.2e}"


class TestMSELoss:

    def test_zero_loss(self):
        loss_fn = MSELoss()
        x = np.ones((5, 3))
        loss = loss_fn.forward(x, x)
        assert np.isclose(loss, 0.0)

    def test_gradient_check(self):
        rng    = np.random.default_rng(7)
        pred   = rng.standard_normal((3, 4))
        target = rng.standard_normal((3, 4)).copy()

        # Compute analytical gradient FIRST on a separate instance
        # so numerical_gradient loop doesn't clobber internal state.
        loss_analytical = MSELoss()
        loss_analytical.forward(pred, target)
        analytical = loss_analytical.backward().copy()   # save before numerical loop

        # Separate instance for numerical gradient
        loss_numerical = MSELoss()
        def loss_fn_scalar():
            return loss_numerical.forward(pred, target)

        numerical = numerical_gradient(loss_fn_scalar, pred)

        err = relative_error(analytical, numerical)
        assert err < 1e-5, f"MSE gradient check failed: {err:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
