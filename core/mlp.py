"""
core/mlp.py
===========
Multi-Layer Perceptron built from core/layers.py primitives.

Wraps a sequence of layers into a unified model with:
  - forward()     : full forward pass
  - backward()    : full backward pass (reverse layer order)
  - parameters()  : all (param, grad) pairs for the optimizer
  - predict()     : argmax of logits (classification)
  - set_training(): toggle Dropout layers between train/eval mode
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from core.layers import Layer, Dropout


class MLP:
    """
    A sequential stack of layers.

    Example:
        from core.layers import Linear, ReLU, Dropout
        from core.mlp import MLP

        model = MLP([
            Linear(16, 64),
            ReLU(),
            Dropout(p=0.3),
            Linear(64, 7),
        ])
    """

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self._last_output: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Pass x through every layer in order.
        Returns raw logits (no softmax applied here).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self._last_output = out
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # -----------------------------------------------------------------------
    # Backward
    # -----------------------------------------------------------------------

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backpropagate loss_grad through every layer in reverse order.
        Updates each layer's internal .dW and .db attributes.

        Args:
            loss_grad: dL/d(logits), shape (batch, num_classes).
                       Returned by loss.backward().
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Collect all (param, grad) pairs from every layer.
        Passed directly to optimizer.step().
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """Zero all gradient arrays. Call before each backward pass."""
        for param, grad in self.parameters():
            grad[:] = 0.0

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns predicted class indices, shape (batch,)."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities, shape (batch, num_classes)."""
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # -----------------------------------------------------------------------
    # Training mode toggle
    # -----------------------------------------------------------------------

    def set_training(self, mode: bool) -> None:
        """Toggle all Dropout layers between train (True) and eval (False)."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = mode

    def train(self) -> "MLP":
        self.set_training(True)
        return self

    def eval(self) -> "MLP":
        self.set_training(False)
        return self

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def param_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.size for p, _ in self.parameters())

    def __repr__(self) -> str:
        lines = ["MLP("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer}")
        lines.append(f"  total params: {self.param_count():,}")
        lines.append(")")
        return "\n".join(lines)
