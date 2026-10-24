"""
core/layers.py
==============
NumPy implementations of neural network layers.

Every layer implements three methods:
  forward(x)   -> output, caches state needed for backward
  backward(grad) -> upstream gradient w.r.t. input
  parameters() -> list of (param_array, grad_array) tuples

Design principle: no autograd. All gradients derived by hand
from the chain rule and verified via numerical gradient check.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Layer:
    """Abstract base for all layers."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns list of (param, grad) pairs for the optimizer."""
        return []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Linear (Dense) layer
# ---------------------------------------------------------------------------

class Linear(Layer):
    """
    Fully-connected layer: output = x @ W + b

    Shapes:
      x:   (batch, in_features)
      W:   (in_features, out_features)
      b:   (out_features,)
      out: (batch, out_features)

    Gradients (chain rule):
      dL/dW = x^T @ grad          shape: (in, out)
      dL/db = sum(grad, axis=0)   shape: (out,)
      dL/dx = grad @ W^T          shape: (batch, in)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weight initialisation
        if init == "he":
            # He (Kaiming) normal — suited for ReLU activations
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            # Xavier/Glorot — suited for tanh/sigmoid
            std = np.sqrt(1.0 / in_features)
        elif init == "small":
            std = 0.01
        else:
            raise ValueError(f"Unknown init: {init}")

        self.W  = np.random.randn(in_features, out_features) * std
        self.dW = np.zeros_like(self.W)

        self.b  = np.zeros(out_features) if bias else None
        self.db = np.zeros(out_features) if bias else None

        # Cache for backward
        self._x: Optional[np.ndarray] = None

    # -- forward ------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x                          # cache for backward
        out = x @ self.W
        if self.use_bias:
            out = out + self.b
        return out

    # -- backward -----------------------------------------------------------

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        grad: upstream gradient, shape (batch, out_features)
        returns: downstream gradient, shape (batch, in_features)
        """
        assert self._x is not None, "Call forward() before backward()"
        self.dW = self._x.T @ grad               # (in, out)
        if self.use_bias:
            self.db = grad.sum(axis=0)           # (out,)
        dx = grad @ self.W.T                     # (batch, in)
        return dx

    # -- parameters ---------------------------------------------------------

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = [(self.W, self.dW)]
        if self.use_bias:
            params.append((self.b, self.db))
        return params

    def __repr__(self) -> str:
        return f"Linear({self.in_features} -> {self.out_features}, bias={self.use_bias})"


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class ReLU(Layer):
    """
    ReLU(x) = max(0, x)
    Gradient: 1 where x > 0, else 0  (sub-gradient at 0 = 0)
    """

    def __init__(self):
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Layer):
    """
    Sigmoid(x) = 1 / (1 + exp(-x))
    Gradient:   sigma(x) * (1 - sigma(x))
    """

    def __init__(self):
        self._out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable: clip to avoid overflow in exp
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Layer):
    """
    tanh(x),  gradient: 1 - tanh(x)^2
    """

    def __init__(self):
        self._out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)

    def __repr__(self) -> str:
        return "Tanh()"


# ---------------------------------------------------------------------------
# Dropout (training regularization)
# ---------------------------------------------------------------------------

class Dropout(Layer):
    """
    Inverted dropout: scale kept units by 1/(1-p) during training
    so that expected output is the same at test time (no scaling needed).

    Args:
        p: probability of ZEROING a unit (drop rate), not keep rate.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, f"Dropout p must be in [0, 1), got {p}"
        self.p = p
        self._mask: Optional[np.ndarray] = None
        self.training = True          # toggle with layer.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        # Bernoulli mask: keep with prob (1-p)
        self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * self._mask / (1.0 - self.p)   # inverted scaling

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return grad
        return grad * self._mask / (1.0 - self.p)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class Flatten(Layer):
    """Flattens all dims except batch: (B, ...) -> (B, -1)."""

    def __init__(self):
        self._input_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._input_shape)

    def __repr__(self) -> str:
        return "Flatten()"
