"""
core/optimizers.py
==================
Gradient-based optimizers implemented from scratch in NumPy.

Each optimizer exposes one method:
    step(params_and_grads)
where params_and_grads is a list of (param_array, grad_array) pairs.
The optimizer modifies param arrays in-place.

Implemented:
  - SGD (with optional momentum)
  - Adam (with bias-corrected moment estimates)
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class SGD:
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule (no momentum):
        param -= lr * grad

    Update rule (with momentum, Polyak's heavy ball):
        v     = momentum * v - lr * grad
        param += v

    L2 weight decay is applied BEFORE the momentum update:
        grad_effective = grad + weight_decay * param

    Args:
        lr:           learning rate (step size)
        momentum:     momentum factor, 0 = vanilla SGD
        weight_decay: L2 regularisation coefficient (lambda)
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        assert lr > 0, f"lr must be > 0, got {lr}"
        assert 0.0 <= momentum < 1.0, f"momentum must be in [0, 1), got {momentum}"
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[int, np.ndarray] = {}   # keyed by param id

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for i, (param, grad) in enumerate(params_and_grads):
            # L2 weight decay: effectively adds lambda*param to gradient
            effective_grad = grad + self.weight_decay * param

            if self.momentum > 0.0:
                if i not in self._velocity:
                    self._velocity[i] = np.zeros_like(param)
                v = self._velocity[i]
                v[:] = self.momentum * v - self.lr * effective_grad
                param += v
            else:
                param -= self.lr * effective_grad

    def __repr__(self) -> str:
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay})")


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad      # 1st moment (mean)
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2    # 2nd moment (variance)
        m̂   = m_t / (1 - beta1^t)                       # bias-corrected mean
        v̂   = v_t / (1 - beta2^t)                       # bias-corrected variance
        param -= lr * m̂ / (sqrt(v̂) + eps)

    L2 weight decay: AdamW style (decay applied to param, not gradient).

    Args:
        lr:           learning rate (typical: 1e-3)
        beta1:        decay for 1st moment estimate (typical: 0.9)
        beta2:        decay for 2nd moment estimate (typical: 0.999)
        eps:          numerical stability constant (typical: 1e-8)
        weight_decay: AdamW-style L2 decay coefficient
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: dict[int, np.ndarray] = {}     # first moments
        self._v: dict[int, np.ndarray] = {}     # second moments
        self._t: int = 0                        # global timestep counter

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._t += 1
        # Precompute bias-correction denominators (same for all params this step)
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i, (param, grad) in enumerate(params_and_grads):
            if i not in self._m:
                self._m[i] = np.zeros_like(param)
                self._v[i] = np.zeros_like(param)

            # Update biased moment estimates
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * grad ** 2

            # Bias-corrected estimates
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2

            # Parameter update (AdamW: weight decay on param, not gradient)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

    def __repr__(self) -> str:
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
                f"eps={self.eps}, weight_decay={self.weight_decay})")
