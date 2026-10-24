"""
core/losses.py
==============
Numerically stable loss functions with hand-derived gradients.

Key principle: CrossEntropyLoss uses the log-sum-exp trick to avoid
overflow/underflow when computing softmax probabilities explicitly.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class CrossEntropyLoss:
    """
    Combines LogSoftmax + NLLLoss in one numerically stable operation.

    Forward:
        shifted   = logits - max(logits, axis=1, keepdims=True)   # stability
        log_probs = shifted - log(sum(exp(shifted), axis=1))       # log-softmax
        loss      = -mean(log_probs[i, y_i])                       # NLL

    Backward:
        dL/d(logits) = (softmax(logits) - one_hot(y)) / batch_size

    Args:
        logits: raw network outputs, shape (batch, num_classes)
        labels: integer class indices, shape (batch,)
    """

    def __init__(self):
        self._probs: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._batch_size: int = 0

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Returns scalar loss value."""
        batch_size = logits.shape[0]
        self._batch_size = batch_size
        self._labels = labels

        # log-sum-exp trick: subtract row max before exp
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp              # (batch, classes)

        self._probs = np.exp(log_probs)                # softmax probabilities

        # Gather log-prob of the correct class for each sample
        nll = -log_probs[np.arange(batch_size), labels]
        return float(nll.mean())

    def backward(self) -> np.ndarray:
        """
        Returns gradient w.r.t. logits, shape (batch, num_classes).
        No input needed: reuses cached softmax probs and labels.
        """
        assert self._probs is not None, "Call forward() before backward()"
        grad = self._probs.copy()
        grad[np.arange(self._batch_size), self._labels] -= 1.0
        return grad / self._batch_size

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> float:
        return self.forward(logits, labels)


class MSELoss:
    """
    Mean Squared Error: L = mean((predictions - targets)^2)

    Gradient: dL/d(pred) = 2 * (pred - target) / batch_size
    """

    def __init__(self):
        self._diff: Optional[np.ndarray] = None
        self._batch_size: int = 0

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self._diff = predictions - targets
        self._n_elements = predictions.size   # total elements (batch x features)
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        assert self._diff is not None, "Call forward() before backward()"
        # d/d(pred) mean((pred - target)^2) = 2*(pred-target) / n_total_elements
        return 2.0 * self._diff / self._n_elements

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(predictions, targets)
