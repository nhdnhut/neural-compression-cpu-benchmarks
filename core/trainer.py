"""
core/trainer.py
===============
Mini-batch training loop for the NumPy MLP framework.

Handles:
  - epoch loop with mini-batch shuffling
  - loss computation and backprop
  - accuracy tracking (train + val)
  - history dict for plotting loss/accuracy curves
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from core.mlp import MLP
from core.losses import CrossEntropyLoss


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer,
    loss_fn: CrossEntropyLoss,
    epochs: int = 50,
    batch_size: int = 64,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    verbose: bool = True,
    verbose_every: int = 10,
) -> dict:
    """
    Train model for `epochs` passes over the data.

    Args:
        model:        MLP instance
        X_train:      training features, shape (n, features)
        y_train:      training labels,   shape (n,), integer class indices
        optimizer:    SGD or Adam instance
        loss_fn:      CrossEntropyLoss instance
        epochs:       number of full passes over training data
        batch_size:   mini-batch size
        X_val/y_val:  optional validation set (evaluated at end of each epoch)
        verbose:      print progress
        verbose_every: print every N epochs

    Returns:
        history dict with keys:
          'train_loss', 'train_acc', 'val_loss' (if val provided), 'val_acc'
    """
    n = X_train.shape[0]
    history = {"train_loss": [], "train_acc": []}
    if X_val is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    for epoch in range(1, epochs + 1):
        # ---- shuffle training data ----------------------------------------
        model.train()
        indices = np.random.permutation(n)
        X_shuf = X_train[indices]
        y_shuf = y_train[indices]

        epoch_losses = []
        epoch_correct = 0

        # ---- mini-batch loop -----------------------------------------------
        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            # Forward
            logits = model.forward(X_batch)
            loss_val = loss_fn.forward(logits, y_batch)

            # Backward
            loss_grad = loss_fn.backward()
            model.backward(loss_grad)

            # Optimizer step
            optimizer.step(model.parameters())

            # Track stats
            epoch_losses.append(loss_val * len(y_batch))   # weighted by batch size
            preds = np.argmax(logits, axis=1)
            epoch_correct += (preds == y_batch).sum()

        # ---- epoch stats ---------------------------------------------------
        train_loss = sum(epoch_losses) / n
        train_acc  = epoch_correct / n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ---- validation (no gradient) -------------------------------------
        if X_val is not None:
            model.eval()
            val_logits = model.forward(X_val)
            val_loss   = loss_fn.forward(val_logits, y_val)
            val_preds  = np.argmax(val_logits, axis=1)
            val_acc    = (val_preds == y_val).mean()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        # ---- logging -------------------------------------------------------
        if verbose and (epoch % verbose_every == 0 or epoch == 1):
            msg = f"Epoch {epoch:>4}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}"
            if X_val is not None:
                msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            print(msg)

    return history


def evaluate(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: CrossEntropyLoss,
) -> dict:
    """
    Full evaluation pass (no gradient, eval mode).

    Returns:
        dict with 'loss', 'accuracy', 'predictions'
    """
    model.eval()
    logits = model.forward(X)
    loss   = loss_fn.forward(logits, y)
    preds  = np.argmax(logits, axis=1)
    acc    = (preds == y).mean()
    return {"loss": loss, "accuracy": acc, "predictions": preds}
