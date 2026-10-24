"""
optim/trainer.py
================
PyTorch training and evaluation engine.

Provides:
  train_one_epoch()   — single epoch, returns loss + accuracy
  evaluate()          — full eval pass with accuracy + F1
  train()             — full training loop with history tracking
  get_f1_scores()     — macro/per-class F1 using sklearn

Follows experimental hygiene:
  - Seed is set externally before calling train()
  - Hyperparameters passed via config dict (not hardcoded)
  - Validation set is never used for tuning inside this module
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import f1_score, classification_report


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> dict:
    """
    One training epoch.

    Returns:
        dict: loss (float), accuracy (float), elapsed_sec (float)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    t0         = time.perf_counter()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    elapsed = time.perf_counter() - t0
    return {
        "loss":        total_loss / total,
        "accuracy":    correct / total,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Full evaluation pass (no gradient).

    Returns:
        dict: loss, accuracy, macro_f1, per_class_f1, predictions, targets
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_total    = len(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        "loss":          total_loss / n_total,
        "accuracy":      (all_preds == all_labels).mean(),
        "macro_f1":      macro_f1,
        "per_class_f1":  per_class_f1.tolist(),
        "predictions":   all_preds,
        "targets":       all_labels,
        "n_samples":     n_total,
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Full training loop.

    Config keys (all required unless marked optional):
        lr:            learning rate
        epochs:        number of training epochs
        optimizer:     'adam' or 'sgd'
        momentum:      SGD momentum (optional, default 0.9)
        weight_decay:  L2 regularisation (optional, default 0.0)
        scheduler:     'cosine' | 'step' | None  (optional)
        step_size:     for StepLR scheduler (optional)
        gamma:         for StepLR scheduler (optional)

    Returns:
        history dict with keys:
            train_loss, train_acc, val_loss, val_acc, epoch_times
    """
    criterion = nn.CrossEntropyLoss()

    # Build optimizer
    opt_name = config.get("optimizer", "adam").lower()
    lr       = config["lr"]
    wd       = config.get("weight_decay", 0.0)

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Build scheduler
    scheduler_name = config.get("scheduler", None)
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"]
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.5),
        )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "epoch_times": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_stats = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["accuracy"])
        history["epoch_times"].append(train_stats["elapsed_sec"])

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:>3}/{config['epochs']}  "
                f"train_loss={train_stats['loss']:.4f}  "
                f"train_acc={train_stats['accuracy']:.4f}  "
                f"val_acc={val_stats['accuracy']:.4f}"
            )

    history["best_val_acc"] = best_val_acc
    return history


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    Call this BEFORE constructing your model and loading data.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
