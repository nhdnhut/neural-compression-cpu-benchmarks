"""
data/loaders.py
===============
Dataset loaders for Fashion-MNIST and UCI Dry Bean Dataset.

Both return standard (X_train, y_train, X_val, y_val, X_test, y_test)
numpy arrays for the NumPy framework, and DataLoaders for PyTorch.

Split strategy:
  Fashion-MNIST: torchvision's default 60K/10K train/test split.
                 We further split 10% of training set as validation.
  Dry Bean:      80/10/10 train/val/test split with stratification.

Preprocessing:
  Fashion-MNIST: normalize to [0,1] (divide by 255), then
                 standardize per-channel using train set statistics only.
  Dry Bean:      StandardScaler fit on train set only (no data leakage).
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Fashion-MNIST
# ---------------------------------------------------------------------------

def load_fashion_mnist(
    data_dir: str = "./data/raw",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> dict:
    """
    Load and preprocess Fashion-MNIST.

    Returns:
        dict with keys:
            train_loader, val_loader, test_loader   — PyTorch DataLoaders
            train_dataset, val_dataset, test_dataset
            input_shape: (1, 28, 28)
            num_classes: 10
            class_names: list of 10 class name strings
    """
    # Normalisation stats computed from Fashion-MNIST training set
    # mean=0.2860, std=0.3530  (per-pixel, across all training images)
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),   # standardise
    ])

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download if not cached
    full_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Split train -> train + val
    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal",       "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
    ]

    return {
        "train_loader":   train_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "train_dataset":  train_dataset,
        "val_dataset":    val_dataset,
        "test_dataset":   test_dataset,
        "input_shape":    (1, 28, 28),
        "num_classes":    10,
        "class_names":    class_names,
        "n_train":        n_train,
        "n_val":          n_val,
        "n_test":         len(test_dataset),
    }


# ---------------------------------------------------------------------------
# UCI Dry Bean Dataset
# ---------------------------------------------------------------------------

def load_dry_bean(
    data_dir: str = "./data/raw",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Load and preprocess the UCI Dry Bean Dataset.
    Downloads automatically if not present.

    Dataset: 7 classes, 13,611 samples, 16 numerical features.
    Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

    Returns:
        dict with keys:
            train_loader, val_loader, test_loader
            X_train, y_train, X_val, y_val, X_test, y_test  (numpy arrays)
            in_features: 16
            num_classes: 7
            class_names: list of 7 bean variety names
    """
    import urllib.request
    import zipfile
    import os

    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "Dry_Bean_Dataset.csv"

    # Download if not cached
    if not csv_path.exists():
        print("[data] Downloading Dry Bean Dataset...")
        _download_dry_bean(raw_dir)

    # Load CSV
    import csv as csv_module
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append(row)

    # Parse features and labels
    feature_cols = [
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
        "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
        "Extent", "Solidity", "roundness", "Compactness",
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
    ]
    class_names = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    X_all = np.array([[float(row[f]) for f in feature_cols] for row in rows], dtype=np.float32)
    y_all = np.array([class_to_idx[row["Class"]] for row in rows], dtype=np.int64)

    # Stratified split
    X_train, y_train, X_val, y_val, X_test, y_test = _stratified_split(
        X_all, y_all, val_fraction, test_fraction, seed
    )

    # Standardise (fit on train set only — no leakage)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    # Build DataLoaders
    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": make_loader(X_train, y_train, shuffle=True),
        "val_loader":   make_loader(X_val,   y_val,   shuffle=False),
        "test_loader":  make_loader(X_test,  y_test,  shuffle=False),
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "in_features":  len(feature_cols),
        "num_classes":  len(class_names),
        "class_names":  class_names,
        "scaler_mean":  mean,
        "scaler_std":   std,
    }


def _stratified_split(X, y, val_frac, test_frac, seed):
    """Simple stratified split into train/val/test."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def _download_dry_bean(raw_dir: Path):
    """
    Download Dry Bean Dataset from UCI.
    Falls back to a minimal embedded version if network unavailable.
    """
    import urllib.request
    url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
    zip_path = raw_dir / "dry_bean.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        # Rename if needed
        candidates = list(raw_dir.rglob("*.csv")) + list(raw_dir.rglob("*.xlsx"))
        if candidates:
            import shutil
            # Try to find the right file and copy/convert
            for c in candidates:
                if "bean" in c.name.lower() or "dry" in c.name.lower():
                    if c.suffix == ".csv":
                        shutil.copy(c, raw_dir / "Dry_Bean_Dataset.csv")
                    elif c.suffix == ".xlsx":
                        _xlsx_to_csv(c, raw_dir / "Dry_Bean_Dataset.csv")
                    break
    except Exception as e:
        print(f"[data] Download failed: {e}")
        print("[data] Please manually download Dry Bean Dataset from:")
        print("       https://archive.ics.uci.edu/dataset/602/dry+bean+dataset")
        print(f"       and place Dry_Bean_Dataset.csv in {raw_dir}")
        raise


def _xlsx_to_csv(xlsx_path: Path, csv_path: Path):
    """Convert xlsx to csv using openpyxl if available."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        import csv as csv_module
        with open(csv_path, "w", newline="") as f:
            writer = csv_module.writer(f)
            for row in ws.iter_rows(values_only=True):
                writer.writerow(row)
        print(f"[data] Converted {xlsx_path.name} -> {csv_path.name}")
    except ImportError:
        print("[data] openpyxl not installed. pip install openpyxl")
        raise
