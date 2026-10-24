"""
nn/architectures.py
===================
PyTorch model definitions for the compression benchmark.

Models:
  SmallCNN  — Primary model for Fashion-MNIST (~206K params)
  SmallMLP  — Alternative MLP baseline for tabular data (Dry Bean)

All models are documented with:
  - input/output shapes
  - approximate parameter count
  - intended dataset
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SmallCNN — Primary model (Fashion-MNIST)
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """
    Small CNN for Fashion-MNIST classification.

    Architecture:
        Conv1:    (1, 28, 28) -> (16, 28, 28)   [3x3, pad=1]
        ReLU + MaxPool2d(2): -> (16, 14, 14)
        Conv2:    (16, 14, 14) -> (32, 14, 14)  [3x3, pad=1]
        ReLU + MaxPool2d(2): -> (32, 7, 7)
        Dropout2d(p)
        Flatten:  -> (1568,)
        FC1:      1568 -> 128
        ReLU + Dropout(p)
        FC2:      128 -> 10

    Approx params:
        Conv1:  1*16*3*3 + 16    =    160
        Conv2: 16*32*3*3 + 32    =  4,640
        FC1:   1568*128  + 128   = 200,832
        FC2:    128*10   + 10    =  1,290
        Total:                  ~= 206,922

    Args:
        num_classes: number of output classes (default 10)
        dropout_p:   dropout rate for Dropout / Dropout2d layers
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 28->14

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 14->7
            nn.Dropout2d(p=dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# SmallMLP — Tabular model (Dry Bean Dataset)
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """
    Small MLP for tabular classification (Dry Bean, 16 features -> 7 classes).

    Architecture:
        FC1: in_features -> 64  + ReLU + Dropout
        FC2: 64 -> 32            + ReLU + Dropout
        FC3: 32 -> num_classes

    Approx params (for Dry Bean, in_features=16):
        FC1: 16*64 + 64 = 1,088
        FC2: 64*32 + 32 = 2,080
        FC3: 32*7  + 7  =   231
        Total:            ~= 3,399

    Args:
        in_features:  number of input features
        num_classes:  number of output classes
        hidden_sizes: tuple of hidden layer sizes
        dropout_p:    dropout rate
    """

    def __init__(
        self,
        in_features: int = 16,
        num_classes: int = 7,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout_p: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev = in_features
        for hidden in hidden_sizes:
            layers += [
                nn.Linear(prev, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            prev = hidden
        layers.append(nn.Linear(prev, num_classes, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build a model by name.

    Args:
        name:   'SmallCNN' or 'SmallMLP'
        kwargs: forwarded to model constructor

    Returns:
        nn.Module instance (on CPU)
    """
    registry = {
        "SmallCNN": SmallCNN,
        "SmallMLP": SmallMLP,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
