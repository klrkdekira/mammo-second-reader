"""Class-imbalance handling via pos_weight in BCEWithLogitsLoss.

pos_weight is computed from the training fold only to prevent label-leakage from val/test.
Optional label smoothing pulls hard 0/1 targets towards 0.5 before the BCE.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn


class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss over smoothed targets: y -> y*(1-eps) + eps/2."""

    def __init__(self, pos_weight: torch.Tensor, smoothing: float) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def make_criterion(
    train_csv: str | Path,
    device: torch.device | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Return BCEWithLogitsLoss with pos_weight = n_neg / n_pos."""
    df = pd.read_csv(train_csv)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    if n_pos == 0:
        raise ValueError(f"No positive examples in {train_csv}")
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)
    if label_smoothing > 0.0:
        return SmoothedBCEWithLogitsLoss(pos_weight, label_smoothing)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
