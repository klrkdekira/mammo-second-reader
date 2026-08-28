"""Loss functions for binary and patch classification."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data.patch_manifest import PATCH_CLASSES


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


def make_patch_criterion(
    train_csv: str | Path,
    device: torch.device | None = None,
    label_smoothing: float = 0.0,
    n_classes: int = len(PATCH_CLASSES),
) -> nn.Module:
    """Build cross-entropy loss with mean-one inverse-frequency weights."""
    frame = pd.read_csv(train_csv)
    counts = frame["class_id"].astype(int).value_counts()
    missing = set(range(n_classes)) - set(counts.index)
    if missing:
        raise ValueError(
            f"{train_csv} has no examples of class id(s) {sorted(missing)}; "
            "a patient-disjoint patch fold must cover every class."
        )
    frequency = torch.tensor(
        [float(counts[index]) for index in range(n_classes)], dtype=torch.float32
    )
    weight = frequency.sum() / frequency
    weight = weight / weight.mean()
    if device is not None:
        weight = weight.to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
