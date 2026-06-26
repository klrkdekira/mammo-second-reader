"""Class-imbalance handling via pos_weight in BCEWithLogitsLoss.

pos_weight is computed from the training fold only to prevent label-leakage from val/test.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


def make_criterion(train_csv: str | Path,
                   device: torch.device | None = None) -> nn.Module:
    """Return BCEWithLogitsLoss with pos_weight = n_neg / n_pos."""
    df = pd.read_csv(train_csv)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    if n_pos == 0:
        raise ValueError(f"No positive examples in {train_csv}")
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
