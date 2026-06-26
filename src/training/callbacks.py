"""Checkpointing and early stopping, both gated on validation AUC not loss.

BCE loss and AUC don't move monotonically.

The best-loss epoch is rarely the best-AUC epoch.
"""

from pathlib import Path

import torch


class BestAUCCheckpoint:
    """Save the model whose val AUC is the highest seen so far."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.best = -1.0

    def __call__(self, val_auc: float, model: torch.nn.Module) -> bool:
        if val_auc > self.best:
            self.best = val_auc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            return True
        return False


class EarlyStopping:
    """Stop when val AUC has not improved for `patience` consecutive epochs."""

    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best = -1.0
        self.bad_epochs = 0

    def __call__(self, val_auc: float) -> bool:
        if val_auc > self.best:
            self.best = val_auc
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
