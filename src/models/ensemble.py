"""Probability-averaged ensemble across multiple models."""

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module, loader: DataLoader, device: torch.device | None = None
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0).ravel()


@torch.no_grad()
def ensemble_predict(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return the mean sigmoid probability across the supplied models."""
    return np.mean(
        np.stack([predict_proba(m, loader, device) for m in models], axis=0),
        axis=0,
    )
