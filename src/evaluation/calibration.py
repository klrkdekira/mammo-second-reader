"""Temperature scaling and reliability metrics.

Temperature scaling is a single-parameter post-hoc calibration method
(Guo et al. 2017). A scalar T is fit on validation by minimising NLL,
then applied to test logits before the sigmoid.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Single-parameter scaling. logits -> logits / T."""

    def __init__(self) -> None:
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor,
                    lr: float = 0.01, max_iter: int = 200) -> float:
    """LBFGS over a scalar T on BCE-with-logits NLL on val. Returns T."""
    scaler = TemperatureScaler()
    optimiser = optim.LBFGS([scaler.T], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = criterion(scaler(val_logits), val_labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(scaler.T.detach().item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """Weighted average gap between predicted and observed across bins.

    Uses equal-width binning over [0, 1] with n_bins bins, per Guo 2017.
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel().astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(probs[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, predicted_means, observed_means) for plotting."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centres, preds, obs = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        preds.append(float(probs[mask].mean()))
        obs.append(float(labels[mask].mean()))
    return np.array(centres), np.array(preds), np.array(obs)
