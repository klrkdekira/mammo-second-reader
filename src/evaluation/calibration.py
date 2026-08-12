"""Temperature scaling and reliability metrics.

Temperature scaling is a single-parameter post-hoc calibration method
(Guo et al. 2017). A scalar T is fit on validation by minimising NLL,
then applied to test logits before the sigmoid.
"""

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

MIN_TEMPERATURE = 1e-6
CALIBRATION_VERSION = 2


@dataclass(frozen=True)
class TemperatureFit:
    """Details from fitting the temperature."""

    temperature: float
    initial_nll: float
    final_nll: float
    n_iter: int
    function_evaluations: int
    finite: bool
    improved: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


class TemperatureScaler(nn.Module):
    """Scale logits with a positive temperature."""

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        if initial_temperature <= MIN_TEMPERATURE:
            raise ValueError("initial_temperature must be positive")
        raw = np.log(np.expm1(initial_temperature - MIN_TEMPERATURE))
        self.raw_temperature = nn.Parameter(torch.tensor([raw], dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + MIN_TEMPERATURE

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def _validate_fit_inputs(
    val_logits: torch.Tensor, val_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = val_logits.detach().float().reshape(-1)
    labels = val_labels.detach().float().reshape(-1)
    if logits.numel() == 0 or labels.numel() == 0:
        raise ValueError("Temperature fitting requires at least one validation case.")
    if logits.shape != labels.shape:
        raise ValueError("Validation logits and labels must have the same shape.")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Validation logits and labels must be finite.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Validation labels must contain only 0 and 1.")
    return logits, labels


def fit_temperature_with_diagnostics(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 200,
) -> TemperatureFit:
    """Fit the temperature and return the details."""
    logits, labels = _validate_fit_inputs(val_logits, val_labels)
    if lr <= 0 or max_iter < 1:
        raise ValueError("lr must be positive and max_iter must be at least 1.")
    scaler = TemperatureScaler().to(logits.device)
    optimiser = optim.LBFGS(
        [scaler.raw_temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()
    initial_nll = float(criterion(scaler(logits), labels).detach().item())

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(scaler.temperature.detach().item())
    final_nll = float(criterion(scaler(logits), labels).detach().item())
    state = optimiser.state.get(scaler.raw_temperature, {})
    finite = bool(np.isfinite(temperature) and np.isfinite(final_nll))
    return TemperatureFit(
        temperature=temperature,
        initial_nll=initial_nll,
        final_nll=final_nll,
        n_iter=int(state.get("n_iter", 0)),
        function_evaluations=int(state.get("func_evals", 0)),
        finite=finite,
        improved=finite and final_nll <= initial_nll + 1e-8,
    )


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    """Fit and return the temperature."""
    return fit_temperature_with_diagnostics(
        val_logits, val_labels, lr=lr, max_iter=max_iter
    ).temperature


def _validate_probability_inputs(
    probs: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probs, dtype=float).ravel()
    targets = np.asarray(labels, dtype=float).ravel()
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if probabilities.size == 0 or targets.size == 0:
        raise ValueError("Calibration metrics require at least one case.")
    if probabilities.shape != targets.shape:
        raise ValueError("Probabilities and labels must have the same shape.")
    if not np.isfinite(probabilities).all() or not np.isfinite(targets).all():
        raise ValueError("Probabilities and labels must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if np.any((targets != 0.0) & (targets != 1.0)):
        raise ValueError("Labels must contain only 0 and 1.")
    return probabilities, targets


def _bin_masks(probs: np.ndarray, n_bins: int):
    """Yield disjoint equal-width masks: [lo, hi), final bin [lo, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (lo, hi) in enumerate(itertools.pairwise(edges)):
        if index == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        yield lo, hi, mask


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted average gap between predicted and observed across bins.

    Uses equal-width binning over [0, 1] with n_bins bins, per Guo 2017.
    """
    probs, labels = _validate_probability_inputs(probs, labels, n_bins)
    ece = 0.0
    for _, _, mask in _bin_masks(probs, n_bins):
        if mask.sum() == 0:
            continue
        avg_conf = float(probs[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, predicted_means, observed_means) for plotting."""
    probs, labels = _validate_probability_inputs(probs, labels, n_bins)
    centres, preds, obs = [], [], []
    for lo, hi, mask in _bin_masks(probs, n_bins):
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        preds.append(float(probs[mask].mean()))
        obs.append(float(labels[mask].mean()))
    return np.array(centres), np.array(preds), np.array(obs)
