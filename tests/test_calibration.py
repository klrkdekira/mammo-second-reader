"""Unit tests for temperature scaling and calibration metrics."""

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.calibration import expected_calibration_error, fit_temperature


def test_fit_temperature_lowers_validation_nll_on_overconfident_logits():
    torch.manual_seed(0)
    n = 500
    labels = torch.randint(0, 2, (n,)).float()
    # 80% accurate but fixed high-magnitude logits, so confidence (~99.75%) far
    # exceeds actual accuracy: the textbook overconfident case T-scaling fixes.
    correct = torch.rand(n) < 0.8
    sign = torch.where(correct, labels * 2 - 1, -(labels * 2 - 1))
    logits = sign * 6.0

    criterion = nn.BCEWithLogitsLoss()
    nll_unscaled = float(criterion(logits, labels))

    T = fit_temperature(logits, labels)
    nll_scaled = float(criterion(logits / T, labels))

    assert T > 1.0
    assert nll_scaled < nll_unscaled


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # Same bin, confidence 0.5, half the labels positive: predicted matches observed.
    probs = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0)
