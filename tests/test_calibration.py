"""Unit tests for temperature scaling and calibration metrics."""

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_with_diagnostics,
    reliability_bins,
)


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


def test_internal_boundary_case_is_counted_once():
    probs = np.array([0.5])
    labels = np.array([0])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.5)
    _, pred, obs = reliability_bins(probs, labels, n_bins=10)
    np.testing.assert_allclose(pred, [0.5])
    np.testing.assert_allclose(obs, [0.0])


def test_probability_one_is_included_in_final_bin():
    centres, pred, obs = reliability_bins(np.array([1.0]), np.array([1]), n_bins=10)
    assert centres.tolist() == pytest.approx([0.95])
    assert pred.tolist() == pytest.approx([1.0])
    assert obs.tolist() == pytest.approx([1.0])


def test_temperature_parameterisation_is_always_positive():
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.raw_temperature.fill_(-1e6)
    assert float(scaler.temperature.item()) > 0.0


def test_temperature_fit_records_diagnostics():
    logits = torch.tensor([-4.0, -4.0, 4.0, 4.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    fit = fit_temperature_with_diagnostics(logits, labels)
    assert fit.temperature > 0.0
    assert fit.finite is True
    assert fit.improved is True
    assert fit.final_nll <= fit.initial_nll
    assert fit.function_evaluations > 0


@pytest.mark.parametrize(
    ("probs", "labels", "message"),
    [
        (np.array([]), np.array([]), "at least one"),
        (np.array([1.2]), np.array([1]), r"\[0, 1\]"),
        (np.array([0.5]), np.array([2]), "only 0 and 1"),
    ],
)
def test_calibration_rejects_invalid_inputs(probs, labels, message):
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(probs, labels)
