"""Unit tests for evaluation metrics, threshold calculation, and loss functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.density_strata import metrics_by_density
from src.evaluation.metrics import (
    evaluate,
    precision_recall_points,
    probability_metrics,
    threshold_at_specificity,
    youden_threshold,
)
from src.training.loss import SmoothedBCEWithLogitsLoss, make_criterion


def test_evaluate_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    panel = evaluate(y_true, y_prob, threshold=0.5)

    assert panel.auc == pytest.approx(1.0)
    assert panel.accuracy == pytest.approx(1.0)
    assert panel.sensitivity == pytest.approx(1.0)
    assert panel.specificity == pytest.approx(1.0)
    assert panel.ppv == pytest.approx(1.0)
    assert panel.npv == pytest.approx(1.0)


def test_evaluate_imperfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    panel = evaluate(y_true, y_prob, threshold=0.5)

    assert panel.accuracy == pytest.approx(0.5)
    assert panel.sensitivity == pytest.approx(0.5)
    assert panel.specificity == pytest.approx(0.5)


def test_youden_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thresh = youden_threshold(y_true, y_prob)
    assert 0.3 <= thresh <= 0.7


def test_threshold_at_specificity_uses_requested_operating_point():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.8, 0.3, 0.5, 0.7, 0.9])

    threshold = threshold_at_specificity(labels, probabilities, 0.75)
    panel = evaluate(labels, probabilities, threshold=threshold)

    assert panel.specificity >= 0.75
    assert panel.sensitivity == pytest.approx(0.75)


def test_probability_metrics_and_precision_recall():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    quality = probability_metrics(labels, probabilities)
    curve = precision_recall_points(labels, probabilities)

    assert quality.average_precision == pytest.approx(1.0)
    assert quality.brier_score < 0.05
    assert quality.negative_log_likelihood > 0.0
    assert curve["average_precision"] == pytest.approx(1.0)
    assert len(curve["precision"]) == len(curve["recall"])


def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate(np.array([0, 1]), np.array([0.2, 1.1]), threshold=0.5)


def test_metrics_by_density():
    df = pd.DataFrame(
        {
            "birads_density": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        }
    )
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.2, 0.8])
    res = metrics_by_density(df, y_prob, threshold=0.5, min_n=5)

    assert len(res) == 4
    d1 = res[res["density"] == 1].iloc[0]
    assert d1["n"] == 10
    assert d1["auc"] is not None
    assert pd.isna(d1["skipped_reason"])

    d2 = res[res["density"] == 2].iloc[0]
    assert d2["n"] == 2
    assert pd.isna(d2["auc"])
    assert d2["skipped_reason"] == "n<5"


def test_smoothed_bce_loss_forward():
    pos_weight = torch.tensor([2.0])
    criterion = SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=0.1)
    logits = torch.tensor([1.5, -1.5])
    targets = torch.tensor([1.0, 0.0])
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert float(loss.item()) > 0.0


def test_make_criterion(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"label": [0, 0, 0, 1]}).to_csv(csv_path, index=False)

    bce = make_criterion(csv_path, label_smoothing=0.0)
    assert isinstance(bce, torch.nn.BCEWithLogitsLoss)

    smoothed = make_criterion(csv_path, label_smoothing=0.1)
    assert isinstance(smoothed, SmoothedBCEWithLogitsLoss)
