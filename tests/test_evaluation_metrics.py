"""Unit tests for evaluation metrics, threshold calculation, and loss functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.density_strata import metrics_by_density
from src.evaluation.metrics import evaluate, youden_threshold
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
    # Density 1 has n=10 >= min_n=5
    d1 = res[res["density"] == 1].iloc[0]
    assert d1["n"] == 10
    assert d1["auc"] is not None
    assert pd.isna(d1["skipped_reason"])

    # Density 2 has n=2 < min_n=5
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
