"""Canonical metrics. See WARNINGS §4.

On this roughly 1.4:1 (benign:malignant) imbalanced dataset, accuracy is a
weak headline scalar (the majority-class floor is already about 59 percent),
so AUC, sensitivity, specificity, and PPV stay the primary reporting set, the
same one every paper this project benchmarks against uses (Wang 2024 Tables 4
and 5, Shen 2019). Accuracy is still computed and stored alongside them for
completeness; just don't read it as the headline.

Everything in this project routes through `evaluate()`. No duplicate
metric definitions are permitted anywhere else.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricPanel:
    auc: float
    accuracy: float
    sensitivity: float  # recall / true-positive rate
    specificity: float  # true-negative rate
    ppv: float  # precision / positive predictive value
    npv: float
    f1: float
    threshold: float
    confusion: np.ndarray


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J: argmax over (TPR - FPR). Returns a probability cutoff."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None
) -> MetricPanel:
    """Compute the full metric panel at the given (or Youden) threshold."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    return MetricPanel(
        auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float((tp + tn) / total) if total else 0.0,
        sensitivity=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        ppv=tp / (tp + fp) if (tp + fp) else 0.0,
        npv=tn / (tn + fn) if (tn + fn) else 0.0,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        confusion=np.array([[tn, fp], [fn, tp]], dtype=int),
    )
