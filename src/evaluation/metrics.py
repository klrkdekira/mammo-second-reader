"""Canonical metrics. See WARNINGS §4.

On this roughly 1.4:1 (benign:malignant) imbalanced dataset, accuracy is a
weak headline scalar (the majority-class floor is already about 59 percent),
so AUC, sensitivity, specificity, and PPV stay the primary reporting set, the
same one every paper this project benchmarks against uses (Wang 2024 Tables 4
and 5, Shen 2019). Accuracy is still computed and stored alongside them for
completeness. Just don't read it as the headline.

Everything in this project routes through `evaluate()`. No duplicate
metric definitions are permitted anywhere else.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
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


@dataclass(frozen=True)
class ProbabilityPanel:
    average_precision: float
    brier_score: float
    negative_log_likelihood: float


def _validated_arrays(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel().astype(int)
    probabilities = np.asarray(y_prob).ravel().astype(float)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError(
            "Labels and probabilities must be non-empty and the same size."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must contain only 0 and 1.")
    return labels, probabilities


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J: argmax over (TPR - FPR). Returns a probability cutoff."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    fpr, tpr, thr = roc_curve(labels, probabilities)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def threshold_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float = 0.8
) -> float:
    """Choose a validation threshold that reaches the requested specificity."""
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be between 0 and 1.")
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if np.unique(labels).size < 2:
        raise ValueError("Both classes are required to choose a threshold.")
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.flatnonzero(
        (fpr <= 1.0 - target_specificity + 1e-12) & np.isfinite(thresholds)
    )
    if valid.size == 0:
        raise ValueError("No finite threshold reaches the requested specificity.")
    best_tpr = np.max(tpr[valid])
    tied = valid[np.isclose(tpr[valid], best_tpr)]
    return float(np.min(thresholds[tied]))


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ProbabilityPanel:
    """Summarise ranking and probability quality."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    return ProbabilityPanel(
        average_precision=float(average_precision_score(labels, probabilities)),
        brier_score=float(brier_score_loss(labels, probabilities)),
        negative_log_likelihood=float(
            log_loss(labels, np.clip(probabilities, 1e-7, 1.0 - 1e-7), labels=[0, 1])
        ),
    )


def precision_recall_points(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, list[float] | float]:
    """Return the precision-recall curve and average precision."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None
) -> MetricPanel:
    """Compute performance metrics at given or Youden's J threshold.

    Pass a validation-derived threshold for test set evaluation to prevent leakage.
    """
    y_true, y_prob = _validated_arrays(y_true, y_prob)
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


@dataclass(frozen=True)
class PatchMetricPanel:
    """Stage 0 five-class patch panel.

    This is the single five-class definition in the project, kept here so the
    module rule above still holds. It redefines none of the binary metrics: the
    patch task has no single positive class and no operating threshold, so
    `evaluate()` cannot express it.

    `lesion_pair_confusion` answers the question the plan asks of the patch
    diagnostic: within true mass patches (and separately true calcification
    patches), how often does the model keep the lesion type but flip
    malignancy, versus leave the pair altogether.
    """

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_sensitivity: dict[str, float]
    per_class_precision: dict[str, float]
    confusion: np.ndarray
    lesion_pair_confusion: dict[str, dict[str, float]]


_LESION_PAIRS: dict[str, tuple[str, str]] = {
    "calcification": ("benign_calcification", "malignant_calcification"),
    "mass": ("benign_mass", "malignant_mass"),
}


def _validated_labels(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true).ravel().astype(int)
    predicted = np.asarray(y_pred).ravel().astype(int)
    if true.size == 0 or true.shape != predicted.shape:
        raise ValueError("Labels and predictions must be non-empty and the same size.")
    for name, array in (("labels", true), ("predictions", predicted)):
        if not np.all((array >= 0) & (array < n_classes)):
            raise ValueError(f"Patch {name} must be class ids in [0, {n_classes}).")
    return true, predicted


def evaluate_patches(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> PatchMetricPanel:
    """Score five-class patch predictions against their frozen class ids."""
    n_classes = len(class_names)
    true, predicted = _validated_labels(y_true, y_pred, n_classes)
    labels = list(range(n_classes))

    matrix = confusion_matrix(true, predicted, labels=labels)
    support = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where(support > 0, diagonal / np.maximum(support, 1), np.nan)
        precision = np.where(
            predicted_counts > 0, diagonal / np.maximum(predicted_counts, 1), np.nan
        )

    pair_confusion: dict[str, dict[str, float]] = {}
    for lesion_type, (benign_name, malignant_name) in _LESION_PAIRS.items():
        if benign_name not in class_names or malignant_name not in class_names:
            continue
        pair_ids = [class_names.index(benign_name), class_names.index(malignant_name)]
        in_true_pair = np.isin(true, pair_ids)
        n_true = int(in_true_pair.sum())
        in_pair = in_true_pair & np.isin(predicted, pair_ids)
        n_in_pair = int(in_pair.sum())
        n_correct = int((in_pair & (true == predicted)).sum())
        pair_confusion[lesion_type] = {
            "n_true": float(n_true),
            "n_predicted_in_pair": float(n_in_pair),
            "n_malignancy_correct": float(n_correct),
            "n_malignancy_flipped": float(n_in_pair - n_correct),
            "n_predicted_off_pair": float(n_true - n_in_pair),
            "malignancy_accuracy_in_pair": (
                float(n_correct / n_in_pair) if n_in_pair else float("nan")
            ),
        }

    return PatchMetricPanel(
        accuracy=float((true == predicted).mean()),
        balanced_accuracy=float(balanced_accuracy_score(true, predicted)),
        macro_f1=float(f1_score(true, predicted, labels=labels, average="macro")),
        per_class_sensitivity={
            name: float(value) for name, value in zip(class_names, sensitivity)
        },
        per_class_precision={
            name: float(value) for name, value in zip(class_names, precision)
        },
        confusion=matrix,
        lesion_pair_confusion=pair_confusion,
    )
