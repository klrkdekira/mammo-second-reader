"""Build the shared probability and subgroup audit for each model."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.evaluation.calibration import (
    CALIBRATION_VERSION,
    expected_calibration_error,
    fit_temperature_with_diagnostics,
    reliability_bins,
)
from src.evaluation.decision_curve import decision_curve
from src.evaluation.density_strata import metrics_by_density
from src.evaluation.lesion_strata import metrics_by_lesion_type
from src.evaluation.metrics import (
    evaluate,
    precision_recall_points,
    probability_metrics,
    threshold_at_specificity,
)

FIXED_SPECIFICITY = 0.80


@dataclass(frozen=True)
class AuditOutput:
    record: dict[str, object]
    validation_probability: np.ndarray
    test_probability: np.ndarray
    validation_calibrated_probability: np.ndarray
    test_calibrated_probability: np.ndarray
    temperature: float
    fixed_specificity_target: float
    fixed_specificity_threshold: float


def logits_to_probability(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities without overflowing."""
    values = np.asarray(logits, dtype=float).ravel() / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def probability_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite logits."""
    values = np.clip(np.asarray(probabilities, dtype=float).ravel(), 1e-7, 1 - 1e-7)
    return np.log(values / (1.0 - values))


def _panel_dict(panel) -> dict[str, object]:
    return {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()}


def build_audit(
    validation_df: pd.DataFrame,
    validation_logits: np.ndarray,
    test_df: pd.DataFrame,
    test_logits: np.ndarray,
    *,
    operating_threshold: float,
    fixed_specificity: float = FIXED_SPECIFICITY,
) -> AuditOutput:
    """Create calibration, subgroup, PR and fixed-specificity records."""
    val_labels = np.asarray(validation_df["label"].to_numpy(), dtype=int)
    test_labels = np.asarray(test_df["label"].to_numpy(), dtype=int)
    val_logits = np.asarray(validation_logits, dtype=float).ravel()
    test_logits = np.asarray(test_logits, dtype=float).ravel()
    if len(validation_df) != val_logits.size or len(test_df) != test_logits.size:
        raise ValueError("Prediction counts do not match their manifests.")

    val_probability = logits_to_probability(val_logits)
    test_probability = logits_to_probability(test_logits)
    fit = fit_temperature_with_diagnostics(
        torch.tensor(val_logits, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.float32),
    )
    val_calibrated = logits_to_probability(val_logits, fit.temperature)
    test_calibrated = logits_to_probability(test_logits, fit.temperature)
    centres, pred_mean, obs_mean = reliability_bins(test_calibrated, test_labels)

    fixed_threshold = threshold_at_specificity(
        val_labels, val_probability, fixed_specificity
    )
    raw_quality = probability_metrics(test_labels, test_probability)
    calibrated_quality = probability_metrics(test_labels, test_calibrated)
    record: dict[str, object] = {
        "density_strata": metrics_by_density(
            test_df, test_probability, operating_threshold
        ).to_dict(orient="records"),
        "calibration": {
            "version": CALIBRATION_VERSION,
            "temperature": fit.temperature,
            "fit": fit.to_dict(),
            "ece_before": expected_calibration_error(test_probability, test_labels),
            "ece_after": expected_calibration_error(test_calibrated, test_labels),
            "reliability": {
                "bin_centre": centres.tolist(),
                "pred_mean": pred_mean.tolist(),
                "obs_mean": obs_mean.tolist(),
            },
        },
        "probability_metrics": {
            "raw": dataclasses.asdict(raw_quality),
            "calibrated": dataclasses.asdict(calibrated_quality),
        },
        "precision_recall": precision_recall_points(test_labels, test_probability),
        "decision_curve": {
            key: np.asarray(value).tolist()
            for key, value in decision_curve(test_labels, test_calibrated).items()
        },
        "fixed_specificity": {
            "target": fixed_specificity,
            "threshold_source": "validation",
            "threshold": fixed_threshold,
            "validation": _panel_dict(
                evaluate(val_labels, val_probability, threshold=fixed_threshold)
            ),
            "test": _panel_dict(
                evaluate(test_labels, test_probability, threshold=fixed_threshold)
            ),
        },
    }
    if "lesion_type" in test_df.columns:
        record["lesion_strata"] = metrics_by_lesion_type(
            test_df, test_probability, operating_threshold
        ).to_dict(orient="records")

    return AuditOutput(
        record=record,
        validation_probability=val_probability,
        test_probability=test_probability,
        validation_calibrated_probability=val_calibrated,
        test_calibrated_probability=test_calibrated,
        temperature=fit.temperature,
        fixed_specificity_target=fixed_specificity,
        fixed_specificity_threshold=fixed_threshold,
    )
