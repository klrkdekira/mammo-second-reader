"""Tests for saved case-level predictions."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.predictions import (
    build_prediction_frame,
    prediction_path,
    write_predictions_atomic,
)


def _manifest():
    return pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "patient_id": ["p1", "p2"],
            "birads_density": [1, 4],
            "lesion_type": ["mass", "calcification"],
            "label": [0, 1],
        }
    )


def test_build_and_write_prediction_frame(tmp_path):
    frame = build_prediction_frame(
        _manifest(),
        run_name="model",
        split="test",
        logits=np.array([-1.0, 1.0]),
        probabilities=np.array([0.2, 0.8]),
        calibrated_probabilities=np.array([0.3, 0.7]),
        threshold=0.5,
        fixed_specificity_target=0.8,
        fixed_specificity_threshold=0.6,
        seed=42,
        checkpoint_sha256="abc",
    )
    path = tmp_path / "predictions" / "model.test.csv"
    write_predictions_atomic(frame, path)
    saved = pd.read_csv(path)

    assert saved["image_id"].tolist() == ["a", "b"]
    assert saved["predicted_label"].tolist() == [0, 1]
    assert saved["patient_id"].tolist() == ["p1", "p2"]
    assert prediction_path("model", "test").name == "model.test.csv"


def test_prediction_frame_requires_patient_ids():
    manifest = _manifest().drop(columns="patient_id")
    with pytest.raises(ValueError, match="patient_id"):
        build_prediction_frame(
            manifest,
            run_name="model",
            split="test",
            logits=np.array([-1.0, 1.0]),
            probabilities=np.array([0.2, 0.8]),
            calibrated_probabilities=np.array([0.3, 0.7]),
            threshold=0.5,
            fixed_specificity_target=0.8,
            fixed_specificity_threshold=0.6,
            seed=42,
            checkpoint_sha256="abc",
        )
