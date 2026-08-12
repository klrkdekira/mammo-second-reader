"""Tests for patient-level uncertainty and paired comparisons."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistics import (
    generate_statistics,
    model_intervals,
    paired_comparison,
    patient_stratified_samples,
    seed_repeat_summary,
)


def _frame(name="model", seed=42, probabilities=None):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    if probabilities is None:
        probabilities = np.array([0.1, 0.2, 0.4, 0.6, 0.3, 0.7, 0.8, 0.9])
    return pd.DataFrame(
        {
            "run_name": name,
            "split": "test",
            "image_id": [f"i{index}" for index in range(8)],
            "patient_id": ["n1", "n1", "n2", "n2", "p1", "p1", "p2", "p2"],
            "birads_density": [1, 1, 2, 2, 3, 3, 4, 4],
            "lesion_type": ["mass"] * 4 + ["calcification"] * 4,
            "label": labels,
            "logit": np.log(probabilities / (1.0 - probabilities)),
            "probability": probabilities,
            "calibrated_probability": probabilities,
            "threshold": 0.5,
            "fixed_specificity_target": 0.8,
            "fixed_specificity_threshold": 0.6,
            "seed": seed,
            "checkpoint_sha256": "abc",
        }
    )


def test_patient_bootstrap_keeps_patient_rows_together():
    frame = _frame()
    sample = next(patient_stratified_samples(frame, n_resamples=1, seed=3))

    for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert np.count_nonzero(sample == first) == np.count_nonzero(sample == second)


def test_model_intervals_are_deterministic():
    first = model_intervals(_frame(), n_resamples=30, seed=8)
    second = model_intervals(_frame(), n_resamples=30, seed=8)

    assert first == second
    auc = first["metrics"]["auc"]
    assert auc["ci_lower"] <= auc["estimate"] <= auc["ci_upper"]


def test_paired_comparison_uses_matched_cases():
    better = _frame(probabilities=np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
    worse = _frame(probabilities=np.array([0.8, 0.6, 0.4, 0.2, 0.7, 0.5, 0.3, 0.1]))
    result = paired_comparison(better, worse, n_resamples=40, seed=5)

    assert result["first_minus_second"]["auc"]["estimate"] > 0
    mismatched = worse.copy()
    mismatched.loc[0, "patient_id"] = "different"
    with pytest.raises(ValueError, match="patient_id"):
        paired_comparison(better, mismatched, n_resamples=5)


def test_generate_statistics_and_seed_summary(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    names = ["model", "model_seed7", "model_seed2026"]
    seeds = [42, 7, 2026]
    for name, seed in zip(names, seeds):
        _frame(name, seed).to_csv(predictions / f"{name}.test.csv", index=False)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"runs": [{"model": name} for name in names]}))
    output = tmp_path / "statistics.json"

    result = generate_statistics(
        metrics,
        predictions,
        output,
        n_resamples=20,
        seed=2,
        comparisons=(),
    )

    assert set(result["models"]) == set(names)
    assert result["seed_repeats"]["model"]["n_seeds"] == 3
    assert json.loads(output.read_text())["method"]["unit"] == "patient"
    assert seed_repeat_summary({"model": _frame()}) == {}
