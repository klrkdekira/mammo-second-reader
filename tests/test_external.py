"""Tests for the external evaluation protocol."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.external import (
    LockedOperatingPoint,
    _align_subset_logits,
    external_audit,
    load_external_config,
    load_locked_operating_point,
    subset_names,
)

_TEMPERATURE = 1.25
_YOUDEN = 0.4846
_FIXED = 0.6005


def _locked() -> LockedOperatingPoint:
    return LockedOperatingPoint(
        source_run="vgg16_imagenet_448",
        youden_threshold=_YOUDEN,
        fixed_specificity_target=0.8,
        fixed_specificity_threshold=_FIXED,
        temperature=_TEMPERATURE,
        checkpoint_sha256="deadbeef",
    )


def _frame(n: int = 40) -> pd.DataFrame:
    """Two images per patient, with the label constant within a patient.

    The patient-level bootstrap strata on `label.max()` per patient, so a
    fixture that gives every patient one malignant image leaves the benign
    stratum empty and cannot be resampled.
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "patient_id": [f"p{i // 2}" for i in range(n)],
            "label": [(i // 2) % 2 for i in range(n)],
            "birads_density": rng.integers(1, 5, size=n),
            "lesion_type": ["mass" if i % 3 else "calcification" for i in range(n)],
        }
    )


def _metrics_file(tmp_path, *, threshold=_YOUDEN, temperature=_TEMPERATURE):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "vgg16_imagenet_448",
                        "val_threshold": threshold,
                        "calibration": {"temperature": temperature},
                        "fixed_specificity": {"target": 0.8, "threshold": _FIXED},
                    }
                ]
            }
        )
    )
    return path


def _sidecar(tmp_path, *, youden=_YOUDEN):
    path = tmp_path / "vgg16_imagenet_448.threshold.json"
    path.write_text(json.dumps({"youden_j": youden}))
    return path


def _checkpoint(tmp_path):
    path = tmp_path / "vgg16_imagenet_448.pt"
    path.write_bytes(b"weights")
    return path


def test_locked_operating_point_is_read_from_the_frozen_record(tmp_path):
    locked = load_locked_operating_point(
        metrics_path=_metrics_file(tmp_path),
        run_name="vgg16_imagenet_448",
        checkpoint_path=_checkpoint(tmp_path),
        threshold_sidecar=_sidecar(tmp_path),
    )

    assert locked.youden_threshold == _YOUDEN
    assert locked.fixed_specificity_threshold == _FIXED
    assert locked.temperature == _TEMPERATURE
    assert locked.to_dict()["refitted_on_external"] is False
    assert locked.to_dict()["fitted_on"] == "cbis_ddsm_validation_fold"


def test_threshold_drift_between_metrics_and_sidecar_stops_the_run(tmp_path):
    with pytest.raises(ValueError, match="Threshold drift"):
        load_locked_operating_point(
            metrics_path=_metrics_file(tmp_path, threshold=0.5),
            run_name="vgg16_imagenet_448",
            checkpoint_path=_checkpoint(tmp_path),
            threshold_sidecar=_sidecar(tmp_path, youden=0.4),
        )


def test_missing_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no frozen record"):
        load_locked_operating_point(
            metrics_path=_metrics_file(tmp_path),
            run_name="some_other_run",
            checkpoint_path=_checkpoint(tmp_path),
            threshold_sidecar=_sidecar(tmp_path),
        )


def test_non_positive_temperature_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-positive temperature"):
        load_locked_operating_point(
            metrics_path=_metrics_file(tmp_path, temperature=0.0),
            run_name="vgg16_imagenet_448",
            checkpoint_path=_checkpoint(tmp_path),
            threshold_sidecar=_sidecar(tmp_path),
        )


def test_audit_uses_the_transferred_thresholds_unchanged():
    frame = _frame()
    logits = np.linspace(-2.0, 2.0, len(frame))

    record = external_audit(frame, logits, _locked()).record

    assert record["test"]["threshold"] == pytest.approx(_YOUDEN)
    assert record["fixed_specificity"]["threshold"] == pytest.approx(_FIXED)
    assert (
        record["fixed_specificity"]["threshold_source"]
        == "internal_validation_fold_transferred_unchanged"
    )
    assert record["calibration"]["temperature"] == pytest.approx(_TEMPERATURE)


def test_audit_reports_achieved_specificity_rather_than_the_target():
    frame = _frame()
    # Push every probability below the fixed threshold so the achieved
    # specificity is 1.0 and cannot coincide with the 0.8 target by luck.
    record = external_audit(frame, np.full(len(frame), -8.0), _locked()).record

    fixed = record["fixed_specificity"]
    assert fixed["target"] == 0.8
    assert fixed["achieved_specificity"] == pytest.approx(1.0)
    assert fixed["achieved_specificity"] != fixed["target"]


def test_audit_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="do not match the manifest"):
        external_audit(_frame(), np.zeros(3), _locked())


def test_audit_reports_subset_shape_and_prevalence():
    frame = _frame(40)

    record = external_audit(frame, np.linspace(-1, 1, 40), _locked()).record

    assert record["n_cases"] == 40
    assert record["n_patients"] == 20
    assert record["n_malignant"] == 20
    assert record["prevalence"] == pytest.approx(0.5)


def test_subset_logits_are_selected_from_the_full_inference():
    full = _frame(40)
    logits = np.linspace(-2.0, 2.0, len(full))
    subset = full.iloc[[17, 2, 31, 8]].reset_index(drop=True)

    selected = _align_subset_logits(full, logits, subset)

    assert np.array_equal(selected, logits[[17, 2, 31, 8]])


def test_subset_logits_reject_images_absent_from_the_full_manifest():
    full = _frame(40)
    subset = full.iloc[:2].copy()
    subset.loc[subset.index[0], "image_id"] = "not-in-full"

    with pytest.raises(ValueError, match="absent from full"):
        _align_subset_logits(full, np.zeros(len(full)), subset)


def test_external_config_round_trip(tmp_path):
    path = tmp_path / "external.toml"
    path.write_text(
        """
run_name = "run__inbreast_cold"
internal_config = "configs/vgg16_highres_448.toml"
internal_run = "vgg16_imagenet_448"

[data]
manifest = "data/inbreast/manifest/test.csv"
lesion_present_manifest = "data/inbreast/manifest/test_lesion_present.csv"
manifest_lock = "data/inbreast/manifest/manifest-lock.json"
image_root = "data/inbreast/cache_448"
image_size = 448
"""
    )

    config = load_external_config(path)

    assert config.internal_run == "vgg16_imagenet_448"
    assert config.image_size == 448
    assert config.lesion_present_manifest is not None
    assert config.dataset == "inbreast"


def test_external_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "external.toml"
    path.write_text(
        """
run_name = "r"
internal_config = "c"
internal_run = "i"
epochs = 5

[data]
manifest = "m"
image_root = "r"
"""
    )

    with pytest.raises(ValueError, match="Unknown key"):
        load_external_config(path)


def test_external_config_requires_a_manifest(tmp_path):
    path = tmp_path / "external.toml"
    path.write_text(
        """
run_name = "r"
internal_config = "c"
internal_run = "i"

[data]
image_root = "r"
"""
    )

    with pytest.raises(ValueError, match="manifest"):
        load_external_config(path)


def test_shipped_config_never_writes_the_internal_evidence_file():
    """The internal version 3 freeze must survive a cold external run."""
    from src.evaluation.external import DEFAULT_OUTPUT, DEFAULT_PREDICTIONS_DIR

    assert DEFAULT_OUTPUT != __import__("pathlib").Path("results/metrics.json")
    assert str(DEFAULT_OUTPUT).startswith("results/external")
    assert str(DEFAULT_PREDICTIONS_DIR).startswith("results/external")


def test_shipped_config_points_at_the_promoted_candidate():
    config = load_external_config(
        __import__("pathlib").Path("configs/inbreast_external.toml")
    )

    assert config.internal_run == "vgg16_imagenet_448"
    assert config.image_size == 448
    assert str(config.image_root) == "data/inbreast/cache_448"
    assert config.manifest_lock is not None, (
        "the manifest lock must be provenance-linked"
    )


def test_external_predictions_feed_the_internal_bootstrap_unchanged(tmp_path):
    """The written prediction file must satisfy the internal statistics reader.

    External intervals are only comparable with internal ones if they come from
    the same estimator, so the prediction schema has to survive `read_predictions`
    without any external-specific relaxation.
    """
    from src.evaluation.predictions import (
        build_prediction_frame,
        write_predictions_atomic,
    )
    from src.evaluation.statistics import model_intervals, read_predictions

    frame = _frame(40)
    locked = _locked()
    audit = external_audit(frame, np.linspace(-2.0, 2.0, len(frame)), locked)

    path = tmp_path / "run__inbreast_cold.test.csv"
    write_predictions_atomic(
        build_prediction_frame(
            frame,
            run_name="run__inbreast_cold",
            split="test",
            logits=np.linspace(-2.0, 2.0, len(frame)),
            probabilities=audit.probability,
            calibrated_probabilities=audit.calibrated_probability,
            threshold=locked.youden_threshold,
            fixed_specificity_target=locked.fixed_specificity_target,
            fixed_specificity_threshold=locked.fixed_specificity_threshold,
            seed=42,
            checkpoint_sha256=locked.checkpoint_sha256,
        ),
        path,
    )

    intervals = model_intervals(read_predictions(path), n_resamples=25, seed=1)

    assert intervals["n_cases"] == 40
    assert intervals["n_patients"] == 20
    assert intervals["threshold"] == pytest.approx(locked.youden_threshold)
    auc = intervals["metrics"]["auc"]
    assert auc["ci_lower"] <= auc["estimate"] <= auc["ci_upper"]
    # The subgroup safeguards used internally must also be computable externally.
    for name in (
        "calcification_sensitivity_at_fixed_specificity",
        "dense_breast_sensitivity_at_fixed_specificity",
    ):
        assert name in intervals["metrics"]


def test_subset_names_survive_a_widened_subset_tuple():
    """Regression: the payload once unpacked two elements from four-tuples.

    `subsets` carries (name, manifest_path, frame, logits). An earlier revision
    widened it from two elements and left the provenance comprehension at two,
    so `run_external_evaluation` raised ValueError after inference but before
    writing the metrics. Any future widening must keep this passing.
    """
    two = [("full", Path("a.csv")), ("lesion_present", Path("b.csv"))]
    four = [
        ("full", Path("a.csv"), object(), object()),
        ("lesion_present", Path("b.csv"), object(), object()),
    ]

    assert subset_names(two) == ["full", "lesion_present"]
    assert subset_names(four) == ["full", "lesion_present"]


def test_subset_names_preserves_evaluation_order():
    assert subset_names([("lesion_present",), ("full",)]) == ["lesion_present", "full"]
