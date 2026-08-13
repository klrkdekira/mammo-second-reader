"""Tests for the external-evaluation web page."""

import json

import pytest

from src.web.external import (
    headline_table,
    load_result,
    locked_markdown,
    readiness,
    readiness_markdown,
    roc_points,
    run_cold_evaluation,
    strata_table,
    summary_markdown,
)

_RESULT = {
    "dataset": "inbreast",
    "protocol": {"label_construct": "BI-RADS assessment, not pathology"},
    "locked_operating_point": {
        "source_run": "vgg16_imagenet_448",
        "youden_threshold": 0.4846,
        "fixed_specificity_target": 0.8,
        "fixed_specificity_threshold": 0.6005,
        "temperature": 1.1225,
        "checkpoint_sha256": "727f9e406b192e1b0baf0bb6d448da5d221aa81",
        "refitted_on_external": False,
    },
    "subsets": {
        "full": {
            "n_cases": 410,
            "n_patients": 108,
            "n_malignant": 100,
            "prevalence": 0.2439,
            "test": {"auc": 0.61, "sensitivity": 0.7, "specificity": 0.45},
            "roc": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.7, 1.0]},
            "calibration": {"ece_before": 0.12, "ece_after": 0.13},
            "fixed_specificity": {
                "target": 0.8,
                "threshold": 0.6005,
                "achieved_specificity": 0.71,
            },
            "density_strata": [
                {"density": 1, "n": 136, "auc": 0.63, "skipped_reason": None},
                {"density": 4, "n": 28, "auc": None, "skipped_reason": "n<10"},
            ],
            "lesion_strata": [
                {"lesion_type": "mass", "n": 28, "auc": 0.6, "skipped_reason": None}
            ],
            "intervals": {
                "metrics": {
                    "auc": {"estimate": 0.61, "ci_lower": 0.55, "ci_upper": 0.67},
                    "sensitivity": {"estimate": 0.7, "ci_lower": 0.6, "ci_upper": 0.8},
                }
            },
        }
    },
}


@pytest.fixture
def result_file(tmp_path):
    path = tmp_path / "metrics-inbreast.json"
    path.write_text(json.dumps(_RESULT))
    return path


def test_readiness_reports_missing_inputs_without_raising(tmp_path):
    state = readiness(tmp_path / "absent.toml", tmp_path / "absent.json")

    assert state.can_run is False
    assert state.has_result is False
    assert state.blocking, "a missing config must surface as a failed check"
    assert "External config" in readiness_markdown(state)


def test_readiness_detects_an_existing_result(result_file, tmp_path):
    state = readiness(tmp_path / "absent.toml", result_file)

    assert state.has_result is True
    assert "already been run" in readiness_markdown(state)


def test_readiness_against_the_shipped_config_does_not_raise():
    """Locally the INbreast data is absent; the page must still render."""
    from pathlib import Path

    state = readiness(Path("configs/inbreast_external.toml"))

    labels = [check.label for check in state.checks]
    assert "External config" in labels
    assert "Locked operating point" in labels
    assert isinstance(readiness_markdown(state), str)


def test_load_result_explains_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No cold external result"):
        load_result(tmp_path / "nope.json")


def test_load_result_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    with pytest.raises(ValueError, match="Cannot read"):
        load_result(path)


def test_locked_markdown_shows_that_nothing_was_refitted(result_file):
    text = locked_markdown(load_result(result_file))

    assert "vgg16_imagenet_448" in text
    assert "False" in text, "the no-refit guarantee must be visible"
    assert "727f9e406b192e1b" in text


def test_headline_table_pairs_estimates_with_intervals(result_file):
    table = headline_table(load_result(result_file), "full")

    row = table[table["Metric"] == "AUC"].iloc[0]
    assert row["Estimate"] == "0.6100"
    assert row["95% CI"] == "0.5500 to 0.6700"
    # Metrics absent from the record render as placeholders, not crashes.
    missing = table[table["Metric"] == "Brier score"].iloc[0]
    assert missing["Estimate"] == "N/A"


def test_summary_names_the_specificity_gap_as_a_result(result_file):
    text = summary_markdown(load_result(result_file), "full")

    assert "410 images" in text
    assert "108" in text
    assert "71.0%" in text
    assert "not an error" in text


def test_strata_table_keeps_skip_reasons_when_any_row_was_skipped(result_file):
    table = strata_table(load_result(result_file), "full", "density_strata")

    assert "skipped_reason" in table.columns
    assert (table["skipped_reason"] == "n<10").any()


def test_strata_table_handles_an_absent_stratum(result_file):
    table = strata_table(load_result(result_file), "full", "gradcam_strata")

    assert "note" in table.columns


def test_roc_points_round_trip(result_file):
    fpr, tpr = roc_points(load_result(result_file), "full")

    assert fpr == [0.0, 0.5, 1.0]
    assert tpr == [0.0, 0.7, 1.0]


def test_run_refuses_without_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="one-shot pre-registered test"):
        run_cold_evaluation(
            tmp_path / "cfg.toml", acknowledged=False, result_path=tmp_path / "out.json"
        )


def test_run_refuses_when_inputs_are_not_ready(tmp_path):
    with pytest.raises(ValueError, match="inputs are not ready"):
        run_cold_evaluation(
            tmp_path / "cfg.toml", acknowledged=True, result_path=tmp_path / "out.json"
        )
