"""Tests for evidence lineage and snapshot freezing."""

import json

import pytest

from src.evaluation.freeze import freeze_evidence
from src.evaluation.provenance import build_run_provenance, sha256_file


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("evidence\n")
    before = sha256_file(path)
    assert before == sha256_file(path)
    path.write_text("changed\n")
    assert sha256_file(path) != before


def test_build_run_provenance_hashes_inputs(tmp_path):
    config = tmp_path / "run.toml"
    checkpoint = tmp_path / "model.pt"
    split = tmp_path / "test.csv"
    threshold = tmp_path / "model.threshold.json"
    predictions = tmp_path / "model.test.csv"
    config.write_text("seed = 42\n")
    checkpoint.write_bytes(b"weights")
    split.write_text("image_id,label\na,0\n")
    threshold.write_text('{"youden_j": 0.5}\n')
    predictions.write_text("image_id,probability\na,0.2\n")

    result = build_run_provenance(
        config_path=config,
        checkpoint_paths=[checkpoint],
        manifest_paths=[split],
        threshold_path=threshold,
        prediction_paths=[predictions],
        extra={"seed": 42},
    )

    assert result["config"]["sha256"] == sha256_file(config)
    assert result["checkpoints"][0]["sha256"] == sha256_file(checkpoint)
    assert result["manifests"][0]["sha256"] == sha256_file(split)
    assert result["prediction_files"][0]["sha256"] == sha256_file(predictions)
    assert result["experiment"]["seed"] == 42
    assert result["code"]["preprocessing_fingerprint"]


def test_build_run_provenance_includes_run_specific_code(tmp_path):
    config = tmp_path / "run.toml"
    checkpoint = tmp_path / "model.pt"
    split = tmp_path / "test.csv"
    preprocessing = tmp_path / "custom_ingest.py"
    evaluation = tmp_path / "custom_evaluation.py"
    config.write_text("seed = 42\n")
    checkpoint.write_bytes(b"weights")
    split.write_text("image_id,label\na,0\n")
    preprocessing.write_text("INGEST_VERSION = 1\n")
    evaluation.write_text("EVALUATION_VERSION = 1\n")

    result = build_run_provenance(
        config_path=config,
        checkpoint_paths=[checkpoint],
        manifest_paths=[split],
        additional_preprocessing_paths=[preprocessing],
        additional_evaluation_paths=[evaluation],
    )

    preprocessing_files = result["code"]["preprocessing_files"]
    evaluation_files = result["code"]["evaluation_files"]
    assert any(
        item["sha256"] == sha256_file(preprocessing) for item in preprocessing_files
    )
    assert any(item["sha256"] == sha256_file(evaluation) for item in evaluation_files)


def _run(
    model: str,
    evidence,
    *,
    commit="abc",
    dirty=False,
    preprocessing="p",
    evaluation="e",
):
    descriptor = {
        "path": str(evidence.resolve()),
        "exists": True,
        "sha256": sha256_file(evidence),
    }
    return {
        "model": model,
        "calibration": {},
        "decision_curve": {},
        "density_strata": [],
        "fixed_specificity": {"density_strata": [], "lesion_strata": []},
        "lesion_strata": [],
        "precision_recall": {},
        "probability_metrics": {},
        "provenance": {
            "version": 3,
            "git": {"commit": commit, "dirty_evidence_files": dirty},
            "config": descriptor,
            "checkpoints": [descriptor],
            "manifests": [descriptor],
            "threshold_sidecar": descriptor,
            "prediction_files": [descriptor, descriptor],
            "code": {
                "preprocessing_fingerprint": preprocessing,
                "evaluation_fingerprint": evaluation,
                "preprocessing_files": [descriptor],
                "evaluation_files": [descriptor],
            },
        },
    }


def _statistics(path, models, evidence=None):
    descriptor = (
        {
            "path": str(evidence.resolve()),
            "exists": True,
            "sha256": sha256_file(evidence),
        }
        if evidence is not None
        else {"path": str(path.resolve()), "exists": False, "sha256": ""}
    )
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "method": {"unit": "patient", "n_resamples": 2000},
                "models": {model: {} for model in models},
                "prediction_files": {model: descriptor for model in models},
                "paired_comparisons": {
                    "vgg16_imagenet_minus_vgg16_scratch": {},
                    "ensemble_minus_vgg16_imagenet": {},
                },
            }
        )
        + "\n"
    )


def test_freeze_evidence_requires_homogeneous_clean_provenance(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    statistics = tmp_path / "statistics.json"
    _statistics(statistics, ["a", "b"], evidence)
    metrics.write_text(
        json.dumps({"runs": [_run("a", evidence), _run("b", evidence)]}) + "\n"
    )

    result = freeze_evidence(metrics, output, statistics_path=statistics)

    assert result["n_runs"] == 2
    assert result["git_commit"] == "abc"
    assert result["statistics"]["sha256"] == sha256_file(statistics)
    assert json.loads(output.read_text())["models"] == ["a", "b"]


def test_freeze_evidence_rejects_legacy_or_dirty_runs(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    metrics.write_text(json.dumps({"runs": [{"model": "legacy"}]}) + "\n")
    statistics = tmp_path / "statistics.json"
    _statistics(statistics, ["legacy"])
    with pytest.raises(ValueError, match="missing provenance"):
        freeze_evidence(metrics, output, statistics_path=statistics)

    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    metrics.write_text(
        json.dumps({"runs": [_run("dirty", evidence, dirty=True)]}) + "\n"
    )
    _statistics(statistics, ["dirty"], evidence)
    with pytest.raises(ValueError, match="dirty evidence"):
        freeze_evidence(metrics, output, statistics_path=statistics)


def test_freeze_evidence_requires_case_level_predictions(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    statistics = tmp_path / "statistics.json"
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    run = _run("model", evidence)
    run["provenance"]["prediction_files"] = []
    metrics.write_text(json.dumps({"runs": [run]}) + "\n")
    _statistics(statistics, ["model"], evidence)

    with pytest.raises(ValueError, match="validation and test predictions"):
        freeze_evidence(metrics, output, statistics_path=statistics)

    other = tmp_path / "other.bin"
    other.write_bytes(b"other")
    metrics.write_text(json.dumps({"runs": [_run("model", evidence)]}) + "\n")
    _statistics(statistics, ["model"], other)
    with pytest.raises(ValueError, match="different test predictions"):
        freeze_evidence(metrics, output, statistics_path=statistics)


def test_focused_model_requires_both_pre_registered_comparisons(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    statistics = tmp_path / "statistics.json"
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    model = "vgg16_imagenet_448"
    metrics.write_text(json.dumps({"runs": [_run(model, evidence)]}) + "\n")
    _statistics(statistics, [model], evidence)

    with pytest.raises(ValueError, match="vgg16_imagenet_448_minus"):
        freeze_evidence(metrics, output, statistics_path=statistics)
