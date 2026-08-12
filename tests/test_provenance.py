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
    config.write_text("seed = 42\n")
    checkpoint.write_bytes(b"weights")
    split.write_text("image_id,label\na,0\n")
    threshold.write_text('{"youden_j": 0.5}\n')

    result = build_run_provenance(
        config_path=config,
        checkpoint_paths=[checkpoint],
        manifest_paths=[split],
        threshold_path=threshold,
        extra={"seed": 42},
    )

    assert result["config"]["sha256"] == sha256_file(config)
    assert result["checkpoints"][0]["sha256"] == sha256_file(checkpoint)
    assert result["manifests"][0]["sha256"] == sha256_file(split)
    assert result["experiment"]["seed"] == 42
    assert result["code"]["preprocessing_fingerprint"]


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
        "provenance": {
            "git": {"commit": commit, "dirty_evidence_files": dirty},
            "config": descriptor,
            "checkpoints": [descriptor],
            "manifests": [descriptor],
            "threshold_sidecar": descriptor,
            "code": {
                "preprocessing_fingerprint": preprocessing,
                "evaluation_fingerprint": evaluation,
                "preprocessing_files": [descriptor],
                "evaluation_files": [descriptor],
            },
        },
    }


def test_freeze_evidence_requires_homogeneous_clean_provenance(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    metrics.write_text(
        json.dumps({"runs": [_run("a", evidence), _run("b", evidence)]}) + "\n"
    )

    result = freeze_evidence(metrics, output)

    assert result["n_runs"] == 2
    assert result["git_commit"] == "abc"
    assert json.loads(output.read_text())["models"] == ["a", "b"]


def test_freeze_evidence_rejects_legacy_or_dirty_runs(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "freeze.json"
    metrics.write_text(json.dumps({"runs": [{"model": "legacy"}]}) + "\n")
    with pytest.raises(ValueError, match="missing provenance"):
        freeze_evidence(metrics, output)

    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"frozen")
    metrics.write_text(
        json.dumps({"runs": [_run("dirty", evidence, dirty=True)]}) + "\n"
    )
    with pytest.raises(ValueError, match="dirty evidence"):
        freeze_evidence(metrics, output)
