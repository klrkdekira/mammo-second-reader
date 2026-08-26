import json
from pathlib import Path

import pytest

from src.evaluation.lineage import sha256_file
from src.evaluation.verify_bundle import verify_bundle


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config = _write(tmp_path / "configs/model.toml", 'run_name = "model"\n')
    checkpoint = _write(tmp_path / "models/model.pt", "checkpoint\n")
    threshold = _write(tmp_path / "models/model.threshold.json", "{}\n")
    source = _write(tmp_path / "src/model.py", "MODEL = 1\n")
    train = _write(tmp_path / "manifests/train.csv", "patient_id,y_true\ntrain,0\n")
    validation_manifest = _write(
        tmp_path / "manifests/val.csv", "patient_id,y_true\nval-a,0\nval-b,1\n"
    )
    test_manifest = _write(
        tmp_path / "manifests/test.csv",
        "patient_id,y_true\ntest-a,0\ntest-b,1\ntest-c,0\n",
    )
    validation_prediction = _write(
        tmp_path / "results/predictions/model.validation.csv",
        "patient_id,y_true,probability\nval-a,0,0.1\nval-b,1,0.9\n",
    )
    test_prediction = _write(
        tmp_path / "results/predictions/model.test.csv",
        "patient_id,y_true,probability\ntest-a,0,0.1\ntest-b,1,0.9\ntest-c,0,0.2\n",
    )
    figure = _write(tmp_path / "results/figures/one.png", "figure\n")

    lineage = {
        "git": {"snapshot": "a" * 64},
        "config": _descriptor(config, tmp_path),
        "checkpoints": [_descriptor(checkpoint, tmp_path)],
        "manifests": [
            _descriptor(train, tmp_path),
            _descriptor(validation_manifest, tmp_path),
            _descriptor(test_manifest, tmp_path),
        ],
        "threshold_sidecar": _descriptor(threshold, tmp_path),
        "prediction_files": [
            _descriptor(validation_prediction, tmp_path),
            _descriptor(test_prediction, tmp_path),
        ],
        "code": {
            "preprocessing_files": [_descriptor(source, tmp_path)],
            "evaluation_files": [_descriptor(source, tmp_path)],
        },
    }
    metrics_path = tmp_path / "results/metrics.json"
    metrics_path.write_text(
        json.dumps({"runs": [{"model": "model", "lineage": lineage}]})
    )

    statistics_path = tmp_path / "results/statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "models": {"model": {"n_cases": 3}},
                "prediction_files": {"model": _descriptor(test_prediction, tmp_path)},
            }
        )
    )
    freeze_path = tmp_path / "results/evidence-freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "n_runs": 1,
                "models": ["model"],
                "source_snapshot": "a" * 64,
                "metrics": _descriptor(metrics_path, tmp_path),
                "statistics": _descriptor(statistics_path, tmp_path),
                "figures": [_descriptor(figure, tmp_path)],
            }
        )
    )
    return metrics_path, statistics_path, freeze_path, test_prediction


def test_verify_bundle_checks_hashes_coverage_and_rows(tmp_path):
    metrics, statistics, frozen, _ = _bundle(tmp_path)

    summary = verify_bundle(
        metrics,
        statistics,
        frozen,
        root=tmp_path,
        expected_runs=None,
        require_corrected_manifests=False,
    )

    assert summary["runs"] == 1
    assert summary["figures"] == 1
    assert summary["validation_rows"] == 2
    assert summary["test_rows"] == 3


def test_verify_bundle_rejects_a_changed_prediction(tmp_path):
    metrics, statistics, frozen, test_prediction = _bundle(tmp_path)
    test_prediction.write_text(test_prediction.read_text() + "test-d,1,0.8\n")

    with pytest.raises(ValueError, match="hash does not match"):
        verify_bundle(
            metrics,
            statistics,
            frozen,
            root=tmp_path,
            expected_runs=None,
            require_corrected_manifests=False,
        )
