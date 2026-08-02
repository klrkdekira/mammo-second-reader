"""Unit tests for the metrics.json upsert behaviour in evaluate.py and ensemble.py."""

import json

from src.evaluation.evaluate import _append_record as _append_evaluate_record
from src.training.ensemble import _append_record as _append_ensemble_record


def test_evaluate_append_record_upserts_by_model(tmp_path):
    path = tmp_path / "metrics.json"

    _append_evaluate_record({"model": "baseline", "test": {"auc": 0.6}}, path=path)
    _append_evaluate_record({"model": "baseline", "test": {"auc": 0.7}}, path=path)
    _append_evaluate_record({"model": "resnet50_imagenet", "test": {"auc": 0.68}}, path=path)

    runs = json.loads(path.read_text())["runs"]
    assert len(runs) == 2
    by_model = {r["model"]: r for r in runs}
    assert by_model["baseline"]["test"]["auc"] == 0.7
    assert by_model["resnet50_imagenet"]["test"]["auc"] == 0.68


def test_ensemble_append_record_upserts_by_model(tmp_path):
    path = tmp_path / "metrics.json"

    _append_ensemble_record({"model": "ensemble", "test": {"auc": 0.6}}, path=path)
    _append_ensemble_record({"model": "ensemble", "test": {"auc": 0.65}}, path=path)

    runs = json.loads(path.read_text())["runs"]
    assert len(runs) == 1
    assert runs[0]["test"]["auc"] == 0.65
