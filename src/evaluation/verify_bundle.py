"""Verify a synced evidence bundle without rewriting it."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click

from src.evaluation.lineage import sha256_file

CORRECTED_MANIFEST_HASHES = {
    "manifests/cbis-ddsm/train.csv": (
        "add9c8b8bc95fe86e21673d7438f8179c2cde344ff5b0cf439690145cfdb8d18"
    ),
    "manifests/cbis-ddsm/val.csv": (
        "9c8dbd5b413d2c2d74cdec93c8ba0bdf6f3a9ae81c952f4705928b87f7b8ea5f"
    ),
    "manifests/cbis-ddsm/test.csv": (
        "225241c53968e10c18e4040c67edef304009e4bd4a3c5f73e67f7958a8e85634"
    ),
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _resolve(recorded: object, root: Path) -> Path:
    path = Path(str(recorded))
    return path if path.is_absolute() else root / path


def _verify_descriptor(
    descriptor: object,
    root: Path,
    *,
    verified: set[tuple[str, str]],
) -> Path:
    if not isinstance(descriptor, dict):
        raise TypeError("Evidence file descriptors must be JSON objects.")
    recorded = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not recorded or not expected:
        raise ValueError(f"Incomplete evidence file descriptor: {descriptor}")
    if descriptor.get("exists") is False:
        raise ValueError(f"Recorded evidence file was missing: {recorded}")

    path = _resolve(recorded, root)
    key = (str(path), str(expected))
    if key in verified:
        return path
    if not path.is_file():
        raise ValueError(f"Evidence file does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Evidence file hash does not match: {path}")
    verified.add(key)
    return path


def _lineage_descriptors(lineage: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield lineage["config"]
    yield from lineage["checkpoints"]
    yield from lineage["manifests"]
    threshold = lineage.get("threshold_sidecar")
    if threshold is not None:
        yield threshold
    code = lineage["code"]
    yield from code["preprocessing_files"]
    yield from code["evaluation_files"]
    yield from lineage["prediction_files"]


def _csv_data_rows(path: Path, cache: dict[Path, int]) -> int:
    if path not in cache:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV has no header: {path}") from exc
            cache[path] = sum(1 for _ in reader)
    return cache[path]


def _prediction_descriptor(lineage: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        descriptor
        for descriptor in lineage["prediction_files"]
        if str(descriptor.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {suffix} prediction file, found {len(matches)}."
        )
    return matches[0]


def _manifest_descriptor(lineage: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        descriptor
        for descriptor in lineage["manifests"]
        if str(descriptor.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} manifest, found {len(matches)}.")
    return matches[0]


def verify_bundle(
    metrics_path: Path = Path("results/metrics.json"),
    statistics_path: Path = Path("results/statistics.json"),
    freeze_path: Path = Path("results/evidence-freeze.json"),
    *,
    root: Path | None = None,
    expected_runs: int | None = 22,
    require_corrected_manifests: bool = True,
) -> dict[str, object]:
    """Verify file hashes, run coverage and prediction row counts."""
    root = root or Path(__file__).resolve().parents[2]
    metrics = _load_object(metrics_path, "metrics")
    statistics = _load_object(statistics_path, "statistics")
    frozen = _load_object(freeze_path, "evidence freeze")

    runs = metrics.get("runs")
    models = statistics.get("models")
    prediction_files = statistics.get("prediction_files")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Metrics must contain at least one run.")
    if not isinstance(models, dict) or not isinstance(prediction_files, dict):
        raise TypeError("Statistics must contain model and prediction-file objects.")

    names = [str(run.get("model", "")) for run in runs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Metric run names must be present and unique.")
    if expected_runs is not None and len(names) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, found {len(names)}.")
    if set(models) != set(names) or set(prediction_files) != set(names):
        raise ValueError("Metrics and statistics do not cover the same run set.")
    if frozen.get("n_runs") != len(names) or set(frozen.get("models", [])) != set(
        names
    ):
        raise ValueError("The evidence freeze does not cover the metric run set.")

    verified: set[tuple[str, str]] = set()
    _verify_descriptor(frozen.get("metrics"), root, verified=verified)
    _verify_descriptor(frozen.get("statistics"), root, verified=verified)
    figures = frozen.get("figures", [])
    if not isinstance(figures, list):
        raise TypeError("The evidence freeze figures field must be a list.")
    for descriptor in figures:
        _verify_descriptor(descriptor, root, verified=verified)

    csv_rows: dict[Path, int] = {}
    snapshots: set[object] = set()
    manifest_hashes: dict[str, set[str]] = {}
    for run in runs:
        lineage = run.get("lineage")
        if not isinstance(lineage, dict):
            raise TypeError(f"Run {run['model']} has no lineage record.")
        snapshots.add(lineage.get("git", {}).get("snapshot"))
        for descriptor in _lineage_descriptors(lineage):
            _verify_descriptor(descriptor, root, verified=verified)

        for suffix in ("train.csv", "val.csv", "test.csv"):
            descriptor = _manifest_descriptor(lineage, suffix)
            recorded = str(descriptor["path"])
            manifest_hashes.setdefault(recorded, set()).add(str(descriptor["sha256"]))

        validation = _prediction_descriptor(lineage, ".validation.csv")
        test = _prediction_descriptor(lineage, ".test.csv")
        validation_path = _resolve(validation["path"], root)
        test_path = _resolve(test["path"], root)
        val_manifest = _resolve(_manifest_descriptor(lineage, "val.csv")["path"], root)
        test_manifest = _resolve(
            _manifest_descriptor(lineage, "test.csv")["path"], root
        )
        if _csv_data_rows(validation_path, csv_rows) != _csv_data_rows(
            val_manifest, csv_rows
        ):
            raise ValueError(
                f"Validation prediction row count differs for {run['model']}."
            )
        test_rows = _csv_data_rows(test_path, csv_rows)
        if test_rows != _csv_data_rows(test_manifest, csv_rows):
            raise ValueError(f"Test prediction row count differs for {run['model']}.")
        model_stats = models[run["model"]]
        if not isinstance(model_stats, dict) or model_stats.get("n_cases") != test_rows:
            raise ValueError(f"Statistics case count differs for {run['model']}.")

        statistics_prediction = prediction_files[run["model"]]
        _verify_descriptor(statistics_prediction, root, verified=verified)
        if statistics_prediction.get("path") != test.get(
            "path"
        ) or statistics_prediction.get("sha256") != test.get("sha256"):
            raise ValueError(
                f"Statistics used a different test prediction file for {run['model']}."
            )

    if None in snapshots or len(snapshots) != 1:
        raise ValueError("Runs do not share one known source snapshot.")
    if frozen.get("source_snapshot") != next(iter(snapshots)):
        raise ValueError("The evidence freeze records a different source snapshot.")
    if any(len(hashes) != 1 for hashes in manifest_hashes.values()):
        raise ValueError("Runs do not share one set of manifest hashes.")
    if require_corrected_manifests:
        observed = {
            path: next(iter(manifest_hashes.get(path, set())), None)
            for path in CORRECTED_MANIFEST_HASHES
        }
        if observed != CORRECTED_MANIFEST_HASHES:
            raise ValueError("The registered corrected manifest hashes do not match.")

    validation_rows = _csv_data_rows(
        _resolve(_manifest_descriptor(runs[0]["lineage"], "val.csv")["path"], root),
        csv_rows,
    )
    test_rows = _csv_data_rows(
        _resolve(_manifest_descriptor(runs[0]["lineage"], "test.csv")["path"], root),
        csv_rows,
    )
    return {
        "runs": len(names),
        "figures": len(figures),
        "verified_files": len(verified),
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "source_snapshot": next(iter(snapshots)),
    }


@click.command()
@click.option(
    "--metrics",
    "metrics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/metrics.json"),
    show_default=True,
)
@click.option(
    "--statistics",
    "statistics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/statistics.json"),
    show_default=True,
)
@click.option(
    "--freeze",
    "freeze_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/evidence-freeze.json"),
    show_default=True,
)
def cli(metrics_path: Path, statistics_path: Path, freeze_path: Path) -> None:
    """Verify the complete local evidence bundle without changing it."""
    try:
        summary = verify_bundle(metrics_path, statistics_path, freeze_path)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        "Verified "
        f"{summary['runs']} runs, {summary['verified_files']} files and "
        f"{summary['figures']} figures; prediction rows are "
        f"{summary['validation_rows']} validation and {summary['test_rows']} test."
    )


if __name__ == "__main__":
    cli()
