"""Check the results and save a freeze manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from src.evaluation.lineage import sha256_file
from src.evaluation.results_io import write_json_atomic

FREEZE_VERSION = 3
MIN_LINEAGE_VERSION = 4
REQUIRED_BOOTSTRAP_RESAMPLES = 2000
REQUIRED_AUDIT_FIELDS = {
    "calibration",
    "decision_curve",
    "density_strata",
    "fixed_specificity",
    "lesion_strata",
    "precision_recall",
    "probability_metrics",
}
REQUIRED_PAIRED_COMPARISONS = {
    "vgg16_imagenet_minus_vgg16_scratch",
    "ensemble_minus_vgg16_imagenet",
}
TRANSFER_SEED_COMPARISONS = {
    "vgg16_imagenet_seed7": "vgg16_imagenet_seed7_minus_vgg16_scratch_seed7",
    "vgg16_imagenet_seed2026": ("vgg16_imagenet_seed2026_minus_vgg16_scratch_seed2026"),
}
FOCUSED_MODEL = "vgg16_imagenet_448"
FOCUSED_COMPARISONS = {
    f"{FOCUSED_MODEL}_minus_vgg16_imagenet",
    f"{FOCUSED_MODEL}_minus_resnet50_imagenet",
}
FOCUSED_SEED_COMPARISONS = {
    "vgg16_imagenet_448_seed7": "vgg16_imagenet_448_seed7_minus_vgg16_imagenet_seed7",
    "vgg16_imagenet_448_seed2026": (
        "vgg16_imagenet_448_seed2026_minus_vgg16_imagenet_seed2026"
    ),
}


def _resolve_recorded_path(recorded: str, root: Path) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else root / path


def _verify_descriptor(descriptor: dict[str, object], root: Path) -> None:
    if not descriptor.get("exists", False):
        raise ValueError(
            f"Recorded evidence input is missing: {descriptor.get('path')}"
        )
    path = _resolve_recorded_path(str(descriptor["path"]), root)
    if not path.is_file():
        raise ValueError(f"Evidence input no longer exists: {path}")
    actual = sha256_file(path)
    if actual != descriptor.get("sha256"):
        raise ValueError(f"Evidence input changed after evaluation: {path}")


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot freeze invalid metrics file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise TypeError("Metrics must contain a top-level runs list.")
    if not data["runs"]:
        raise ValueError("Metrics contains no run records.")
    return data


def _load_statistics(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot freeze invalid statistics file {path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise TypeError("Statistics must contain a top-level models object.")
    method = data.get("method")
    if not isinstance(method, dict):
        raise TypeError("Statistics must describe its bootstrap method.")
    if method.get("unit") != "patient":
        raise ValueError("Confidence intervals must use patient-level resampling.")
    if method.get("n_resamples") != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"Evidence freezing requires {REQUIRED_BOOTSTRAP_RESAMPLES} resamples."
        )
    paired = data.get("paired_comparisons")
    if not isinstance(paired, dict):
        raise TypeError("Statistics must contain paired model comparisons.")
    if not isinstance(data.get("prediction_files"), dict):
        raise TypeError("Statistics must record its prediction files.")
    return data


def freeze_evidence(
    metrics_path: Path,
    output_path: Path,
    *,
    statistics_path: Path = Path("results/statistics.json"),
    figures_dir: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Check that all runs match, then freeze the results."""
    metrics_path = Path(metrics_path)
    data = _load_metrics(metrics_path)
    statistics_path = Path(statistics_path)
    statistics = _load_statistics(statistics_path)
    runs = data["runs"]
    missing = [run.get("model", "<unnamed>") for run in runs if "lineage" not in run]
    if missing:
        raise ValueError(f"Run records are missing lineage: {missing}")
    old_lineage = [
        run.get("model", "<unnamed>")
        for run in runs
        if run["lineage"].get("version", 0) < MIN_LINEAGE_VERSION
    ]
    if old_lineage:
        raise ValueError(
            f"Run records do not include case-level prediction lineage: {old_lineage}"
        )
    model_names = [str(run["model"]) for run in runs]
    if set(statistics["models"]) != set(model_names):
        raise ValueError("Statistics and metrics contain different model sets.")
    if set(statistics["prediction_files"]) != set(model_names):
        raise ValueError("Statistics does not cover every model prediction file.")
    required_comparisons = set(REQUIRED_PAIRED_COMPARISONS)
    for model, comparison in TRANSFER_SEED_COMPARISONS.items():
        if model in model_names:
            required_comparisons.add(comparison)
    if FOCUSED_MODEL in model_names:
        required_comparisons.update(FOCUSED_COMPARISONS)
    for model, comparison in FOCUSED_SEED_COMPARISONS.items():
        if model in model_names:
            required_comparisons.add(comparison)
    missing_comparisons = required_comparisons - set(statistics["paired_comparisons"])
    if missing_comparisons:
        raise ValueError(
            f"Statistics is missing paired comparisons: {sorted(missing_comparisons)}"
        )
    incomplete_audits = {
        run["model"]: sorted(REQUIRED_AUDIT_FIELDS - set(run))
        for run in runs
        if REQUIRED_AUDIT_FIELDS - set(run)
    }
    if incomplete_audits:
        raise ValueError(
            f"Run records have incomplete statistical audits: {incomplete_audits}"
        )
    ensemble = next((run for run in runs if run["model"] == "ensemble"), None)
    if ensemble is not None and "gradcam_policy" not in ensemble:
        raise ValueError("The ensemble must record its Grad-CAM audit policy.")
    incomplete_fixed_strata = [
        run["model"]
        for run in runs
        if not {"density_strata", "lesion_strata"}.issubset(run["fixed_specificity"])
    ]
    if incomplete_fixed_strata:
        raise ValueError(
            "Run records are missing fixed-specificity subgroup audits: "
            f"{incomplete_fixed_strata}"
        )

    commits = {run["lineage"]["git"]["commit"] for run in runs}
    sources = {run["lineage"]["git"].get("source", "git") for run in runs}
    snapshots = {run["lineage"]["git"].get("snapshot") for run in runs}
    preprocessing = {
        run["lineage"]["code"]["preprocessing_fingerprint"] for run in runs
    }
    evaluation = {run["lineage"]["code"]["evaluation_fingerprint"] for run in runs}
    dirty = [
        run["model"]
        for run in runs
        if run["lineage"]["git"].get(
            "dirty_evidence_files",
            run["lineage"]["git"].get("dirty_tracked_files", False),
        )
    ]
    if None in commits or len(commits) != 1:
        raise ValueError(
            f"Runs do not share one known Git commit: {sorted(map(str, commits))}"
        )
    if len(sources) != 1:
        raise ValueError("Runs use different source deployment methods.")
    if len(snapshots) != 1:
        raise ValueError("Runs use different source worktree snapshots.")
    if sources == {"cuda_sync_snapshot"} and snapshots == {None}:
        raise ValueError("CUDA-synced runs are missing a source worktree snapshot.")
    if len(preprocessing) != 1 or len(evaluation) != 1:
        raise ValueError("Runs use different preprocessing/evaluation code.")
    if dirty and not allow_dirty:
        raise ValueError(
            f"Runs use dirty evidence worktrees: {dirty}. "
            "Commit the implementation and rerun evaluation before freezing."
        )

    root = Path(__file__).resolve().parents[2]
    for run in runs:
        lineage = run["lineage"]
        _verify_descriptor(lineage["config"], root)
        for descriptor in lineage["checkpoints"]:
            _verify_descriptor(descriptor, root)
        for descriptor in lineage["manifests"]:
            _verify_descriptor(descriptor, root)
        threshold = lineage.get("threshold_sidecar")
        if threshold is not None:
            _verify_descriptor(threshold, root)
        for descriptor in lineage["code"]["preprocessing_files"]:
            _verify_descriptor(descriptor, root)
        for descriptor in lineage["code"]["evaluation_files"]:
            _verify_descriptor(descriptor, root)
        predictions = lineage.get("prediction_files", [])
        if len(predictions) != 2:
            raise ValueError(
                f"Run {run['model']} must record validation and test predictions."
            )
        for descriptor in predictions:
            _verify_descriptor(descriptor, root)
        statistics_prediction = statistics["prediction_files"][run["model"]]
        _verify_descriptor(statistics_prediction, root)
        if statistics_prediction.get("path") != predictions[1].get(
            "path"
        ) or statistics_prediction.get("sha256") != predictions[1].get("sha256"):
            raise ValueError(
                f"Statistics for {run['model']} used different test predictions."
            )

    figures: list[dict[str, object]] = []
    if figures_dir and Path(figures_dir).is_dir():
        for path in sorted(Path(figures_dir).glob("*.png")):
            figures.append(
                {
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frozen = {
        "version": FREEZE_VERSION,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "metrics": {
            "path": metrics_path.as_posix(),
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        },
        "statistics": {
            "path": statistics_path.as_posix(),
            "size_bytes": statistics_path.stat().st_size,
            "sha256": sha256_file(statistics_path),
            "method": statistics["method"],
        },
        "git_commit": next(iter(commits)),
        "source_snapshot": next(iter(snapshots)),
        "source": next(iter(sources)),
        "preprocessing_fingerprint": next(iter(preprocessing)),
        "evaluation_fingerprint": next(iter(evaluation)),
        "models": [run["model"] for run in runs],
        "n_runs": len(runs),
        "dirty_override": bool(dirty),
        "figures": figures,
    }
    write_json_atomic(Path(output_path), frozen)
    return frozen


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
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("results/evidence-freeze.json"),
    show_default=True,
)
@click.option(
    "--figures-dir",
    type=click.Path(path_type=Path),
    default=Path("results/figures"),
    show_default=True,
)
@click.option("--allow-dirty", is_flag=True, default=False)
def cli(
    metrics_path: Path,
    statistics_path: Path,
    output_path: Path,
    figures_dir: Path,
    allow_dirty: bool,
) -> None:
    try:
        frozen = freeze_evidence(
            metrics_path,
            output_path,
            statistics_path=statistics_path,
            figures_dir=figures_dir,
            allow_dirty=allow_dirty,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Frozen {frozen['n_runs']} runs to {output_path}")


if __name__ == "__main__":
    cli()
