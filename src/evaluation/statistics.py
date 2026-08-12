"""Patient-level confidence intervals and paired model comparisons."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate, probability_metrics
from src.evaluation.provenance import describe_file
from src.evaluation.results_io import write_json_atomic

STATISTICS_VERSION = 1
FOCUSED_MODEL = "vgg16_imagenet_448"
FOCUSED_SEED_MODELS = (
    f"{FOCUSED_MODEL}_seed7",
    f"{FOCUSED_MODEL}_seed2026",
)
DEFAULT_COMPARISONS = (
    ("vgg16_imagenet", "vgg16_scratch"),
    ("ensemble", "vgg16_imagenet"),
    (FOCUSED_MODEL, "vgg16_imagenet"),
    (FOCUSED_MODEL, "resnet50_imagenet"),
    (FOCUSED_SEED_MODELS[0], "vgg16_imagenet_seed7"),
    (FOCUSED_SEED_MODELS[1], "vgg16_imagenet_seed2026"),
)
REQUIRED_COLUMNS = {
    "run_name",
    "split",
    "image_id",
    "patient_id",
    "birads_density",
    "lesion_type",
    "label",
    "logit",
    "probability",
    "calibrated_probability",
    "threshold",
    "fixed_specificity_target",
    "fixed_specificity_threshold",
    "seed",
    "checkpoint_sha256",
}


def read_predictions(path: Path) -> pd.DataFrame:
    """Read and validate one test-prediction file."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Prediction file {path} is missing columns: {sorted(missing)}"
        )
    if frame.empty or set(frame["split"]) != {"test"}:
        raise ValueError(
            f"Prediction file {path} must contain one non-empty test split."
        )
    if frame["patient_id"].isna().any() or frame["image_id"].isna().any():
        raise ValueError(f"Prediction file {path} has missing case identifiers.")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Prediction file {path} has duplicate image IDs.")
    constant_columns = (
        "run_name",
        "threshold",
        "fixed_specificity_target",
        "fixed_specificity_threshold",
        "seed",
        "checkpoint_sha256",
    )
    if any(frame[column].nunique() != 1 for column in constant_columns):
        raise ValueError(f"Prediction file {path} mixes runs or thresholds.")
    labels = frame["label"].to_numpy()
    probabilities = frame["probability"].to_numpy(dtype=float)
    calibrated = frame["calibrated_probability"].to_numpy(dtype=float)
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(f"Prediction file {path} has invalid labels.")
    if not np.isfinite(probabilities).all() or not np.isfinite(calibrated).all():
        raise ValueError(f"Prediction file {path} has non-finite probabilities.")
    if not np.all((probabilities >= 0) & (probabilities <= 1)) or not np.all(
        (calibrated >= 0) & (calibrated <= 1)
    ):
        raise ValueError(f"Prediction file {path} has invalid probabilities.")
    return frame.sort_values("image_id").reset_index(drop=True)


def patient_stratified_samples(
    frame: pd.DataFrame, n_resamples: int, seed: int
) -> Iterator[np.ndarray]:
    """Yield cluster-bootstrap row indices, stratified by patient outcome."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")
    patient_labels = frame.groupby("patient_id", sort=True)["label"].max().astype(int)
    strata = {
        label: patient_labels[patient_labels == label].index.to_numpy()
        for label in (0, 1)
    }
    if any(values.size == 0 for values in strata.values()):
        raise ValueError("Both patient outcome strata are required for bootstrapping.")
    rows_by_patient = {
        patient: np.flatnonzero(frame["patient_id"].to_numpy() == patient)
        for patient in patient_labels.index
    }
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        sampled_rows: list[np.ndarray] = []
        for label in (0, 1):
            patients = strata[label]
            sampled = rng.choice(patients, size=patients.size, replace=True)
            sampled_rows.extend(rows_by_patient[patient] for patient in sampled)
        yield np.concatenate(sampled_rows)


def _metric_values(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, float]:
    sample = frame.iloc[indices]
    labels = sample["label"].to_numpy(dtype=int)
    probabilities = sample["probability"].to_numpy(dtype=float)
    threshold = float(frame["threshold"].iloc[0])
    panel = evaluate(labels, probabilities, threshold=threshold)
    quality = probability_metrics(labels, probabilities)
    calibrated_quality = probability_metrics(
        labels, sample["calibrated_probability"].to_numpy(dtype=float)
    )
    fixed_panel = evaluate(
        labels,
        probabilities,
        threshold=float(frame["fixed_specificity_threshold"].iloc[0]),
    )
    fixed_threshold = float(frame["fixed_specificity_threshold"].iloc[0])
    lesion_types = sample["lesion_type"].astype(str).str.strip().str.lower().to_numpy()
    densities = sample["birads_density"].to_numpy()

    def subgroup_sensitivity(mask: np.ndarray) -> float:
        positive = mask & (labels == 1)
        if not positive.any():
            return float("nan")
        predicted = probabilities >= fixed_threshold
        return float(predicted[positive].mean())

    return {
        "auc": panel.auc,
        "accuracy": panel.accuracy,
        "sensitivity": panel.sensitivity,
        "specificity": panel.specificity,
        "ppv": panel.ppv,
        "npv": panel.npv,
        "f1": panel.f1,
        "average_precision": quality.average_precision,
        "brier_score": quality.brier_score,
        "negative_log_likelihood": quality.negative_log_likelihood,
        "calibrated_brier_score": calibrated_quality.brier_score,
        "calibrated_negative_log_likelihood": (
            calibrated_quality.negative_log_likelihood
        ),
        "sensitivity_at_fixed_specificity": fixed_panel.sensitivity,
        "specificity_at_fixed_specificity": fixed_panel.specificity,
        "calcification_sensitivity_at_fixed_specificity": subgroup_sensitivity(
            lesion_types == "calcification"
        ),
        "dense_breast_sensitivity_at_fixed_specificity": subgroup_sensitivity(
            densities == 4
        ),
    }


def _summary(estimate: float, samples: np.ndarray) -> dict[str, float | int]:
    valid = samples[np.isfinite(samples)]
    if not np.isfinite(estimate) or valid.size == 0:
        raise ValueError("A metric has no valid bootstrap samples.")
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_valid_resamples": int(valid.size),
    }


def model_intervals(
    frame: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 42
) -> dict[str, object]:
    """Calculate patient-level bootstrap intervals for one model."""
    full = np.arange(len(frame))
    estimate = _metric_values(frame, full)
    samples: dict[str, list[float]] = {name: [] for name in estimate}
    for indices in patient_stratified_samples(frame, n_resamples, seed):
        values = _metric_values(frame, indices)
        for name, value in values.items():
            samples[name].append(value)
    return {
        "n_cases": len(frame),
        "n_patients": int(frame["patient_id"].nunique()),
        "threshold": float(frame["threshold"].iloc[0]),
        "fixed_specificity_target": float(frame["fixed_specificity_target"].iloc[0]),
        "fixed_specificity_threshold": float(
            frame["fixed_specificity_threshold"].iloc[0]
        ),
        "metrics": {
            name: _summary(value, np.asarray(samples[name], dtype=float))
            for name, value in estimate.items()
        },
    }


def _aligned_frames(
    first: pd.DataFrame, second: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = first.sort_values("image_id").reset_index(drop=True)
    second = second.sort_values("image_id").reset_index(drop=True)
    for column in ("image_id", "patient_id", "label"):
        if not first[column].equals(second[column]):
            raise ValueError(f"Paired models do not match on {column}.")
    return first, second


def paired_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Compare two models on the same patient bootstrap samples."""
    first, second = _aligned_frames(first, second)
    full = np.arange(len(first))
    first_estimate = _metric_values(first, full)
    second_estimate = _metric_values(second, full)
    names = (
        "auc",
        "sensitivity",
        "specificity",
        "average_precision",
        "sensitivity_at_fixed_specificity",
        "specificity_at_fixed_specificity",
        "calcification_sensitivity_at_fixed_specificity",
        "dense_breast_sensitivity_at_fixed_specificity",
    )
    differences: dict[str, list[float]] = {name: [] for name in names}
    for indices in patient_stratified_samples(first, n_resamples, seed):
        first_values = _metric_values(first, indices)
        second_values = _metric_values(second, indices)
        for name in names:
            differences[name].append(first_values[name] - second_values[name])

    metrics: dict[str, object] = {}
    for name in names:
        values = np.asarray(differences[name], dtype=float)
        valid = values[np.isfinite(values)]
        result = _summary(first_estimate[name] - second_estimate[name], values)
        n = len(valid)
        lower_tail = (np.count_nonzero(valid <= 0.0) + 1) / (n + 1)
        upper_tail = (np.count_nonzero(valid >= 0.0) + 1) / (n + 1)
        result["p_two_sided"] = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
        metrics[name] = result
    return {"first_minus_second": metrics}


def seed_repeat_summary(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Summarise groups that contain at least three seed runs."""
    grouped: dict[str, list[tuple[str, int, float]]] = {}
    for name, frame in frames.items():
        base = re.sub(r"_seed\d+$", "", name)
        seed = int(frame["seed"].iloc[0])
        auc = _metric_values(frame, np.arange(len(frame)))["auc"]
        grouped.setdefault(base, []).append((name, seed, auc))

    summaries: dict[str, object] = {}
    for base, runs in grouped.items():
        unique_seeds = {seed for _, seed, _ in runs}
        if len(unique_seeds) < 3:
            continue
        runs.sort(key=lambda item: item[1])
        values = np.asarray([auc for _, _, auc in runs], dtype=float)
        summaries[base] = {
            "n_seeds": len(unique_seeds),
            "auc_mean": float(values.mean()),
            "auc_standard_deviation": float(values.std(ddof=1)),
            "auc_min": float(values.min()),
            "auc_max": float(values.max()),
            "runs": [
                {"model": name, "seed": seed, "auc": auc} for name, seed, auc in runs
            ],
        }
    return summaries


def _model_names(metrics_path: Path) -> list[str]:
    try:
        data = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read metrics file {metrics_path}: {exc}") from exc
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError("Metrics must contain a non-empty runs list.")
    return [str(run["model"]) for run in runs]


def generate_statistics(
    metrics_path: Path,
    predictions_dir: Path,
    output_path: Path,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    comparisons: Iterable[tuple[str, str]] = DEFAULT_COMPARISONS,
) -> dict[str, Any]:
    """Build the statistics file from saved test predictions."""
    names = _model_names(metrics_path)
    frames = {}
    prediction_files = {}
    root = Path(__file__).resolve().parents[2]
    for name in names:
        path = predictions_dir / f"{name}.test.csv"
        frame = read_predictions(path)
        if str(frame["run_name"].iloc[0]) != name:
            raise ValueError(f"Prediction file for {name} records another run name.")
        frames[name] = frame
        prediction_files[name] = describe_file(path, root)
    models = {
        name: model_intervals(frame, n_resamples=n_resamples, seed=seed)
        for name, frame in frames.items()
    }
    paired = {}
    for first, second in comparisons:
        if first not in frames or second not in frames:
            continue
        paired[f"{first}_minus_{second}"] = paired_comparison(
            frames[first], frames[second], n_resamples=n_resamples, seed=seed
        )
    result: dict[str, Any] = {
        "version": STATISTICS_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": {
            "unit": "patient",
            "stratified_by": "patient_any_malignant",
            "n_resamples": n_resamples,
            "confidence_level": 0.95,
            "interval": "percentile",
            "paired_resampling": "same_patient_samples",
            "p_value": "two_sided_bootstrap_with_plus_one_correction",
            "multiplicity_adjustment": "none",
            "calibration_uncertainty": "conditional_on_saved_temperature",
            "seed": seed,
        },
        "models": models,
        "prediction_files": prediction_files,
        "paired_comparisons": paired,
        "seed_repeats": seed_repeat_summary(frames),
    }
    write_json_atomic(output_path, result)
    return result


@click.command()
@click.option(
    "--metrics",
    "metrics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/metrics.json"),
    show_default=True,
)
@click.option(
    "--predictions-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results/predictions"),
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("results/statistics.json"),
    show_default=True,
)
@click.option(
    "--n-resamples", type=click.IntRange(min=1), default=2000, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
def cli(
    metrics_path: Path,
    predictions_dir: Path,
    output_path: Path,
    n_resamples: int,
    seed: int,
) -> None:
    try:
        result = generate_statistics(
            metrics_path,
            predictions_dir,
            output_path,
            n_resamples=n_resamples,
            seed=seed,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote intervals for {len(result['models'])} models to {output_path}")


if __name__ == "__main__":
    cli()
