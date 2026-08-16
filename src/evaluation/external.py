"""Run the locked external evaluation on INbreast.

The checkpoint, thresholds, and calibration temperature are loaded from the
frozen CBIS-DDSM run. Results are written separately from the internal evidence
file and include patient-level bootstrap intervals.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.config import get_device, load_config, setup_logging
from src.data import manifest as _manifest
from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.audit import logits_to_probability
from src.evaluation.calibration import (
    CALIBRATION_VERSION,
    expected_calibration_error,
    reliability_bins,
)
from src.evaluation.decision_curve import decision_curve
from src.evaluation.density_strata import metrics_by_density
from src.evaluation.lesion_strata import metrics_by_lesion_type
from src.evaluation.metrics import (
    evaluate,
    precision_recall_points,
    probability_metrics,
)
from src.evaluation.predictions import (
    build_prediction_frame,
    write_predictions_atomic,
)
from src.evaluation.provenance import build_run_provenance, sha256_file
from src.evaluation.results_io import write_json_atomic
from src.evaluation.statistics import model_intervals, read_predictions
from src.models import build_model

LOGGER = logging.getLogger(__name__)

EXTERNAL_VERSION = 1
DEFAULT_OUTPUT = Path("results/external/metrics-inbreast.json")
DEFAULT_PREDICTIONS_DIR = Path("results/external/predictions")

_TOP_LEVEL_KEYS = {"run_name", "internal_config", "internal_run", "dataset", "data"}
_DATA_KEYS = {
    "manifest",
    "lesion_present_manifest",
    "manifest_lock",
    "image_root",
    "image_size",
    "batch_size",
    "num_workers",
}


@dataclass(frozen=True)
class ExternalConfig:
    """Declares one cold external evaluation."""

    run_name: str
    internal_config: Path
    internal_run: str
    manifest: Path
    image_root: Path
    dataset: str = "inbreast"
    lesion_present_manifest: Path | None = None
    manifest_lock: Path | None = None
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 2


@dataclass(frozen=True)
class LockedOperatingPoint:
    """Parameters transferred from the internal run."""

    source_run: str
    youden_threshold: float
    fixed_specificity_target: float
    fixed_specificity_threshold: float
    temperature: float
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "fitted_on": "cbis_ddsm_validation_fold",
            "refitted_on_external": False,
        }


def load_external_config(path: Path) -> ExternalConfig:
    """Load and validate an external-evaluation TOML."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown key(s) in {path}: {sorted(unknown)}")
    data = raw.get("data", {})
    unknown_data = set(data) - _DATA_KEYS
    if unknown_data:
        raise ValueError(f"Unknown key(s) in [data] of {path}: {sorted(unknown_data)}")
    for required in ("run_name", "internal_config", "internal_run"):
        if required not in raw:
            raise ValueError(f"{path} is missing required key {required!r}")
    for required in ("manifest", "image_root"):
        if required not in data:
            raise ValueError(f"{path} is missing required [data] key {required!r}")
    optional = {
        name: Path(data[name])
        for name in ("lesion_present_manifest", "manifest_lock")
        if data.get(name)
    }
    return ExternalConfig(
        run_name=str(raw["run_name"]),
        internal_config=Path(raw["internal_config"]),
        internal_run=str(raw["internal_run"]),
        dataset=str(raw.get("dataset", "inbreast")),
        manifest=Path(data["manifest"]),
        image_root=Path(data["image_root"]),
        image_size=int(data.get("image_size", 448)),
        batch_size=int(data.get("batch_size", 16)),
        num_workers=int(data.get("num_workers", 2)),
        **optional,
    )


def _run_record(metrics_path: Path, run_name: str) -> dict:
    payload = json.loads(Path(metrics_path).read_text())
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{metrics_path} does not contain a non-empty runs list.")
    for run in runs:
        if str(run.get("model")) == run_name:
            return run
    raise ValueError(f"{metrics_path} has no frozen record for run {run_name!r}.")


def load_locked_operating_point(
    *,
    metrics_path: Path,
    run_name: str,
    checkpoint_path: Path,
    threshold_sidecar: Path,
) -> LockedOperatingPoint:
    """Read the frozen internal operating point and verify it is self-consistent.

    The threshold appears in two independent places - the metrics record and the
    checkpoint's threshold sidecar - and a mismatch means the checkpoint and the
    frozen results have drifted apart. That has to stop a cold run, because the
    whole point is that the transferred threshold is the one that was reported.
    """
    record = _run_record(metrics_path, run_name)
    sidecar = json.loads(Path(threshold_sidecar).read_text())
    youden = float(record["val_threshold"])
    sidecar_youden = float(sidecar["youden_j"])
    if not np.isclose(youden, sidecar_youden, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Threshold drift for {run_name}: metrics.json records {youden} but "
            f"{threshold_sidecar} records {sidecar_youden}."
        )
    fixed = record.get("fixed_specificity")
    if not isinstance(fixed, dict) or not {"target", "threshold"} <= set(fixed):
        raise ValueError(
            f"Run {run_name!r} has no usable fixed_specificity record; the cold "
            "run needs both its target and its validation-derived threshold."
        )
    calibration = record.get("calibration")
    if not isinstance(calibration, dict) or "temperature" not in calibration:
        raise ValueError(f"Run {run_name!r} has no stored calibration temperature.")
    temperature = float(calibration["temperature"])
    if not temperature > 0.0:
        raise ValueError(f"Run {run_name!r} has a non-positive temperature.")
    return LockedOperatingPoint(
        source_run=run_name,
        youden_threshold=youden,
        fixed_specificity_target=float(fixed["target"]),
        fixed_specificity_threshold=float(fixed["threshold"]),
        temperature=temperature,
        checkpoint_sha256=sha256_file(Path(checkpoint_path)),
    )


def _panel_dict(panel) -> dict[str, object]:
    return {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()}


@dataclass(frozen=True)
class ExternalAudit:
    """One scored external subset, plus the probabilities it was scored from."""

    record: dict[str, object]
    probability: np.ndarray
    calibrated_probability: np.ndarray


def external_audit(
    frame: pd.DataFrame,
    logits: np.ndarray,
    locked: LockedOperatingPoint,
) -> ExternalAudit:
    """Score one external subset at the locked operating point.

    Mirrors `build_audit` but fits nothing: the temperature and both thresholds
    arrive already fixed, and no `threshold_at_specificity` call is made on
    external labels.
    """
    labels = np.asarray(frame["label"].to_numpy(), dtype=int)
    logits = np.asarray(logits, dtype=float).ravel()
    if labels.size != logits.size:
        raise ValueError("External predictions do not match the manifest length.")
    probability = logits_to_probability(logits)
    calibrated = logits_to_probability(logits, locked.temperature)
    centres, pred_mean, obs_mean = reliability_bins(calibrated, labels)

    youden_panel = evaluate(labels, probability, threshold=locked.youden_threshold)
    fixed_panel = evaluate(
        labels, probability, threshold=locked.fixed_specificity_threshold
    )
    fpr, tpr, _ = roc_curve(labels, probability)
    record: dict[str, object] = {
        "n_cases": len(frame),
        "n_patients": int(frame["patient_id"].nunique()),
        "n_malignant": int((labels == 1).sum()),
        "prevalence": float(labels.mean()),
        "test": _panel_dict(youden_panel),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "density_strata": metrics_by_density(
            frame, probability, locked.youden_threshold
        ).to_dict(orient="records"),
        "calibration": {
            "version": CALIBRATION_VERSION,
            "temperature": locked.temperature,
            "temperature_source": "internal_validation_fold_no_external_refit",
            "ece_before": expected_calibration_error(probability, labels),
            "ece_after": expected_calibration_error(calibrated, labels),
            "reliability": {
                "bin_centre": centres.tolist(),
                "pred_mean": pred_mean.tolist(),
                "obs_mean": obs_mean.tolist(),
            },
        },
        "probability_metrics": {
            "raw": dataclasses.asdict(probability_metrics(labels, probability)),
            "calibrated": dataclasses.asdict(probability_metrics(labels, calibrated)),
        },
        "precision_recall": precision_recall_points(labels, probability),
        "decision_curve": {
            key: np.asarray(value).tolist()
            for key, value in decision_curve(labels, calibrated).items()
        },
        "fixed_specificity": {
            "target": locked.fixed_specificity_target,
            "threshold_source": "internal_validation_fold_transferred_unchanged",
            "threshold": locked.fixed_specificity_threshold,
            # The target applies to the internal validation fold. Report the
            # specificity achieved here without retuning.
            "achieved_specificity": fixed_panel.specificity,
            "test": _panel_dict(fixed_panel),
            "density_strata": metrics_by_density(
                frame, probability, locked.fixed_specificity_threshold
            ).to_dict(orient="records"),
        },
    }
    if "lesion_type" in frame.columns:
        record["lesion_strata"] = metrics_by_lesion_type(
            frame, probability, locked.youden_threshold
        ).to_dict(orient="records")
    return ExternalAudit(
        record=record, probability=probability, calibrated_probability=calibrated
    )


def _subset_name(run_name: str, subset: str) -> str:
    return run_name if subset == "full" else f"{run_name}_{subset}"


def subset_names(subsets: Sequence[tuple[object, ...]]) -> list[str]:
    """Return the pre-registered subset names in evaluation order.

    Kept separate from the payload literal so that widening the subset tuple
    cannot silently break the provenance record again. An earlier revision
    added the frame and logits to each tuple and left this unpacking at two
    elements, which raised `ValueError` after inference but before the metrics
    were written.
    """
    return [str(entry[0]) for entry in subsets]


def _predict_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits.append(model(images).cpu().numpy().ravel())
            labels.append(targets.cpu().numpy().ravel())
    return np.concatenate(labels), np.concatenate(logits)


def _align_subset_logits(
    full_frame: pd.DataFrame,
    full_logits: np.ndarray,
    subset_frame: pd.DataFrame,
) -> np.ndarray:
    """Select full-set logits in the order of a registered subset manifest."""
    full_ids = full_frame["image_id"].astype(str)
    subset_ids = subset_frame["image_id"].astype(str)
    if full_ids.duplicated().any():
        raise ValueError("The full external manifest contains duplicate image IDs.")
    if subset_ids.duplicated().any():
        raise ValueError("An external subset manifest contains duplicate image IDs.")
    logits = np.asarray(full_logits, dtype=float).ravel()
    if logits.size != len(full_frame):
        raise ValueError("Full external predictions do not match the manifest length.")
    by_image_id = pd.Series(logits, index=full_ids)
    missing = subset_ids[~subset_ids.isin(by_image_id.index)].tolist()
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"External subset contains image IDs absent from full: {preview}"
        )
    return by_image_id.loc[subset_ids].to_numpy()


def evaluate_subset(
    *,
    frame: pd.DataFrame,
    logits: np.ndarray,
    manifest_path: Path,
    external: ExternalConfig,
    locked: LockedOperatingPoint,
    subset: str,
    predictions_dir: Path,
    internal_seed: int,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Score and save predictions for one pre-registered subset."""
    audit = external_audit(frame, logits, locked)
    record = audit.record
    run_name = _subset_name(external.run_name, subset)
    path = Path(predictions_dir) / f"{run_name}.test.csv"
    write_predictions_atomic(
        build_prediction_frame(
            frame,
            run_name=run_name,
            split="test",
            logits=logits,
            probabilities=audit.probability,
            calibrated_probabilities=audit.calibrated_probability,
            threshold=locked.youden_threshold,
            fixed_specificity_target=locked.fixed_specificity_target,
            fixed_specificity_threshold=locked.fixed_specificity_threshold,
            # The seed identifies the trained model being transferred, so it is
            # the internal run's seed. External inference is deterministic.
            seed=internal_seed,
            checkpoint_sha256=locked.checkpoint_sha256,
        ),
        path,
    )
    record["prediction_file"] = str(path)
    record["manifest"] = str(manifest_path)
    record["run_name"] = run_name
    record["subset"] = subset
    # Use the same patient-level bootstrap as the internal evaluation.
    record["intervals"] = model_intervals(
        read_predictions(path), n_resamples=n_resamples, seed=seed
    )
    return record


def main(
    external_config: Path,
    *,
    metrics_path: Path = Path("results/metrics.json"),
    output_path: Path = DEFAULT_OUTPUT,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    setup_logging()
    external = load_external_config(external_config)
    internal = load_config(external.internal_config)
    device = get_device()

    checkpoint = internal.output_dir / f"{external.internal_run}.pt"
    sidecar = internal.output_dir / f"{external.internal_run}.threshold.json"
    locked = load_locked_operating_point(
        metrics_path=metrics_path,
        run_name=external.internal_run,
        checkpoint_path=checkpoint,
        threshold_sidecar=sidecar,
    )
    LOGGER.info(
        "Locked operating point from %s: youden=%.4f, fixed=%.4f (target %.2f), T=%.4f",
        external.internal_run,
        locked.youden_threshold,
        locked.fixed_specificity_threshold,
        locked.fixed_specificity_target,
        locked.temperature,
    )

    model = build_model(
        internal.model.name,
        pretrained=False,
        dropout_conv=internal.model.dropout_conv,
        dropout_head=internal.model.dropout_head,
        head_hidden=internal.model.head_hidden,
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model = model.to(device)

    dataset = MammogramDataset(
        external.manifest,
        external.image_root,
        transform=val_augment(external.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=external.batch_size,
        shuffle=False,
        num_workers=external.num_workers,
    )
    labels, full_logits = _predict_logits(model, loader, device)
    manifest_labels = dataset.df["label"].to_numpy(dtype=int)
    if not np.array_equal(labels.astype(int), manifest_labels):
        raise ValueError("External predictions lost full-manifest order.")

    subsets: list[tuple[str, Path, pd.DataFrame, np.ndarray]] = [
        ("full", external.manifest, dataset.df, full_logits)
    ]
    if external.lesion_present_manifest is not None:
        subset_frame = _manifest.read(external.lesion_present_manifest)
        subset_logits = _align_subset_logits(dataset.df, full_logits, subset_frame)
        subsets.append(
            (
                "lesion_present",
                external.lesion_present_manifest,
                subset_frame,
                subset_logits,
            )
        )

    results: dict[str, dict[str, object]] = {}
    prediction_paths: list[Path] = []
    for subset, manifest_path, frame, logits in subsets:
        LOGGER.info("Cold external scoring on %s subset: %s", subset, manifest_path)
        record = evaluate_subset(
            frame=frame,
            logits=logits,
            manifest_path=manifest_path,
            external=external,
            locked=locked,
            subset=subset,
            predictions_dir=predictions_dir,
            internal_seed=internal.seed,
            n_resamples=n_resamples,
            seed=seed,
        )
        results[subset] = record
        prediction_paths.append(Path(str(record["prediction_file"])))
        panel = cast(dict[str, float], record["test"])
        LOGGER.info(
            "%s: AUC %.4f, sens %.4f, spec %.4f at the transferred Youden threshold",
            subset,
            panel["auc"],
            panel["sensitivity"],
            panel["specificity"],
        )

    manifest_paths = [path for _, path, _, _ in subsets]
    if external.manifest_lock is not None:
        manifest_paths.append(external.manifest_lock)
    payload: dict[str, object] = {
        "version": EXTERNAL_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": external.dataset,
        "role": "cold_external_test",
        "protocol": {
            "fitted_on_external": "nothing",
            "model_source": external.internal_run,
            "threshold_source": "internal_validation_fold",
            "calibration_source": "internal_validation_fold",
            "subsets_pre_registered_before_inference": subset_names(subsets),
            "label_construct": (
                "INbreast BI-RADS assessment, not biopsy-confirmed pathology as "
                "in CBIS-DDSM training labels"
            ),
        },
        "locked_operating_point": locked.to_dict(),
        "subsets": results,
        "provenance": build_run_provenance(
            config_path=external_config,
            checkpoint_paths=[checkpoint],
            manifest_paths=manifest_paths,
            threshold_path=sidecar,
            prediction_paths=prediction_paths,
            additional_preprocessing_paths=[
                Path("src/data/inbreast.py"),
                Path("src/data/inbreast_roi.py"),
            ],
            additional_evaluation_paths=[Path("src/evaluation/external.py")],
            extra={
                "run_name": external.run_name,
                "internal_run": external.internal_run,
                "image_size": external.image_size,
                "image_root": str(external.image_root),
                "threshold_source": "internal_metrics_and_sidecar_cross_checked",
            },
        ),
    }
    write_json_atomic(Path(output_path), payload)
    LOGGER.info("Wrote cold external results to %s", output_path)
    return payload


@click.command()
@click.option(
    "--config",
    "external_config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/inbreast_external.toml"),
    show_default=True,
)
@click.option(
    "--metrics",
    "metrics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/metrics.json"),
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
@click.option(
    "--predictions-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PREDICTIONS_DIR,
    show_default=True,
)
@click.option(
    "--n-resamples", type=click.IntRange(min=1), default=2000, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
def cli(
    external_config: Path,
    metrics_path: Path,
    output_path: Path,
    predictions_dir: Path,
    n_resamples: int,
    seed: int,
) -> None:
    try:
        main(
            external_config,
            metrics_path=metrics_path,
            output_path=output_path,
            predictions_dir=predictions_dir,
            n_resamples=n_resamples,
            seed=seed,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
