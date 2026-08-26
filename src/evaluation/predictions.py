"""Save case-level predictions for later statistical analysis."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.lineage import sha256_file

PREDICTIONS_DIR = Path("results/predictions")
METADATA_COLUMNS = (
    "image_id",
    "patient_id",
    "birads_density",
    "lesion_type",
)


def checkpoint_set_hash(paths: Iterable[Path]) -> str:
    """Hash an ordered set of checkpoint hashes."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(sha256_file(Path(path)).encode())
    return digest.hexdigest()


def prediction_path(run_name: str, split: str, root: Path = PREDICTIONS_DIR) -> Path:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    return Path(root) / f"{run_name}.{split}.csv"


def build_prediction_frame(
    manifest: pd.DataFrame,
    *,
    run_name: str,
    split: str,
    logits: np.ndarray,
    probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
    fixed_specificity_target: float,
    fixed_specificity_threshold: float,
    seed: int,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    """Join predictions with the identifiers needed for paired analysis."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    for column in ("image_id", "patient_id", "label"):
        if column not in manifest.columns or manifest[column].isna().any():
            raise ValueError(f"{column} is required for statistical analysis.")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Image IDs must be unique within a prediction file.")
    values = {
        "logit": np.asarray(logits, dtype=float).ravel(),
        "probability": np.asarray(probabilities, dtype=float).ravel(),
        "calibrated_probability": np.asarray(
            calibrated_probabilities, dtype=float
        ).ravel(),
    }
    if any(array.size != len(manifest) for array in values.values()):
        raise ValueError("Prediction counts do not match the manifest.")
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Predictions must be finite.")
    for name in ("probability", "calibrated_probability"):
        if not np.all((values[name] >= 0.0) & (values[name] <= 1.0)):
            raise ValueError(f"{name} must be between 0 and 1.")

    frame = pd.DataFrame(index=manifest.index)
    for column in METADATA_COLUMNS:
        frame[column] = manifest[column] if column in manifest.columns else pd.NA
    frame.insert(0, "split", split)
    frame.insert(0, "run_name", run_name)
    frame["label"] = np.asarray(manifest["label"], dtype=int)
    for name, array in values.items():
        frame[name] = array
    frame["threshold"] = float(threshold)
    frame["fixed_specificity_target"] = float(fixed_specificity_target)
    frame["fixed_specificity_threshold"] = float(fixed_specificity_threshold)
    frame["predicted_label"] = (frame["probability"] >= threshold).astype(int)
    frame["seed"] = int(seed)
    frame["checkpoint_sha256"] = checkpoint_sha256
    return frame.reset_index(drop=True)


def write_predictions_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a prediction CSV without leaving a partial result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
