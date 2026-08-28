"""CLI evaluation entry-point. Reads a trained checkpoint and appends a
metrics record to results/metrics.json.
"""

import dataclasses
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.config import Config, get_device, load_config, setup_logging
from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.data.manifest import validate_split_paths
from src.evaluation.audit import build_audit, logits_to_probability
from src.evaluation.lineage import build_run_lineage, sha256_file
from src.evaluation.metrics import evaluate
from src.evaluation.predictions import (
    PREDICTIONS_DIR,
    build_prediction_frame,
    prediction_path,
    write_predictions_atomic,
)
from src.evaluation.results_io import upsert_run_record
from src.models import build_model

LOGGER = logging.getLogger(__name__)

METRICS_PATH = Path("results/metrics.json")


def _load_threshold(cfg: Config) -> float:
    sidecar = cfg.output_dir / f"{cfg.run_name}.threshold.json"
    return float(json.loads(sidecar.read_text())["youden_j"])


def _predict_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Collect labels and logits from a loader."""
    model.eval()
    ys, ls = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            ls.append(model(x).cpu().numpy().ravel())
            ys.append(y.cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(ls)


def _gradcam_roi_panel(
    model: torch.nn.Module,
    test_ds,
    model_name: str,
    y_prob: np.ndarray,
    threshold: float,
    device: torch.device,
) -> tuple[dict | None, str | None]:
    """Compute Grad-CAM localisation metrics for malignant cases."""
    from src.evaluation.gradcam import TARGET_LAYERS, compute_gradcam
    from src.evaluation.gradcam_roi import grad_cam_subset_stats, is_degenerate

    target_layer = TARGET_LAYERS.get(model_name.lower())
    if target_layer is None:
        return None, f"no Grad-CAM target layer registered for model '{model_name}'"
    mal_idx = np.where(test_ds.df["label"].values == 1)[0]
    cams: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    correct: list[bool] = []
    n_no_roi = 0
    n_degenerate = 0
    for i in mal_idx:
        image, _ = test_ds[int(i)]
        cam = compute_gradcam(model, image.unsqueeze(0).to(device), target_layer)
        roi = test_ds.load_roi(int(i), cam.shape)
        if roi is None:
            n_no_roi += 1
            continue
        if is_degenerate(cam):
            n_degenerate += 1
            continue
        cams.append(cam)
        rois.append(roi)
        correct.append(bool(y_prob[i] >= threshold))
    if not cams:
        if n_degenerate:
            return None, (
                f"{n_degenerate} malignant case(s) had a zero-energy Grad-CAM "
                "heatmap (degenerate rectifier output) and the remaining "
                f"{n_no_roi} had no usable ROI mask"
            )
        return (
            None,
            f"no malignant test case has a usable ROI mask ({n_no_roi} checked)",
        )
    correct_arr = np.array(correct, dtype=bool)
    n = len(cams)
    return {
        "n_malignant_scored": n,
        "n_degenerate_excluded": n_degenerate,
        "n_no_roi_excluded": n_no_roi,
        "all": grad_cam_subset_stats(cams, rois, np.ones(n, dtype=bool)),
        "tp": grad_cam_subset_stats(cams, rois, correct_arr),
        "fn": grad_cam_subset_stats(cams, rois, ~correct_arr),
    }, None


def _append_record(record: dict[str, object], path: Path = METRICS_PATH) -> None:
    upsert_run_record(record, path)


def main(
    config_path: Path,
    *,
    seed: int | None = None,
    run_name: str | None = None,
    metrics_path: Path | None = None,
    predictions_dir: Path | None = None,
) -> None:
    """Evaluate one run and write its metrics and predictions."""
    setup_logging()
    cfg = load_config(config_path)
    cfg = dataclasses.replace(
        cfg,
        seed=cfg.seed if seed is None else seed,
        run_name=cfg.run_name if run_name is None else run_name,
    )
    validate_split_paths(cfg.data.train_csv, cfg.data.val_csv, cfg.data.test_csv)
    device = get_device()

    weights_path = cfg.output_dir / f"{cfg.run_name}.pt"

    model = build_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        dropout_conv=cfg.model.dropout_conv,
        dropout_head=cfg.model.dropout_head,
        head_hidden=cfg.model.head_hidden,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    test_ds = MammogramDataset(
        cfg.data.test_csv,
        cfg.data.image_root,
        transform=val_augment(cfg.data.image_size),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    y_true, test_logits = _predict_logits(model, test_loader, device)
    if not np.array_equal(y_true.astype(int), test_ds.df["label"].to_numpy(dtype=int)):
        raise ValueError("Test predictions do not match manifest order.")
    y_prob = logits_to_probability(test_logits)
    threshold = _load_threshold(cfg)
    panel = evaluate(y_true, y_prob, threshold=threshold)
    LOGGER.info("Test panel: %s", panel)

    record: dict = {
        "model": cfg.run_name,
        "val_threshold": threshold,
        "test": {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()},
    }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    record["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    val_ds = MammogramDataset(
        cfg.data.val_csv,
        cfg.data.image_root,
        transform=val_augment(cfg.data.image_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    val_true, val_logits = _predict_logits(model, val_loader, device)
    if not np.array_equal(val_true.astype(int), val_ds.df["label"].to_numpy(dtype=int)):
        raise ValueError("Validation predictions do not match manifest order.")
    audit = build_audit(
        val_ds.df,
        val_logits,
        test_ds.df,
        test_logits,
        operating_threshold=threshold,
    )
    record.update(audit.record)

    gradcam_roi, gradcam_skip_reason = _gradcam_roi_panel(
        model, test_ds, cfg.model.name, audit.test_probability, threshold, device
    )
    if gradcam_roi is not None:
        record["gradcam_roi"] = gradcam_roi
    else:
        LOGGER.warning("Grad-CAM-ROI (novelty A) skipped: %s", gradcam_skip_reason)

    checkpoint_hash = sha256_file(weights_path)
    predictions_root = predictions_dir or PREDICTIONS_DIR
    validation_predictions = prediction_path(
        cfg.run_name, "validation", root=predictions_root
    )
    test_predictions = prediction_path(cfg.run_name, "test", root=predictions_root)
    write_predictions_atomic(
        build_prediction_frame(
            val_ds.df,
            run_name=cfg.run_name,
            split="validation",
            logits=val_logits,
            probabilities=audit.validation_probability,
            calibrated_probabilities=audit.validation_calibrated_probability,
            threshold=threshold,
            fixed_specificity_target=audit.fixed_specificity_target,
            fixed_specificity_threshold=audit.fixed_specificity_threshold,
            seed=cfg.seed,
            checkpoint_sha256=checkpoint_hash,
        ),
        validation_predictions,
    )
    write_predictions_atomic(
        build_prediction_frame(
            test_ds.df,
            run_name=cfg.run_name,
            split="test",
            logits=test_logits,
            probabilities=audit.test_probability,
            calibrated_probabilities=audit.test_calibrated_probability,
            threshold=threshold,
            fixed_specificity_target=audit.fixed_specificity_target,
            fixed_specificity_threshold=audit.fixed_specificity_threshold,
            seed=cfg.seed,
            checkpoint_sha256=checkpoint_hash,
        ),
        test_predictions,
    )

    record["lineage"] = build_run_lineage(
        config_path=config_path,
        checkpoint_paths=[weights_path],
        manifest_paths=[cfg.data.train_csv, cfg.data.val_csv, cfg.data.test_csv],
        threshold_path=cfg.output_dir / f"{cfg.run_name}.threshold.json",
        prediction_paths=[validation_predictions, test_predictions],
        extra={
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "image_size": cfg.data.image_size,
            "image_root": str(cfg.data.image_root),
            "threshold_source": "validation_sidecar",
        },
    )

    _append_record(record, metrics_path or METRICS_PATH)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="TOML experiment config.",
)
@click.option("--seed", type=int, help="Override the config seed.")
@click.option("--run-name", help="Read and save this run under a different name.")
@click.option(
    "--metrics-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the run record here instead of results/metrics.json.",
)
@click.option(
    "--predictions-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write prediction CSVs here instead of results/predictions.",
)
def cli(
    config_path: Path,
    seed: int | None,
    run_name: str | None,
    metrics_path: Path | None,
    predictions_dir: Path | None,
) -> None:
    main(
        config_path,
        seed=seed,
        run_name=run_name,
        metrics_path=metrics_path,
        predictions_dir=predictions_dir,
    )


if __name__ == "__main__":
    cli()
