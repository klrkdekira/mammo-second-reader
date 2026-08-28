"""Ensemble evaluation entry-point.

Loads each member checkpoint listed in an ensemble config, averages their
sigmoid probabilities over the test set, and appends a metrics record to
results/metrics.json.
"""

import logging
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.config import get_device, load_ensemble_config, setup_logging
from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.data.manifest import validate_split_paths
from src.evaluation.audit import build_audit, probability_to_logits
from src.evaluation.lineage import build_run_lineage
from src.evaluation.metrics import evaluate, youden_threshold
from src.evaluation.predictions import (
    build_prediction_frame,
    checkpoint_set_hash,
    prediction_path,
    write_predictions_atomic,
)
from src.evaluation.results_io import upsert_run_record
from src.models import build_model
from src.models.ensemble import ensemble_predict

LOGGER = logging.getLogger(__name__)

METRICS_PATH = Path("results/metrics.json")


def _load_member(name: str, output_dir: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(name.replace("_imagenet", ""), pretrained=False)
    weights = output_dir / f"{name}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}. Train {name} first.")
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _append_record(record: dict, path: Path = METRICS_PATH) -> None:
    upsert_run_record(record, path)


def main(config_path: Path) -> None:
    setup_logging()
    cfg = load_ensemble_config(config_path)
    validate_split_paths(cfg.train_csv, cfg.val_csv, cfg.test_csv)

    device = get_device()
    LOGGER.info("Using device %s", device)

    models = []
    for name in cfg.members:
        LOGGER.info("Loading member %s", name)
        models.append(_load_member(name, cfg.output_dir, device))

    test_ds = MammogramDataset(
        cfg.test_csv, cfg.image_root, transform=val_augment(cfg.image_size)
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    y_true = np.asarray(test_ds.df["label"].values, dtype=np.int64)
    y_prob = ensemble_predict(models, test_loader, device)

    val_ds = MammogramDataset(
        cfg.val_csv, cfg.image_root, transform=val_augment(cfg.image_size)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    val_true = np.asarray(val_ds.df["label"].values, dtype=np.int64)
    val_prob = ensemble_predict(models, val_loader, device)
    threshold = float(youden_threshold(val_true, val_prob))
    panel = evaluate(y_true, y_prob, threshold=threshold)
    LOGGER.info("Ensemble test panel: %s", panel)

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    import dataclasses

    record = {
        "model": cfg.run_name,
        "members": cfg.members,
        "val_threshold": float(threshold),
        "test": {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
    val_logits = probability_to_logits(val_prob)
    test_logits = probability_to_logits(y_prob)
    audit = build_audit(
        val_ds.df,
        val_logits,
        test_ds.df,
        test_logits,
        operating_threshold=threshold,
    )
    record.update(audit.record)
    record["gradcam_policy"] = {
        "scope": "member_level",
        "reason": (
            "The probability ensemble has no single convolutional feature map. "
            "Its four members retain separate Grad-CAM audits."
        ),
        "members": cfg.members,
    }

    checkpoint_paths = [cfg.output_dir / f"{name}.pt" for name in cfg.members]
    checkpoint_hash = checkpoint_set_hash(checkpoint_paths)
    validation_predictions = prediction_path(cfg.run_name, "validation")
    test_predictions = prediction_path(cfg.run_name, "test")
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
        checkpoint_paths=checkpoint_paths,
        manifest_paths=[cfg.train_csv, cfg.val_csv, cfg.test_csv],
        prediction_paths=[validation_predictions, test_predictions],
        extra={
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "image_size": cfg.image_size,
            "image_root": str(cfg.image_root),
            "threshold_source": "validation_predictions",
        },
    )
    _append_record(record)
    LOGGER.info("Ensemble done. Test AUC = %.4f", panel.auc)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="TOML ensemble config.",
)
def cli(config_path: Path) -> None:
    main(config_path)


if __name__ == "__main__":
    cli()
