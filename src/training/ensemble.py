"""Ensemble evaluation entry-point.

Loads each member checkpoint listed in an ensemble config, averages their
sigmoid probabilities over the test set, and appends a metrics record to
results/metrics.json.
"""

import json
import logging
import tomllib
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from src.config import get_device, setup_logging
from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.metrics import evaluate, youden_threshold
from src.models import build_model
from src.models.ensemble import ensemble_predict

LOGGER = logging.getLogger(__name__)

METRICS_PATH = Path("results/metrics.json")


def _load_member(name: str, output_dir: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(name.replace("_imagenet", ""), pretrained=False)
    weights = output_dir / f"{name}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}. Train {name} first.")
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _append_record(record: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"runs": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    data.setdefault("runs", []).append(record)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(config_path: Path) -> None:
    setup_logging()
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    output_dir = Path(cfg.get("output_dir", "models"))
    members: list[str] = cfg["members"]
    data_cfg = cfg["data"]
    image_size: int = data_cfg.get("image_size", 224)
    test_csv = Path(data_cfg["test_csv"])
    image_root = Path(data_cfg["image_root"])

    device = get_device()
    LOGGER.info("Using device %s", device)

    models = []
    for name in members:
        LOGGER.info("Loading member %s", name)
        models.append(_load_member(name, output_dir, device))

    test_ds = MammogramDataset(test_csv, image_root, transform=val_augment(image_size))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    y_true = test_ds.df["label"].values.astype(int)
    y_prob = ensemble_predict(models, test_loader, device)

    # The operating threshold MUST come from validation, never the test set, to
    # match the single-model discipline in evaluate.py. Deriving Youden's J on
    # the test labels and then scoring those same labels is test-set leakage
    # that inflates sensitivity/specificity/PPV/F1 (AUC, being threshold-free,
    # is unaffected).
    val_csv = data_cfg.get("val_csv")
    if val_csv:
        val_ds = MammogramDataset(
            Path(val_csv), image_root, transform=val_augment(image_size)
        )
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        val_true = val_ds.df["label"].values.astype(int)
        val_prob = ensemble_predict(models, val_loader, device)
        threshold = youden_threshold(val_true, val_prob)
    else:
        LOGGER.warning(
            "No val_csv in ensemble config, falling back to a "
            "test-derived threshold, which leaks the test set into "
            "the operating point. Add val_csv to fix."
        )
        threshold = youden_threshold(y_true, y_prob)
    panel = evaluate(y_true, y_prob, threshold=threshold)
    LOGGER.info("Ensemble test panel: %s", panel)

    # Store ROC points so the ensemble appears in the model-comparison overlay
    # alongside the single models (make_figures skips records without them).
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    import dataclasses

    record = {
        "model": cfg.get("run_name", "ensemble"),
        "members": members,
        "val_threshold": float(threshold),
        "test": {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
    _append_record(record)
    LOGGER.info("Ensemble done. Test AUC = %.4f", panel.auc)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="YAML ensemble config.",
)
def cli(config_path: Path) -> None:
    main(config_path)


if __name__ == "__main__":
    cli()
