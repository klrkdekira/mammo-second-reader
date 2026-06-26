"""Fine-tune a saved checkpoint; trains the classifier head only by default."""

import json
import logging
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from src.config import setup_logging
from src.data.augment import train_augment, val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.metrics import evaluate
from src.models import build_model
from src.training.callbacks import BestAUCCheckpoint
from src.training.loss import make_criterion
from src.training.train import _fit, _predict

LOGGER = logging.getLogger(__name__)


def main(
    workdir: Path,
    model_name: str,
    epochs: int,
    lr: float,
    freeze_backbone: bool,
    base_checkpoint: Path,
) -> None:
    setup_logging()
    workdir = Path(workdir)
    train_csv = workdir / "train.csv"
    val_csv = workdir / "val.csv"
    image_root = workdir / "processed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=True)
    state = torch.load(base_checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        head = (
            model.backbone.classifier[-1]
            if hasattr(model.backbone, "classifier")
            else model.backbone.fc
        )
        for p in head.parameters():
            p.requires_grad = True

    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = make_criterion(train_csv, device)

    train_loader = DataLoader(
        MammogramDataset(train_csv, image_root, transform=train_augment()),
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        MammogramDataset(val_csv, image_root, transform=val_augment()),
        batch_size=16,
        shuffle=False,
        num_workers=2,
    )

    ckpt = BestAUCCheckpoint(workdir / "adapter.pt")
    history = _fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimiser,
        device,
        epochs,
        ckpt,
        stopper=None,
    )
    (workdir / "history.json").write_text(json.dumps(history, indent=2))

    val_y, val_p = _predict(model, val_loader, device)
    panel = evaluate(val_y, val_p)
    LOGGER.info("Fine-tune done. Val AUC = %.4f", panel.auc)


@click.command()
@click.option(
    "--workdir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory containing train.csv, val.csv, and processed/.",
)
@click.option(
    "--model", "model_name", required=True, help="Model name, e.g. vgg16_imagenet."
)
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option(
    "--freeze-backbone",
    is_flag=True,
    default=False,
    help="Freeze backbone; train head only.",
)
@click.option(
    "--base-checkpoint",
    type=click.Path(path_type=Path),
    required=True,
    help="Checkpoint to fine-tune from.",
)
def cli(
    workdir: Path,
    model_name: str,
    epochs: int,
    lr: float,
    freeze_backbone: bool,
    base_checkpoint: Path,
) -> None:
    main(workdir, model_name, epochs, lr, freeze_backbone, base_checkpoint)


if __name__ == "__main__":
    cli()
