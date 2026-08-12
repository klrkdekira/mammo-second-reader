"""Fine-tune a model from the web app."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import get_device
from src.data import manifest
from src.data.augment import train_augment, val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.metrics import evaluate
from src.models import build_model
from src.models.transfer import freeze_backbone as freeze_model_backbone
from src.models.transfer import unfreeze_head
from src.training.callbacks import BestAUCCheckpoint
from src.training.loss import make_criterion
from src.training.train import _predict, _train_one_epoch
from src.web.archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_FILES,
    MAX_EXTRACTED_BYTES,
    deidentify_dicom_in_place,
    extract_flat_archive,
    move_file,
)


def materialise_workdir(zip_path: str, workdir: Path) -> Path:
    """Unpack the training and validation files."""
    workdir = Path(workdir)
    staging = workdir / ".extracting"
    extracted = extract_flat_archive(
        zip_path,
        staging,
        max_archive_bytes=MAX_ARCHIVE_BYTES,
        max_extracted_bytes=MAX_EXTRACTED_BYTES,
        max_files=MAX_ARCHIVE_FILES,
    )
    by_name = {path.name: path for path in extracted}
    if "train.csv" not in by_name or "val.csv" not in by_name:
        raise ValueError("Fine-tuning archives must contain train.csv and val.csv.")

    workdir.mkdir(parents=True, exist_ok=True)
    train_csv = move_file(by_name.pop("train.csv"), workdir / "train.csv")
    val_csv = move_file(by_name.pop("val.csv"), workdir / "val.csv")
    manifest.read(train_csv)
    manifest.read(val_csv)

    processed = workdir / "processed"
    for path in by_name.values():
        target = move_file(path, processed / path.name)
        if target.suffix.lower() == ".dcm":
            deidentify_dicom_in_place(target)
    staging.rmdir()
    return workdir


def _clean_model_name(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("_imagenet", "")
        .replace("_transfer", "")
        .replace("_scratch", "")
    )


def stream_finetune_epochs(
    workdir: Path,
    model_name: str,
    base_checkpoint: Path,
    *,
    epochs: int = 5,
    lr: float = 1e-5,
    freeze_backbone: bool = True,
    image_size: int = 224,
    batch_size: int = 16,
) -> Iterator[dict[str, float | int]]:
    """Fine-tune a model and return metrics after each epoch."""
    workdir = Path(workdir)
    if not 1 <= int(epochs) <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if not 0.0 < float(lr) <= 0.1:
        raise ValueError("lr must be greater than 0 and no more than 0.1")
    train_csv = workdir / "train.csv"
    val_csv = workdir / "val.csv"
    image_root = workdir / "processed"
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    device = get_device()
    clean_name = _clean_model_name(model_name)
    model = build_model(clean_name, pretrained=False)
    state = torch.load(base_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    if freeze_backbone and hasattr(model, "backbone"):
        freeze_model_backbone(model)
        unfreeze_head(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("The selected freeze policy leaves no trainable parameters.")

    train_loader = DataLoader(
        MammogramDataset(train_csv, image_root, transform=train_augment(image_size)),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MammogramDataset(val_csv, image_root, transform=val_augment(image_size)),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )
    optimiser = torch.optim.Adam(trainable, lr=float(lr))
    criterion = make_criterion(train_csv, device)
    checkpoint = BestAUCCheckpoint(workdir / "adapter.pt")
    history: list[dict[str, float | int]] = []

    for epoch in range(int(epochs)):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_y, val_p = _predict(model, val_loader, device)
        panel = evaluate(val_y, val_p)
        checkpoint(panel.auc, model)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(panel.auc),
            "val_sensitivity": float(panel.sensitivity),
            "val_specificity": float(panel.specificity),
        }
        history.append(record)
        (workdir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        yield record
