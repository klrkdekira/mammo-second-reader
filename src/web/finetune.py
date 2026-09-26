"""Fine-tune a model from the web app."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import get_device
from src.data import manifest
from src.data.augment import train_augment, val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.metrics import evaluate, youden_threshold
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
from src.web import inference
from src.web.inference import (
    available_models,
    checkpoint_path,
    model_image_size,
    resolve_arch,
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


_OUTPUT_NAME = re.compile(r"^[a-z0-9_]+$")


def default_output_name(base_model: str) -> str:
    return f"{base_model}_finetuned"


def _validate_output_name(base_model: str, output_name: str) -> Path:
    """Return the new checkpoint path, refusing to overwrite a trained run."""
    if not _OUTPUT_NAME.match(output_name):
        raise ValueError(
            "Output name may only contain lower-case letters, digits and underscores."
        )
    if resolve_arch(output_name) != resolve_arch(base_model):
        raise ValueError(
            f"Output name {output_name!r} must keep the base run prefix so its "
            "architecture can be resolved, for example "
            f"{default_output_name(base_model)!r}."
        )
    if model_image_size(output_name) != model_image_size(base_model):
        raise ValueError("Output name must keep the base run's resolution suffix.")
    target = inference.MODEL_DIR / f"{output_name}.pt"
    if target.exists() or output_name in available_models():
        raise ValueError(
            f"A checkpoint named {output_name!r} already exists. Choose another name."
        )
    return target


def stream_finetune_epochs(
    workdir: Path,
    base_model: str,
    output_name: str,
    *,
    epochs: int = 5,
    lr: float = 1e-5,
    freeze_backbone: bool = True,
    batch_size: int = 16,
) -> Iterator[dict[str, object]]:
    """Fine-tune a trained checkpoint and return metrics after each epoch.

    The best-AUC weights are saved to models/<output_name>.pt with history
    and threshold sidecars, so the result appears in the Inference tab.
    """
    workdir = Path(workdir)
    if not 1 <= int(epochs) <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if not 0.0 < float(lr) <= 0.1:
        raise ValueError("lr must be greater than 0 and no more than 0.1")
    arch = resolve_arch(base_model)
    base_checkpoint = checkpoint_path(base_model)
    if arch is None or not base_checkpoint.is_file():
        raise FileNotFoundError(f"No trained checkpoint for model {base_model!r}.")
    target = _validate_output_name(base_model, output_name)
    image_size = model_image_size(base_model)
    train_csv = workdir / "train.csv"
    val_csv = workdir / "val.csv"
    image_root = workdir / "processed"

    device = get_device()
    model = build_model(arch, pretrained=False)
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
    # Train into the work directory so an aborted run leaves models/ untouched.
    staged = workdir / "adapter.pt"
    checkpoint = BestAUCCheckpoint(staged)
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

    model.load_state_dict(torch.load(staged, map_location=device, weights_only=True))
    val_y, val_p = _predict(model, val_loader, device)
    best_auc = float(evaluate(val_y, val_p).auc)
    model_dir = inference.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged, target)
    (model_dir / f"{output_name}.history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (model_dir / f"{output_name}.threshold.json").write_text(
        json.dumps(
            {
                "youden_j": youden_threshold(val_y, val_p),
                "val_auc_at_best": best_auc,
                "val_auc_final_epoch": history[-1]["val_auc"],
                "run_name": output_name,
                "base_model": base_model,
            },
            indent=2,
        )
        + "\n"
    )
    yield {**history[-1], "saved_as": output_name, "best_val_auc": best_auc}
