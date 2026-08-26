"""Stage 0 patch-classifier entry-point. One config in, one checkpoint out.

This trains the five-class lesion/background head whose convolutional weights
the patch-transfer comparison moves into the 448-pixel whole-image classifier.

It is deliberately a separate entry-point from `src.training.train`: the patch
task is multi-class, has no operating threshold to persist and has no test
fold. It reuses that module's optimiser, scheduler and worker-seeding helpers
unchanged, and repeats its staged-unfreezing shape, so the patch run and the
locked whole-image runs remain directly comparable.

Patch classification is an intermediate diagnostic. A high patch score is not
itself project success.
"""

import dataclasses
import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    PatchConfig,
    get_device,
    load_patch_config,
    set_global_seed,
    setup_logging,
)
from src.data.augment import train_augment, val_augment
from src.data.manifest import assert_patient_disjoint
from src.data.patch_dataset import PatchDataset
from src.data.patch_manifest import PATCH_CLASSES
from src.evaluation.metrics import PatchMetricPanel, evaluate_patches
from src.models import build_model
from src.models.transfer import (
    ARCHS,
    freeze_backbone,
    unfreeze_head,
    unfreeze_top_blocks,
)
from src.training.callbacks import BestAUCCheckpoint, EarlyStopping
from src.training.loss import make_patch_criterion
from src.training.sampler import balanced_sampler
from src.training.train import _build_optimiser, _build_scheduler, _seed_worker

LOGGER = logging.getLogger(__name__)


def _preflight(cfg: PatchConfig) -> None:
    """Enforce the registered patch leakage rules before any weight moves.

    Rules 1, 2 and 5 of the patch contract put every patch from one patient in
    exactly one split; rule 3 keeps test patients out of the patch task
    entirely. Both are cheap to check and expensive to discover later.
    """
    frames = {
        "patch train": pd.read_csv(cfg.data.train_csv),
        "patch val": pd.read_csv(cfg.data.val_csv),
    }
    assert_patient_disjoint(frames)
    if cfg.data.exclusion_test_csv is None:
        LOGGER.warning(
            "No exclusion_test_csv configured; skipping the test-patient check."
        )
        return
    test_patients = set(
        pd.read_csv(cfg.data.exclusion_test_csv)["patient_id"].astype(str)
    )
    for name, frame in frames.items():
        overlap = set(frame["patient_id"].astype(str)) & test_patients
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"{name} manifest contains locked test patient(s): {examples}"
            )
    LOGGER.info(
        "Preflight passed: %d train and %d val patches, patient-disjoint, "
        "no locked test patients.",
        len(frames["patch train"]),
        len(frames["patch val"]),
    )


def _build_loaders(cfg: PatchConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = PatchDataset(
        cfg.data.train_csv,
        cfg.data.patch_root,
        transform=train_augment(
            cfg.data.patch_size, level=cfg.data.augment, seed=cfg.seed
        ),
    )
    val_ds = PatchDataset(
        cfg.data.val_csv,
        cfg.data.patch_root,
        transform=val_augment(cfg.data.patch_size),
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    sampler = None
    shuffle = True
    if cfg.train.sampler == "balanced":
        sampler = balanced_sampler(
            train_ds.df["class_id"].tolist(), generator=generator
        )
        shuffle = False  # mutually exclusive with a sampler
    elif cfg.train.sampler != "shuffle":
        raise ValueError(
            f"Unknown train.sampler {cfg.train.sampler!r}; "
            "expected 'shuffle' or 'balanced'."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
        multiprocessing_context="spawn" if cfg.data.num_workers > 0 else None,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
        multiprocessing_context="spawn" if cfg.data.num_workers > 0 else None,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    running, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()
        running += float(loss.item()) * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


@torch.no_grad()
def _predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true class ids, predicted class ids) over a loader."""
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy().ravel())
        ys.append(y.cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(preds)


def _selection_score(panel: PatchMetricPanel, metric: str) -> float:
    if metric == "macro_f1":
        return panel.macro_f1
    if metric == "balanced_accuracy":
        return panel.balanced_accuracy
    raise ValueError(f"Unknown selection metric {metric!r}")


def _fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    ckpt: BestAUCCheckpoint,
    stopper: EarlyStopping | None,
    selection_metric: str,
    scheduler: object = None,
    start_epoch: int = 0,
    grad_clip: float | None = None,
) -> list[dict]:
    """Train for `epochs`, checkpointing on the configured selection metric.

    `BestAUCCheckpoint` and `EarlyStopping` compare a scalar where higher is
    better, so they carry macro F1 here without modification.
    """
    history: list[dict] = []
    for step in range(epochs):
        epoch = start_epoch + step
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            grad_clip=grad_clip,
        )
        val_y, val_pred = _predict(model, val_loader, device)
        panel = evaluate_patches(val_y, val_pred, PATCH_CLASSES)
        score = _selection_score(panel, selection_metric)
        improved = ckpt(score, model)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_macro_f1=%.4f val_bal_acc=%.4f "
            "val_acc=%.4f%s",
            epoch,
            train_loss,
            panel.macro_f1,
            panel.balanced_accuracy,
            panel.accuracy,
            " *" if improved else "",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": panel.macro_f1,
                "val_balanced_accuracy": panel.balanced_accuracy,
                "val_accuracy": panel.accuracy,
                "val_sensitivity": panel.per_class_sensitivity,
                "selection_score": score,
                "selected": improved,
            }
        )
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()  # type: ignore[attr-defined]
        if stopper is not None and stopper(score):
            LOGGER.info("Early stopping at epoch %d", epoch)
            break
    return history


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _report(
    cfg: PatchConfig, panel: PatchMetricPanel, history: list[dict]
) -> dict[str, object]:
    selected = [entry for entry in history if entry["selected"]]
    return {
        "schema_version": 1,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "task": "stage0_five_class_patch",
        "class_names": list(PATCH_CLASSES),
        "selection_metric": cfg.train.selection_metric,
        "selected_epoch": selected[-1]["epoch"] if selected else None,
        "n_epochs_run": len(history),
        "validation": {
            "accuracy": panel.accuracy,
            "balanced_accuracy": panel.balanced_accuracy,
            "macro_f1": panel.macro_f1,
            "per_class_sensitivity": panel.per_class_sensitivity,
            "per_class_precision": panel.per_class_precision,
            "confusion": panel.confusion.tolist(),
            "confusion_axes": "rows=true class, columns=predicted class",
            "lesion_pair_confusion": panel.lesion_pair_confusion,
        },
        "inputs": {
            "train_csv": str(cfg.data.train_csv),
            "val_csv": str(cfg.data.val_csv),
            "patch_root": str(cfg.data.patch_root),
        },
    }


def main(
    config_path: Path, *, seed: int | None = None, run_name: str | None = None
) -> None:
    setup_logging()
    cfg = load_patch_config(config_path)
    cfg = dataclasses.replace(
        cfg,
        seed=cfg.seed if seed is None else seed,
        run_name=cfg.run_name if run_name is None else run_name,
    )
    _preflight(cfg)
    set_global_seed(cfg.seed)

    device = get_device()
    LOGGER.info("Using device %s", device)

    train_loader, val_loader = _build_loaders(cfg)
    model = build_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        dropout_conv=cfg.model.dropout_conv,
        dropout_head=cfg.model.dropout_head,
        head_hidden=cfg.model.head_hidden,
        num_classes=len(PATCH_CLASSES),
    ).to(device)

    criterion: torch.nn.Module
    if cfg.train.class_weighted_loss:
        criterion = make_patch_criterion(
            cfg.data.train_csv, device, label_smoothing=cfg.train.label_smoothing
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    LOGGER.info(
        "Loss: %s, sampler: %s",
        "class-weighted cross-entropy"
        if cfg.train.class_weighted_loss
        else "unweighted cross-entropy",
        cfg.train.sampler,
    )

    ckpt = BestAUCCheckpoint(cfg.output_dir / f"{cfg.run_name}.pt")
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience or 10)

    history: list[dict] = []
    if cfg.model.name.lower() in ARCHS and cfg.model.pretrained:
        freeze_backbone(model)
        unfreeze_head(model)
        optimiser = _build_optimiser(model, cfg.train)
        LOGGER.info("Stage 1: head-only for %d epochs", cfg.train.stage1_epochs)
        history += _fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimiser,
            device,
            cfg.train.stage1_epochs,
            ckpt,
            stopper=None,
            selection_metric=cfg.train.selection_metric,
            grad_clip=cfg.train.grad_clip,
        )
        unfreeze_top_blocks(model, cfg.model.name)
        stage2_cfg = dataclasses.replace(cfg.train, lr=cfg.train.stage2_lr)
        optimiser = _build_optimiser(model, stage2_cfg)
        remaining = max(cfg.train.epochs - cfg.train.stage1_epochs, 1)
        scheduler = _build_scheduler(optimiser, stage2_cfg, remaining)
        LOGGER.info(
            "Stage 2: top blocks unfrozen for %d epochs at lr=%g",
            remaining,
            cfg.train.stage2_lr,
        )
        history += _fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimiser,
            device,
            remaining,
            ckpt,
            stopper,
            selection_metric=cfg.train.selection_metric,
            scheduler=scheduler,
            start_epoch=cfg.train.stage1_epochs,
            grad_clip=cfg.train.grad_clip,
        )
    else:
        optimiser = _build_optimiser(model, cfg.train)
        scheduler = _build_scheduler(optimiser, cfg.train, cfg.train.epochs)
        history += _fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimiser,
            device,
            cfg.train.epochs,
            ckpt,
            stopper,
            selection_metric=cfg.train.selection_metric,
            scheduler=scheduler,
            grad_clip=cfg.train.grad_clip,
        )

    _save_json(cfg.output_dir / f"{cfg.run_name}.history.json", history)

    # Score the selected checkpoint, not the final epoch: the transfer takes
    # these weights, so the reported panel must describe the weights on disk.
    best_path = cfg.output_dir / f"{cfg.run_name}.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    val_y, val_pred = _predict(model, val_loader, device)
    panel = evaluate_patches(val_y, val_pred, PATCH_CLASSES)
    _save_json(
        cfg.output_dir / f"{cfg.run_name}.patch-metrics.json",
        _report(cfg, panel, history),
    )
    LOGGER.info(
        "Patch training done. Selected checkpoint: macro F1 = %.4f, "
        "balanced accuracy = %.4f",
        panel.macro_f1,
        panel.balanced_accuracy,
    )
    for name, value in panel.per_class_sensitivity.items():
        LOGGER.info("  sensitivity %-24s %.4f", name, value)
    for lesion_type, stats in panel.lesion_pair_confusion.items():
        LOGGER.info(
            "  %s malignancy accuracy within pair = %.4f (%d off-pair)",
            lesion_type,
            stats["malignancy_accuracy_in_pair"],
            int(stats["n_predicted_off_pair"]),
        )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("configs/patch_learning/vgg16_patch.toml"),
    show_default=True,
)
@click.option("--seed", type=int, help="Override the config seed.")
@click.option("--run-name", help="Save this run under a different name.")
def cli(config_path: Path, seed: int | None, run_name: str | None) -> None:
    """Train the Stage 0 five-class patch classifier."""
    main(config_path, seed=seed, run_name=run_name)


if __name__ == "__main__":
    cli()
