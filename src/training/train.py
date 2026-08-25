"""Training entry-point. One config in, one checkpoint plus history out."""

import dataclasses
import json
import logging
import random
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    Config,
    TrainConfig,
    get_device,
    load_config,
    set_global_seed,
    setup_logging,
)
from src.data.augment import train_augment, val_augment
from src.data.dataset import MammogramDataset
from src.data.manifest import validate_split_paths
from src.evaluation.metrics import evaluate, youden_threshold
from src.models import build_model
from src.models.transfer import (
    ARCHS,
    freeze_backbone,
    unfreeze_head,
    unfreeze_top_blocks,
)
from src.training.callbacks import BestAUCCheckpoint, EarlyStopping
from src.training.loss import make_criterion
from src.training.sampler import balanced_sampler

LOGGER = logging.getLogger(__name__)


def _seed_worker(_worker_id: int) -> None:
    """Seed numpy/random in each DataLoader worker from its torch seed.

    Spawn workers start with fresh numpy/random state, so Albumentations'
    (numpy-backed) random ops would differ run-to-run despite set_global_seed.
    torch assigns each worker a deterministic seed derived from the loader's
    generator. Mirror it into numpy and random so augmentation is reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_ds = MammogramDataset(
        cfg.data.train_csv,
        cfg.data.image_root,
        transform=train_augment(cfg.data.image_size, level=cfg.data.augment),
    )
    val_ds = MammogramDataset(
        cfg.data.val_csv,
        cfg.data.image_root,
        transform=val_augment(cfg.data.image_size),
    )
    # A seeded generator makes both the shuffle order and the worker seeds
    # (and any WeightedRandomSampler draws) deterministic across runs.
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    sampler = None
    shuffle = True
    if cfg.train.sampler == "balanced":
        sampler = balanced_sampler(train_ds.df["label"].tolist(), generator=generator)
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


def _build_optimiser(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimiser.lower() == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimiser.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimiser.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimiser {cfg.optimiser!r}")


def _build_scheduler(optimiser: torch.optim.Optimizer, cfg: TrainConfig, epochs: int):
    """Build an LR scheduler from the config, or return None."""
    if cfg.scheduler is None:
        return None
    s = cfg.scheduler.lower()
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    if s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", patience=5, factor=0.5
        )
    raise ValueError(f"Unknown scheduler {cfg.scheduler!r}")


def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a convex combination of the batch with a shuffled copy of itself.

    Soft targets go straight into BCEWithLogitsLoss, so no per-sample loss
    mixing is needed.

    Interaction with pos_weight: the criterion still applies the fixed
    n_neg/n_pos pos_weight to these interpolated soft targets, so a mixed
    sample's positive term is up-weighted by the full imbalance factor even
    though its target is only fractionally positive. The effect is small and
    consistent across runs. Configs combining mixup and pos_weight (e.g.
    regularised_combined) should note it rather than read the two as
    independent. See WARNINGS / writeup.
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], lam * y + (1.0 - lam) * y[perm]


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    mixup_alpha: float = 0.0,
) -> float:
    model.train()
    running, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_alpha > 0.0:
            x, y = _mixup(x, y, mixup_alpha)
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
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ps.append(torch.sigmoid(logits).cpu().numpy().ravel())
        ys.append(y.cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(ps)


def _save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n")


def _persist_val_threshold(
    cfg: Config, threshold: float, val_auc: float, final_val_auc: float | None = None
) -> None:
    sidecar = cfg.output_dir / f"{cfg.run_name}.threshold.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "youden_j": threshold,
                "val_auc_at_best": val_auc,
                "val_auc_final_epoch": final_val_auc,
                "run_name": cfg.run_name,
            },
            indent=2,
        )
        + "\n"
    )


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
    scheduler=None,
    start_epoch: int = 0,
    grad_clip: float | None = None,
    mixup_alpha: float = 0.0,
) -> list[dict]:
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
            mixup_alpha=mixup_alpha,
        )
        val_y, val_p = _predict(model, val_loader, device)
        panel = evaluate(val_y, val_p)
        improved = ckpt(panel.auc, model)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_auc=%.4f val_sens=%.4f val_spec=%.4f%s",
            epoch,
            train_loss,
            panel.auc,
            panel.sensitivity,
            panel.specificity,
            " *" if improved else "",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": panel.auc,
                "val_sens": panel.sensitivity,
                "val_spec": panel.specificity,
                "val_ppv": panel.ppv,
            }
        )
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(panel.auc)
            else:
                scheduler.step()
        if stopper is not None and stopper(panel.auc):
            LOGGER.info("Early stopping at epoch %d", epoch)
            break
    return history


def main(
    config_path: Path, *, seed: int | None = None, run_name: str | None = None
) -> None:
    setup_logging()
    cfg = load_config(config_path)
    cfg = dataclasses.replace(
        cfg,
        seed=cfg.seed if seed is None else seed,
        run_name=cfg.run_name if run_name is None else run_name,
    )
    validate_split_paths(cfg.data.train_csv, cfg.data.val_csv, cfg.data.test_csv)
    set_global_seed(cfg.seed)

    device = get_device()
    LOGGER.info("Using device %s", device)

    train_loader, val_loader = _build_loaders(cfg)
    model_kwargs: dict[str, object] = {
        "dropout_conv": cfg.model.dropout_conv,
        "dropout_head": cfg.model.dropout_head,
        "head_hidden": cfg.model.head_hidden,
    }
    model = build_model(cfg.model.name, pretrained=cfg.model.pretrained, **model_kwargs)
    model = model.to(device)
    criterion = make_criterion(
        cfg.data.train_csv, device, label_smoothing=cfg.train.label_smoothing
    )
    ckpt = BestAUCCheckpoint(cfg.output_dir / f"{cfg.run_name}.pt")
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience or 10)

    is_transfer = cfg.model.name.lower() in ARCHS
    history: list[dict] = []
    if is_transfer and cfg.model.pretrained:
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
            grad_clip=cfg.train.grad_clip,
            mixup_alpha=cfg.train.mixup_alpha,
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
            scheduler=scheduler,
            start_epoch=cfg.train.stage1_epochs,
            grad_clip=cfg.train.grad_clip,
            mixup_alpha=cfg.train.mixup_alpha,
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
            scheduler=scheduler,
            grad_clip=cfg.train.grad_clip,
            mixup_alpha=cfg.train.mixup_alpha,
        )

    _save_history(cfg.output_dir / f"{cfg.run_name}.history.json", history)

    # Reload best-AUC checkpoint before calculating operating threshold.
    best_path = cfg.output_dir / f"{cfg.run_name}.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    val_y, val_p = _predict(model, val_loader, device)
    panel = evaluate(val_y, val_p)
    final_val_auc = history[-1]["val_auc"] if history else None
    _persist_val_threshold(
        cfg, youden_threshold(val_y, val_p), panel.auc, final_val_auc
    )
    LOGGER.info(
        "Training done. Best val AUC = %.4f (final-epoch val AUC = %s, gap = %s)",
        panel.auc,
        f"{final_val_auc:.4f}" if final_val_auc is not None else "n/a",
        f"{panel.auc - final_val_auc:+.4f}" if final_val_auc is not None else "n/a",
    )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="TOML experiment config.",
)
@click.option("--seed", type=int, help="Override the config seed.")
@click.option("--run-name", help="Save this run under a different name.")
def cli(config_path: Path, seed: int | None, run_name: str | None) -> None:
    main(config_path, seed=seed, run_name=run_name)


if __name__ == "__main__":
    cli()
