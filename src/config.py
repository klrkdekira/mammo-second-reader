"""Configuration, seeding, and logging utilities."""

import logging
import os
import random
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    image_root: Path
    image_size: int = 224
    cache_dir: Path | None = None
    augment: str = "default"
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool = True
    dropout_conv: float = 0.3
    dropout_head: float = 0.5
    head_hidden: int = 256
    init_from_patch_checkpoint: Path | None = None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimiser: str = "adam"
    scheduler: str | None = None
    early_stop_patience: int | None = None
    stage1_epochs: int = 5
    stage2_lr: float = 1e-5
    grad_clip: float | None = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    sampler: str = "shuffle"  # "shuffle" or "balanced" (WeightedRandomSampler)


@dataclass(frozen=True)
class Config:
    seed: int
    run_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output_dir: Path = field(default_factory=lambda: Path("models"))


@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble configuration schema."""

    seed: int
    run_name: str
    members: list[str]
    train_csv: Path
    val_csv: Path
    test_csv: Path
    image_root: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    output_dir: Path = field(default_factory=lambda: Path("models"))


@dataclass(frozen=True)
class PatchDataConfig:
    """Patch inputs with no test fold."""

    train_csv: Path
    val_csv: Path
    patch_root: Path
    patch_size: int = 224
    augment: str = "light"
    num_workers: int = 2
    exclusion_test_csv: Path | None = None


@dataclass(frozen=True)
class PatchTrainConfig(TrainConfig):
    """Training configuration for the five-class patch model."""

    class_weighted_loss: bool = True
    selection_metric: str = "macro_f1"  # or "balanced_accuracy"


@dataclass(frozen=True)
class PatchConfig:
    """Patch-classifier configuration schema."""

    seed: int
    run_name: str
    data: PatchDataConfig
    model: ModelConfig
    train: PatchTrainConfig
    output_dir: Path = field(default_factory=lambda: Path("models/patch_learning"))


_TOP_LEVEL_KEYS = {"seed", "run_name", "output_dir", "data", "model", "train"}
_ENSEMBLE_TOP_KEYS = {"seed", "run_name", "output_dir", "members", "data"}
_ENSEMBLE_DATA_KEYS = {
    "train_csv",
    "val_csv",
    "test_csv",
    "image_root",
    "image_size",
    "batch_size",
    "num_workers",
}


def _reject_unknown_keys(
    section: str, provided: Iterable[str], allowed: Iterable[str]
) -> None:
    """Reject keys outside the section schema."""
    unknown = set(provided) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )


def _model_config(raw: dict[str, object]) -> ModelConfig:
    """Build a ModelConfig, coercing the optional patch-checkpoint path."""
    values = dict(raw)
    checkpoint = values.pop("init_from_patch_checkpoint", None)
    return ModelConfig(
        init_from_patch_checkpoint=Path(str(checkpoint)) if checkpoint else None,
        **values,  # type: ignore[arg-type]
    )


def load_config(path: Path) -> Config:
    """Load a single-model TOML config."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    _reject_unknown_keys("top level", raw, _TOP_LEVEL_KEYS)
    for section, cls in (
        ("data", DataConfig),
        ("model", ModelConfig),
        ("train", TrainConfig),
    ):
        if section in raw:
            _reject_unknown_keys(
                f"[{section}]", raw[section], {f.name for f in fields(cls)}
            )

    return Config(
        seed=int(raw["seed"]),
        run_name=str(raw["run_name"]),
        data=DataConfig(
            train_csv=Path(raw["data"]["train_csv"]),
            val_csv=Path(raw["data"]["val_csv"]),
            test_csv=Path(raw["data"]["test_csv"]),
            image_root=Path(raw["data"]["image_root"]),
            image_size=int(raw["data"].get("image_size", 224)),
            cache_dir=Path(raw["data"].get("cache_dir"))
            if raw["data"].get("cache_dir")
            else None,
            augment=str(raw["data"].get("augment", "default")),
            num_workers=int(
                raw["data"].get("num_workers", os.environ.get("MAMMO_NUM_WORKERS", "2"))
            ),
        ),
        model=_model_config(raw["model"]),
        train=TrainConfig(**raw["train"]),
        output_dir=Path(raw.get("output_dir", "models")),
    )


def load_patch_config(path: Path) -> PatchConfig:
    """Load a Stage 0 patch-classifier TOML config.

    Rejects a `test_csv` key outright rather than ignoring it, so an attempt to
    point the patch task at test patients fails at config load.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    _reject_unknown_keys("top level", raw, _TOP_LEVEL_KEYS)
    for section, cls in (
        ("data", PatchDataConfig),
        ("model", ModelConfig),
        ("train", PatchTrainConfig),
    ):
        if section in raw:
            _reject_unknown_keys(
                f"[{section}]", raw[section], {f.name for f in fields(cls)}
            )

    train = PatchTrainConfig(**raw["train"])
    if train.mixup_alpha > 0.0:
        raise ValueError(
            "mixup_alpha is not supported for the five-class patch head: the "
            "whole-image implementation mixes binary soft targets for BCE."
        )
    if train.selection_metric not in ("macro_f1", "balanced_accuracy"):
        raise ValueError(
            f"Unknown train.selection_metric {train.selection_metric!r}; "
            "expected 'macro_f1' or 'balanced_accuracy'."
        )

    return PatchConfig(
        seed=int(raw["seed"]),
        run_name=str(raw["run_name"]),
        data=PatchDataConfig(
            train_csv=Path(raw["data"]["train_csv"]),
            val_csv=Path(raw["data"]["val_csv"]),
            patch_root=Path(raw["data"]["patch_root"]),
            patch_size=int(raw["data"].get("patch_size", 224)),
            augment=str(raw["data"].get("augment", "light")),
            num_workers=int(
                raw["data"].get("num_workers", os.environ.get("MAMMO_NUM_WORKERS", "2"))
            ),
            exclusion_test_csv=Path(raw["data"]["exclusion_test_csv"])
            if raw["data"].get("exclusion_test_csv")
            else None,
        ),
        model=_model_config(raw["model"]),
        train=train,
        output_dir=Path(raw.get("output_dir", "models/patch_learning")),
    )


def load_ensemble_config(path: Path) -> EnsembleConfig:
    """Load an ensemble TOML config into a validated `EnsembleConfig`."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    _reject_unknown_keys("top level", raw, _ENSEMBLE_TOP_KEYS)
    data = raw.get("data", {})
    _reject_unknown_keys("[data]", data, _ENSEMBLE_DATA_KEYS)

    return EnsembleConfig(
        seed=int(raw["seed"]),
        run_name=str(raw["run_name"]),
        members=list(raw["members"]),
        train_csv=Path(data["train_csv"]),
        val_csv=Path(data["val_csv"]),
        test_csv=Path(data["test_csv"]),
        image_root=Path(data["image_root"]),
        image_size=int(data.get("image_size", 224)),
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(
            data.get("num_workers", os.environ.get("MAMMO_NUM_WORKERS", "2"))
        ),
        output_dir=Path(raw.get("output_dir", "models")),
    )


def set_global_seed(seed: int) -> None:
    """Make a training run reproducible.

    Sets Python, NumPy, and PyTorch seeds, plus the cuDNN deterministic flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> "torch.device":
    """Return available torch device, preferring GPU if available."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once"""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
