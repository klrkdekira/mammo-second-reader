"""Configuration system, seeding, and logging.

Every experiment is driven by a TOML config so the training entry-point is
one function and re-runs are pure-data changes.
"""

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
    """Schema for an ensemble config.

    Ensemble configs differ from single-model configs: they carry a top-level
    `members` list of checkpoint names and have no [model]/[train] sections, so
    `load_config`/`Config` cannot represent them. Use `load_ensemble_config`.
    """

    seed: int
    run_name: str
    members: list[str]
    test_csv: Path
    image_root: Path
    val_csv: Path | None = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    output_dir: Path = field(default_factory=lambda: Path("models"))


_TOP_LEVEL_KEYS = {"seed", "run_name", "output_dir", "data", "model", "train"}
_ENSEMBLE_TOP_KEYS = {"seed", "run_name", "output_dir", "members", "data"}
_ENSEMBLE_DATA_KEYS = {
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
    """Raise `ValueError` if `provided` contains keys outside `allowed`.

    Guards against silently-dropped typos (e.g. `augmnet` in [data]), which
    otherwise leave the default in force with no warning.
    """
    unknown = set(provided) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )


def load_config(path: Path) -> Config:
    """Load a single-model TOML config file.

    For ensemble configs (top-level `members`, no [model]/[train]) use
    `load_ensemble_config` instead. This loader requires those sections.
    """
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
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        output_dir=Path(raw.get("output_dir", "models")),
    )


def load_ensemble_config(path: Path) -> EnsembleConfig:
    """Load an ensemble TOML config into a validated `EnsembleConfig`."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    _reject_unknown_keys("top level", raw, _ENSEMBLE_TOP_KEYS)
    data = raw.get("data", {})
    _reject_unknown_keys("[data]", data, _ENSEMBLE_DATA_KEYS)

    val_csv = data.get("val_csv")
    return EnsembleConfig(
        seed=int(raw["seed"]),
        run_name=str(raw["run_name"]),
        members=list(raw["members"]),
        test_csv=Path(data["test_csv"]),
        image_root=Path(data["image_root"]),
        val_csv=Path(val_csv) if val_csv else None,
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
