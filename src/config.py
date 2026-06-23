"""Configuration system, seeding, and logging.

Every experiment is driven by a TOML config so the training entry-point is
one function and re-runs are pure-data changes.
"""

import logging
import os
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    image_root: Path
    image_size: int = 224
    cache_dir: Path | None = None


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


@dataclass(frozen=True)
class Config:
    seed: int
    run_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output_dir: Path = field(default_factory=lambda: Path("models"))


def load_config(path: Path) -> Config:
    """Load a TOML config file"""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

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
        ),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
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


def get_device():
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
