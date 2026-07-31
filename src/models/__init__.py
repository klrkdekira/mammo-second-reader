from typing import Any

import torch.nn as nn

from src.models.baseline import BaselineCNN
from src.models.regularised import DeeperCNN

__all__ = ["BaselineCNN", "DeeperCNN"]


def build_model(name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    """Single dispatch entry-point used by the training loop."""
    from src.models import transfer

    name = name.lower()
    if name == "baseline":
        return BaselineCNN(**kwargs)
    if name == "deeper":
        return DeeperCNN(**kwargs)
    return transfer.build_model(name, pretrained=pretrained, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return the total and trainable number of parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
