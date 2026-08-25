from typing import Any

from torch import nn

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
