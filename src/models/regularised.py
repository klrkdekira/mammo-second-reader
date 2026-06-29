"""Deeper CNN variant: five conv blocks, same head as BaselineCNN.

Weight-decay regularisation lives in the optimiser (TrainConfig.weight_decay), not here.
"""

import torch
import torch.nn as nn

from src.models.baseline import BaselineCNN


class DeeperCNN(nn.Module):
    """Five-block conv stack to compare capacity at matched head width."""

    def __init__(
        self,
        dropout_conv: float = 0.3,
        dropout_head: float = 0.5,
        head_hidden: int = 256,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            BaselineCNN._block(in_channels, 32, dropout_conv),
            BaselineCNN._block(32, 64, dropout_conv),
            BaselineCNN._block(64, 128, dropout_conv),
            BaselineCNN._block(128, 256, dropout_conv),
            BaselineCNN._block(256, 512, dropout_conv),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
