"""Baseline CNN: three conv blocks, global average pool, small linear head.

GAP before the head keeps parameter count low.
A naive Flatten on a 28x28x128 feature map would cost ~25 M parameters in the head alone.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Three-block conv stack followed by GAP and a small classifier head."""

    def __init__(
        self,
        dropout_conv: float = 0.3,
        dropout_head: float = 0.5,
        head_hidden: int = 256,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32, dropout_conv),
            self._block(32, 64, dropout_conv),
            self._block(64, 128, dropout_conv),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
