"""ImageNet-pretrained backbones with a swapped classifier head."""

from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn


class ArchSpec(NamedTuple):
    weights_attr: str  # e.g. "VGG16_Weights"
    weights_ver: str  # e.g. "IMAGENET1K_V1"
    head_attr: str  # attribute on the backbone holding the classifier
    top_block: tuple  # (attr,) or (attr, from_idx) for fine-tune unfreezing


ARCHS: dict[str, ArchSpec] = {
    "vgg16": ArchSpec("VGG16_Weights", "IMAGENET1K_V1", "classifier", ("features", 24)),
    "vgg19": ArchSpec("VGG19_Weights", "IMAGENET1K_V1", "classifier", ("features", 28)),
    "resnet50": ArchSpec("ResNet50_Weights", "IMAGENET1K_V2", "fc", ("layer4",)),
    "efficientnet_b4": ArchSpec(
        "EfficientNet_B4_Weights", "IMAGENET1K_V1", "classifier", ("features", 7)
    ),
}


class ThreeChannelWrapper(nn.Module):
    """Repeats a grayscale channel into three to match ImageNet input format."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def build_model(
    name: str,
    pretrained: bool = True,
    head_hidden: int = 256,
    dropout_head: float = 0.5,
    dropout_conv: float = 0.3,
    **_: Any,
) -> nn.Module:
    import torchvision.models as M

    name = name.lower()
    spec = ARCHS[name]
    weights_cls = getattr(M, spec.weights_attr)
    weights = getattr(weights_cls, spec.weights_ver) if pretrained else None
    backbone = getattr(M, name)(weights=weights)

    old_head = getattr(backbone, spec.head_attr)
    last_layer = old_head[-1] if isinstance(old_head, nn.Sequential) else old_head
    in_features = cast(nn.Linear, last_layer).in_features
    # dropout_conv is repurposed as a second head dropout.
    # no conv layer is exposed on a pretrained backbone.
    # Defaults (0.5, 0.3) preserve the original behaviour.
    new_head = nn.Sequential(
        nn.Dropout(dropout_head),
        nn.Linear(in_features, head_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_conv),
        nn.Linear(head_hidden, 1),
    )
    if isinstance(old_head, nn.Sequential):
        old_head[-1] = new_head
    else:
        setattr(backbone, spec.head_attr, new_head)
    return ThreeChannelWrapper(backbone)


def freeze_backbone(model: nn.Module) -> None:
    """Freezes all parameters in the wrapped backbone."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model: nn.Module) -> None:
    """Re-enables gradients for the classifier head only."""
    wrapped = getattr(model, "backbone", model)
    head = getattr(wrapped, "classifier", None) or getattr(wrapped, "fc", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_top_blocks(model: nn.Module, name: str) -> None:
    """Unfreezes the top convolutional blocks for the given architecture."""
    wrapped = getattr(model, "backbone", model)
    top = ARCHS[name].top_block
    block = getattr(wrapped, top[0])
    region = block[top[1] :] if len(top) > 1 else block
    for p in region.parameters():
        p.requires_grad = True
