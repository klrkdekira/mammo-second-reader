"""ImageNet-pretrained backbones with a swapped classifier head.

`load_patch_backbone` is the patch-transfer bridge: it moves a patch
classifier's convolutional weights into a whole-image model of the same
architecture, leaving the binary head alone.
"""

from pathlib import Path
from typing import NamedTuple, cast

import torch
from torch import nn


class ArchSpec(NamedTuple):
    weights_attr: str  # e.g. "VGG16_Weights"
    weights_ver: str  # e.g. "IMAGENET1K_V1"
    head_attr: str  # attribute on the backbone holding the classifier
    top_block: (
        tuple[str, int] | tuple[str]
    )  # (attr,) or (attr, from_idx) for fine-tune unfreezing


ARCHS: dict[str, ArchSpec] = {
    "vgg16": ArchSpec("VGG16_Weights", "IMAGENET1K_V1", "classifier", ("features", 24)),
    "vgg19": ArchSpec("VGG19_Weights", "IMAGENET1K_V1", "classifier", ("features", 28)),
    "resnet50": ArchSpec("ResNet50_Weights", "IMAGENET1K_V2", "fc", ("layer4",)),
    "efficientnet_b4": ArchSpec(
        "EfficientNet_B4_Weights", "IMAGENET1K_V1", "classifier", ("features", 7)
    ),
}


class ThreeChannelWrapper(nn.Module):
    """Repeats a greyscale channel into three to match ImageNet input format."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return cast(torch.Tensor, self.backbone(x))


def build_model(
    name: str,
    pretrained: bool = True,
    head_hidden: int = 256,
    dropout_head: float = 0.5,
    dropout_conv: float = 0.3,
    num_classes: int = 1,
    **_: object,
) -> nn.Module:
    import torchvision.models as M

    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
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
    # num_classes defaults to 1 (one logit for BCEWithLogitsLoss); the Stage 0
    # patch task passes 5 for CrossEntropyLoss. Every locked whole-image run
    # uses the default, so their head geometry is unchanged.
    new_head = nn.Sequential(
        nn.Dropout(dropout_head),
        nn.Linear(in_features, head_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_conv),
        nn.Linear(head_hidden, num_classes),
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


def load_patch_backbone(
    model: nn.Module,
    checkpoint_path: str | Path,
    name: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, int]:
    """Copy a patch classifier's feature weights into a whole-image model.

    The transfer moves the convolutional feature extractor only. Everything
    under the architecture's classifier attribute is left as built, so VGG's
    two 4096-unit fully-connected layers stay ImageNet-initialised and the
    binary head is untouched: the whole-image model keeps its own head and
    gains only mammography-specific local features.

    The mapping is by exact state-dict key, and it is deliberately strict. A
    missing or reshaped feature tensor raises rather than transferring part of
    the network, because a silently partial transfer would look like a trained
    model and invalidate the controlled comparison.

    Returns a summary suitable for logging into a run's provenance record.
    """
    name = name.lower()
    if name not in ARCHS:
        raise ValueError(
            f"Unknown architecture {name!r}; expected one of {sorted(ARCHS)}"
        )
    head_attr = ARCHS[name].head_attr

    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{checkpoint_path} does not contain a state dict")

    # Accept checkpoints saved from a ThreeChannelWrapper or from a bare
    # backbone by normalising the optional wrapper prefix on both sides.
    def _strip(key: str) -> str:
        return key.removeprefix("backbone.")

    source = {_strip(key): value for key, value in state.items()}
    target = model.state_dict()

    transferred: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    skipped = 0
    for key, current in target.items():
        if _strip(key).startswith(f"{head_attr}."):
            skipped += 1
            continue
        candidate = source.get(_strip(key))
        if candidate is None:
            missing.append(key)
            continue
        if tuple(candidate.shape) != tuple(current.shape):
            mismatched.append(
                f"{key}: checkpoint {tuple(candidate.shape)} != model {tuple(current.shape)}"
            )
            continue
        transferred[key] = candidate

    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing from checkpoint: {missing[:5]}")
        if mismatched:
            details.append(f"shape mismatch: {mismatched[:5]}")
        raise ValueError(
            f"Cannot transfer patch weights from {checkpoint_path} into a "
            f"{name} whole-image model: " + "; ".join(details)
        )
    target.update(transferred)
    model.load_state_dict(target)
    return {
        "n_tensors_copied": len(transferred),
        "n_parameters_copied": int(
            sum(tensor.numel() for tensor in transferred.values())
        ),
        "n_tensors_left_as_built": skipped,
    }
