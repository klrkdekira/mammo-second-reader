"""Grad-CAM heatmap generation via the pytorch-grad-cam library.

The algorithm is the gradient-weighted feature-map aggregation defined
in Selvaraju et al. 2020, with the layer-mapping for each architecture
listed below.
"""

from typing import cast

import numpy as np
import torch

TARGET_LAYERS = {
    "baseline": "features.2.0",
    "deeper": "features.4.0",
    "vgg16": "backbone.features.28",
    "vgg19": "backbone.features.34",
    "resnet18": "backbone.layer4.1.conv2",
    "resnet50": "backbone.layer4.2.conv3",
    "densenet121": "backbone.features.norm5",
    "efficientnet_b0": "backbone.features.8",
    "efficientnet_b4": "backbone.features.8",
    "mobilenet_v3_large": "backbone.features.16",
    "convnext_tiny": "backbone.features.7",
}


def _resolve_layer(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    curr: object = model
    for part in dotted.split("."):
        if part.isdigit():
            curr = curr.__getitem__(int(part))
        else:
            curr = getattr(curr, part)
    return cast(torch.nn.Module, curr)


def compute_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer_name: str,
    target_class: int = 0,
) -> np.ndarray:
    """Return a (H, W) heatmap normalised to [0, 1] for the given image.

    image must be (1, C, H, W). target_class is the logit index. For the
    single-output binary head it is always 0.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = _resolve_layer(model, target_layer_name)
    targets = [ClassifierOutputTarget(target_class)]
    with GradCAM(model=model, target_layers=[layer]) as cam:
        greyscale_cam = cam(input_tensor=image, targets=targets)
    return greyscale_cam[0].astype(np.float32)
