"""Single-image inference for the webapp.

Loads checkpoints on demand and returns the probability, label, threshold,
and a colour Grad-CAM overlay (jet heatmap blended over the input).
"""

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.evaluation.gradcam import TARGET_LAYERS, compute_gradcam
from src.models import build_model

MODEL_DIR = Path("models")

# checkpoint registry mapping the trained model
MODEL_REGISTRY: dict[str, str] = {
    "baseline": "baseline",
    "regularised": "deeper",
    "vgg16_scratch": "vgg16",
    "vgg16_imagenet": "vgg16",
}


def available_models() -> list[str]:
    """Registered checkpoints present on disk, ordered for display."""
    return [p.stem for p in sorted(MODEL_DIR.glob("*.pt")) if p.stem in MODEL_REGISTRY]


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_REGISTRY[model_name], pretrained=False)
    weights = MODEL_DIR / f"{model_name}.pt"
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@lru_cache(maxsize=4)
def model_threshold(model_name: str) -> float:
    """Youden-J operating threshold for a model, defaulting to 0.5."""
    sidecar = MODEL_DIR / f"{model_name}.threshold.json"
    if sidecar.exists():
        return float(json.loads(sidecar.read_text())["youden_j"])
    return 0.5


def _preprocess_bytes(contents: bytes, filename: str) -> np.ndarray:
    """Decode DICOM or PNG/JPG bytes and run the shared preprocessing pipeline.

    Returns the segmented, CLAHE-equalised image in the unit range. ImageNet
    normalisation is applied separately so the array can double as the
    Grad-CAM overlay base.
    """
    from src.data.preprocessing import preprocess_array

    if filename.lower().endswith(".dcm"):
        import pydicom

        arr = pydicom.dcmread(io.BytesIO(contents)).pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        from PIL import Image

        img = Image.open(io.BytesIO(contents)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return preprocess_array(arr)


def _overlay_to_b64(image: np.ndarray, heatmap: np.ndarray) -> str:
    """Blend a jet heatmap over the grayscale image and PNG-encode to base64."""
    from PIL import Image
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rgb = np.stack([np.clip(image, 0.0, 1.0)] * 3, axis=-1).astype(np.float32)
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True, image_weight=0.5)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_single_inference(
    contents: bytes, filename: str, model_name: str, threshold: float | None
) -> dict:
    """Classify one image and return probability, label, threshold, and overlay."""
    from src.data.preprocessing import normalise

    image = _preprocess_bytes(contents, filename)
    model = _load_model(model_name)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalise(image)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor)).item())
    thr = threshold if threshold is not None else model_threshold(model_name)

    overlay_b64 = ""
    target = TARGET_LAYERS.get(MODEL_REGISTRY[model_name])
    if target is not None:
        try:
            overlay_b64 = _overlay_to_b64(image, compute_gradcam(model, tensor, target))
        except Exception:
            pass

    return {
        "probability": prob,
        "label": int(prob >= thr),
        "threshold": thr,
        "gradcam_overlay": overlay_b64,
    }
