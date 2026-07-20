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
from src.models.transfer import ARCHS

MODEL_DIR = Path("models")

# Reject oversized uploads before decoding. Full-field DICOM mammograms are
# typically 10-30 MB; 100 MB leaves generous headroom while bounding the work
# a single malformed/hostile upload can trigger.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# checkpoint registry mapping run_name -> architecture.
MODEL_REGISTRY: dict[str, str] = {
    "baseline": "baseline",
    "regularised_base": "deeper",
    "regularised_heavy_aug": "deeper",
    "regularised_label_smooth": "deeper",
    "regularised_mixup": "deeper",
    "regularised_combined": "deeper",
    "vgg16_scratch": "vgg16",
    **{f"{arch}_imagenet": arch for arch in ARCHS},
}


def available_models() -> list[str]:
    """Registered checkpoints present on disk, ordered for display."""
    return [p.stem for p in sorted(MODEL_DIR.glob("*.pt")) if p.stem in MODEL_REGISTRY]


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_REGISTRY[model_name], pretrained=False)
    weights = MODEL_DIR / f"{model_name}.pt"
    # Never serve predictions from an uninitialised network: a randomly
    # initialised model still returns confident-looking probabilities, which
    # is unacceptable in a clinical-facing demo.
    if not weights.exists():
        raise FileNotFoundError(
            f"No checkpoint for model {model_name!r} at {weights}. "
            "Train the model or pick one of the available checkpoints."
        )
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
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

    Returns the segmented, CLAHE-equalised image in the unit range [0, 1].
    ImageNet normalisation is deliberately NOT applied here: it happens once,
    in `run_single_inference`, so this array can double as the (unnormalised)
    Grad-CAM overlay base. Do not normalise twice.

    Raises `ValueError` for oversized or undecodable uploads so the caller can
    surface a user-facing message instead of a raw stack trace.
    """
    from src.data.preprocessing import preprocess_array

    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Upload is {len(contents) // (1024 * 1024)} MB, over the "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    if filename.lower().endswith(".dcm"):
        import pydicom

        from src.data.preprocessing import dicom_to_array

        try:
            arr = dicom_to_array(pydicom.dcmread(io.BytesIO(contents)))
        except Exception as exc:
            raise ValueError(f"Could not decode the file as a DICOM image: {exc}") from exc
    else:
        from PIL import Image, UnidentifiedImageError

        try:
            img = Image.open(io.BytesIO(contents)).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(
                f"Could not decode the file as a PNG/JPEG image: {exc}"
            ) from exc
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
