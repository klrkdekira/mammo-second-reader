"""Single-image inference for the webapp.

Loads checkpoints on demand and returns the probability, label, threshold,
and a colour Grad-CAM overlay (jet heatmap blended over the input).
"""

import base64
import io
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.evaluation.gradcam import TARGET_LAYERS, compute_gradcam
from src.models import build_model
from src.models.transfer import ARCHS

LOGGER = logging.getLogger(__name__)

MODEL_DIR = Path("models")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Training writes whole-image runs to models/ and patch-transfer runs to
# models/patch_learning/, so both are scanned.
SEARCH_SUBDIRS: tuple[str, ...] = ("", "patch_learning")

_SEED_SUFFIX = re.compile(r"_seed\d+$")
# Five-class patch classifiers share the models tree but not the binary head.
_PATCH_CLASSIFIER = re.compile(r"_patch(_aug)?$")


def resolve_arch(model_name: str) -> str | None:
    """Architecture a run name was trained with, or None if unknown.

    Run names follow the config convention: `baseline`, `regularised_*`
    (DeeperCNN), and `<arch>_*` for torchvision backbones, with optional
    `_448`, `_120` and `_seed<N>` suffixes.
    """
    stem = _SEED_SUFFIX.sub("", model_name.lower())
    if _PATCH_CLASSIFIER.search(stem):
        return None
    if stem == "baseline" or stem.startswith("baseline_"):
        return "baseline"
    if stem.startswith("regularised"):
        return "deeper"
    for arch in sorted(ARCHS, key=len, reverse=True):
        if stem == arch or stem.startswith(f"{arch}_"):
            return arch
    return None


def model_image_size(model_name: str) -> int:
    """Input resolution a run was trained at."""
    return 448 if "_448" in model_name else 224


def _checkpoints() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for sub in SEARCH_SUBDIRS:
        directory = MODEL_DIR / sub if sub else MODEL_DIR
        for path in sorted(directory.glob("*.pt")):
            if path.stem in found or resolve_arch(path.stem) is None:
                continue
            if path.with_name(f"{path.stem}.patch-metrics.json").exists():
                continue
            found[path.stem] = path
    return found


def checkpoint_path(model_name: str) -> Path:
    """On-disk checkpoint for a model name, defaulting to MODEL_DIR."""
    return _checkpoints().get(model_name, MODEL_DIR / f"{model_name}.pt")


def available_models() -> list[str]:
    """Binary whole-image checkpoints present on disk, ordered for display."""
    return sorted(_checkpoints())


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = resolve_arch(model_name)
    weights = checkpoint_path(model_name)
    if arch is None or not weights.exists():
        raise FileNotFoundError(
            f"No checkpoint for model {model_name!r} at {weights}. "
            "Train the model or pick one of the available checkpoints."
        )
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


@lru_cache(maxsize=4)
def model_threshold(model_name: str) -> float:
    """Youden-J operating threshold for a model, defaulting to 0.5."""
    sidecar = checkpoint_path(model_name).with_suffix(".threshold.json")
    if sidecar.exists():
        return float(json.loads(sidecar.read_text())["youden_j"])
    return 0.5


def _preprocess_bytes(
    contents: bytes, filename: str, image_size: int = 224
) -> np.ndarray:
    """Decode and preprocess an uploaded DICOM, PNG, or JPEG image."""
    from src.data.preprocessing import preprocess_array

    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Upload is {len(contents) // (1024 * 1024)} MB, over the "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    if filename.lower().endswith(".dcm"):
        import pydicom

        from src.data.deidentify import strip_identifying_tags
        from src.data.preprocessing import dicom_to_array

        try:
            dataset = pydicom.dcmread(io.BytesIO(contents))
            strip_identifying_tags(dataset)
            arr = dicom_to_array(dataset)
        except Exception as exc:
            raise ValueError(
                f"Could not decode the file as a DICOM image: {exc}"
            ) from exc
    else:
        from PIL import Image, UnidentifiedImageError

        try:
            img = Image.open(io.BytesIO(contents)).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(
                f"Could not decode the file as a PNG/JPEG image: {exc}"
            ) from exc
    return preprocess_array(arr, image_size)


def _overlay_to_b64(image: np.ndarray, heatmap: np.ndarray) -> str:
    """Blend a jet heatmap over the greyscale image and PNG-encode to base64."""
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

    image = _preprocess_bytes(contents, filename, model_image_size(model_name))
    model = _load_model(model_name)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalise(image)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor)).item())
    thr = threshold if threshold is not None else model_threshold(model_name)

    overlay_b64 = ""
    target = TARGET_LAYERS.get(resolve_arch(model_name) or "")
    if target is not None:
        try:
            overlay_b64 = _overlay_to_b64(image, compute_gradcam(model, tensor, target))
        except Exception:
            LOGGER.warning(
                "Grad-CAM failed for model %s; returning prediction without overlay",
                model_name,
                exc_info=True,
            )

    return {
        "probability": prob,
        "label": int(prob >= thr),
        "threshold": thr,
        "gradcam_overlay": overlay_b64,
    }
