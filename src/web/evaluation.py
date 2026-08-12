"""Run batch evaluation from the web app."""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.metrics import evaluate
from src.web.archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_FILES,
    MAX_EXTRACTED_BYTES,
    deidentify_dicom_in_place,
    extract_flat_archive,
)
from src.web.inference import _load_model, model_threshold

LOGGER = logging.getLogger(__name__)


def _extract_batch(zip_path: str, workdir: Path) -> Path:
    """Unpack a batch, remove DICOM tags, and find its CSV file."""
    extracted = extract_flat_archive(
        zip_path,
        workdir,
        max_archive_bytes=MAX_ARCHIVE_BYTES,
        max_extracted_bytes=MAX_EXTRACTED_BYTES,
        max_files=MAX_ARCHIVE_FILES,
    )
    manifests = [path for path in extracted if path.suffix.lower() == ".csv"]
    if not manifests:
        raise ValueError("No CSV manifest was found in the archive.")
    if len(manifests) != 1:
        raise ValueError("Batch evaluation requires exactly one CSV manifest.")
    for path in extracted:
        if path.suffix.lower() == ".dcm":
            deidentify_dicom_in_place(path)
    return manifests[0]


def run_batch_evaluation(
    zip_path: str,
    model_name: str,
    *,
    image_size: int = 224,
    batch_size: int = 32,
) -> dict[str, object]:
    """Evaluate a model using an uploaded ZIP file."""
    if not model_name:
        raise ValueError("A model must be selected.")
    with tempfile.TemporaryDirectory(prefix="mammo-evaluation-") as tmp:
        workdir = Path(tmp)
        manifest_csv = _extract_batch(zip_path, workdir)
        dataset = MammogramDataset(
            manifest_csv,
            workdir,
            transform=val_augment(image_size),
        )
        if len(dataset) == 0:
            raise ValueError("The batch manifest contains no images.")
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
        )
        model = _load_model(model_name)
        device = next(model.parameters()).device
        probabilities: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                logits = model(images.to(device))
                probabilities.append(torch.sigmoid(logits).cpu().numpy().ravel())
        y_prob = np.concatenate(probabilities)
        y_true = np.asarray(dataset.df["label"], dtype=np.int64)
        threshold = float(model_threshold(model_name))
        panel = evaluate(y_true, y_prob, threshold=threshold)
        result = dataclasses.asdict(panel)
        result["confusion"] = panel.confusion.tolist()
        result["n"] = len(dataset)
        result["threshold"] = threshold
        return result
