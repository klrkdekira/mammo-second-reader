"""Tests for the five-class patch dataset loader."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.patch_dataset import PatchDataset
from src.data.patch_manifest import CLASS_TO_ID


def _write_manifest(tmp_path, **overrides):
    row = {
        "patch_id": "patch-a",
        "patch_path": "patches/train/malignant_mass/patch-a.npy",
        "patient_id": "patient-a",
        "split": "train",
        "patch_class": "malignant_mass",
        "class_id": CLASS_TO_ID["malignant_mass"],
    }
    row.update(overrides)
    path = tmp_path / "train.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path, row


def test_patch_dataset_loads_single_channel_tensor_and_class_id(tmp_path):
    manifest, row = _write_manifest(tmp_path)
    patch = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = tmp_path / row["patch_path"]
    path.parent.mkdir(parents=True)
    np.save(path, patch)

    dataset = PatchDataset(manifest, tmp_path)
    image, label = dataset[0]

    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert label.item() == CLASS_TO_ID["malignant_mass"]
    np.testing.assert_array_equal(image.numpy()[0], patch)


def test_patch_dataset_applies_albumentations_style_transform(tmp_path):
    manifest, row = _write_manifest(tmp_path)
    path = tmp_path / row["patch_path"]
    path.parent.mkdir(parents=True)
    np.save(path, np.ones((8, 8), dtype=np.float32))

    dataset = PatchDataset(
        manifest, tmp_path, transform=lambda image: {"image": image * 2}
    )
    image, _ = dataset[0]
    assert torch.all(image == 2)


def test_patch_dataset_rejects_mismatched_class_id(tmp_path):
    manifest, _ = _write_manifest(tmp_path, class_id=CLASS_TO_ID["background"])
    with pytest.raises(ValueError, match="names and IDs"):
        PatchDataset(manifest, tmp_path)
