"""Tests for web inference."""

import io

import numpy as np
import pydicom
import pytest

from src.data.preprocessing import normalise
from src.web import inference


def test_missing_checkpoint_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    inference._load_model.cache_clear()
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        inference._load_model("vgg16_imagenet")


def _png_bytes(arr_uint8: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr_uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_bytes_returns_unnormalised_unit_range():
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    out = inference._preprocess_bytes(_png_bytes(ramp), "scan.png")
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-5


def test_normalise_is_the_single_normalisation_step():
    unit = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert normalise(unit).min() < 0.0


def test_oversized_upload_rejected(monkeypatch):
    monkeypatch.setattr(inference, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="over the"):
        inference._preprocess_bytes(b"x" * 11, "scan.png")


def test_malformed_dicom_raises_value_error():
    with pytest.raises(ValueError, match="DICOM"):
        inference._preprocess_bytes(b"not a dicom", "scan.dcm")


def test_malformed_image_raises_value_error():
    with pytest.raises(ValueError, match="PNG/JPEG"):
        inference._preprocess_bytes(b"not an image", "scan.png")


def test_dicom_upload_is_deidentified_before_decoding(monkeypatch):
    ds = pydicom.Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"

    captured = {}

    def fake_dcmread(*_args, **_kwargs):
        return ds

    def fake_dicom_to_array(dataset):
        captured["patient_name_present"] = "PatientName" in dataset
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(pydicom, "dcmread", fake_dcmread)
    monkeypatch.setattr("src.data.preprocessing.dicom_to_array", fake_dicom_to_array)
    monkeypatch.setattr("src.data.preprocessing.preprocess_array", lambda arr: arr)

    inference._preprocess_bytes(b"irrelevant bytes", "scan.dcm")

    assert captured["patient_name_present"] is False


@pytest.mark.parametrize(
    ("name", "arch"),
    [
        ("baseline", "baseline"),
        ("regularised_mixup_120", "deeper"),
        ("vgg16_scratch_seed7", "vgg16"),
        ("vgg16_imagenet_448_seed2026", "vgg16"),
        ("vgg16_patch_imagenet_448", "vgg16"),
        ("efficientnet_b4_imagenet", "efficientnet_b4"),
        ("resnet50_imagenet", "resnet50"),
        ("vgg16_patch", None),
        ("vgg16_patch_aug", None),
        ("regularised", "deeper"),
        ("unknown_model", None),
    ],
)
def test_resolve_arch(name, arch):
    assert inference.resolve_arch(name) == arch


def test_available_models_lists_every_whole_image_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    patch_dir = tmp_path / "patch_learning"
    patch_dir.mkdir()
    for name in ("vgg16_imagenet_seed7", "regularised_base_120", "vgg19_imagenet"):
        (tmp_path / f"{name}.pt").touch()
    (patch_dir / "vgg16_patch_imagenet_448.pt").touch()
    (patch_dir / "vgg16_patch.pt").touch()
    (patch_dir / "vgg16_imagenet_448_quarantined.pt").touch()

    assert inference.available_models() == [
        "regularised_base_120",
        "vgg16_imagenet_seed7",
        "vgg16_imagenet_448_quarantined",
        "vgg16_patch_imagenet_448",
        "vgg19_imagenet",
    ]
    assert inference.checkpoint_path("vgg16_patch_imagenet_448").parent == patch_dir
    assert inference.model_image_size("vgg16_patch_imagenet_448") == 448
