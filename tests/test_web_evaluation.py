"""Tests for the web batch-evaluation flow: de-identification-on-extract,
manifest handling, and the run_batch_evaluation wiring."""

import zipfile

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset

from src.web import evaluation


def _make_dicom_bytes(tmp_path, name: str, patient_name: str = "Test^Patient"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.PatientID = "12345"

    path = tmp_path / name
    ds.save_as(path, enforce_file_format=True)
    return path


def test_extract_batch_deidentifies_dicoms_in_place(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _make_dicom_bytes(src_dir, "a.dcm")
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "labels.csv", index=False
    )

    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src_dir.iterdir():
            zf.write(f, arcname=f.name)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    manifest_csv = evaluation._extract_batch(str(zip_path), workdir)

    assert manifest_csv.name == "labels.csv"
    stripped = pydicom.dcmread(str(workdir / "a.dcm"))
    assert "PatientName" not in stripped
    assert "PatientID" not in stripped


def test_extract_batch_flattens_nested_zip(tmp_path):
    src_dir = tmp_path / "src" / "nested"
    src_dir.mkdir(parents=True)
    _make_dicom_bytes(src_dir, "a.dcm")
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "labels.csv", index=False
    )

    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src_dir.iterdir():
            zf.write(f, arcname=f"nested/{f.name}")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    evaluation._extract_batch(str(zip_path), workdir)

    assert (workdir / "a.dcm").exists()


def test_extract_batch_rejects_a_zip_bomb_by_declared_size(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _make_dicom_bytes(src_dir, "a.dcm")
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "labels.csv", index=False
    )
    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src_dir.iterdir():
            zf.write(f, arcname=f.name)

    class _HugeZipInfo:
        file_size = evaluation.MAX_EXTRACTED_BYTES + 1

    # A real zip bomb declares a huge uncompressed size in its own metadata;
    # mocking infolist() exercises that check without needing gigabytes of data.
    monkeypatch.setattr(zipfile.ZipFile, "infolist", lambda self: [_HugeZipInfo()])

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    with pytest.raises(ValueError, match="decompress"):
        evaluation._extract_batch(str(zip_path), workdir)


def test_extract_batch_raises_without_csv(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _make_dicom_bytes(src_dir, "a.dcm")

    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src_dir / "a.dcm", arcname="a.dcm")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    with pytest.raises(ValueError, match="No CSV manifest"):
        evaluation._extract_batch(str(zip_path), workdir)


class _TinyModel(torch.nn.Module):
    """Alternates confident correct logits by batch position; only the
    run_batch_evaluation wiring is under test, not model correctness."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        logits = [6.0 if i % 2 else -6.0 for i in range(n)]
        return torch.tensor(logits, dtype=torch.float32).unsqueeze(1)


def test_run_batch_evaluation_end_to_end(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    image_ids = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    for image_id in image_ids:
        (batch_dir / f"{image_id}.dcm").write_bytes(b"placeholder")
    pd.DataFrame({"image_id": image_ids, "label": labels}).to_csv(
        batch_dir / "labels.csv", index=False
    )

    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in batch_dir.iterdir():
            zf.write(f, arcname=f.name)

    # The placeholder .dcm files aren't valid DICOM, so de-identification in
    # _extract_batch logs a warning and skips them (graceful, not fatal).
    # preprocess() is patched so MammogramDataset never needs real pixel data:
    # only the evaluation wiring is under test here.
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess",
        lambda path, image_size=224, use_clahe=True: np.zeros(
            (image_size, image_size), dtype=np.float32
        ),
    )
    monkeypatch.setattr(evaluation, "_load_model", lambda name: _TinyModel())
    monkeypatch.setattr(evaluation, "model_threshold", lambda name: 0.5)

    result = evaluation.run_batch_evaluation(str(zip_path), "vgg16_imagenet")

    assert result["n"] == 4
    assert result["auc"] == pytest.approx(1.0)
    assert result["confusion"] == [[2, 0], [0, 2]]
