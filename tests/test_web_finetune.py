"""Tests for web fine-tuning."""

import zipfile

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset

from src.models import build_model
from src.web import finetune


def _make_dicom_bytes(tmp_path, name: str, patient_name: str = "Test^Patient"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)  # type: ignore[arg-type]
    ds.PatientName = patient_name
    ds.PatientID = "12345"

    path = tmp_path / name
    ds.save_as(path, enforce_file_format=True)
    return path


def test_materialise_workdir_requires_train_and_val_csv(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "train.csv", index=False
    )
    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src_dir / "train.csv", arcname="train.csv")

    workdir = tmp_path / "workdir"
    with pytest.raises(ValueError, match="train.csv and val.csv"):
        finetune.materialise_workdir(str(zip_path), workdir)


def test_materialise_workdir_rejects_a_zip_bomb_by_declared_size(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "train.csv", index=False
    )
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        src_dir / "val.csv", index=False
    )
    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src_dir.iterdir():
            zf.write(f, arcname=f.name)

    class _HugeZipInfo:
        file_size = finetune.MAX_EXTRACTED_BYTES + 1

    monkeypatch.setattr(zipfile.ZipFile, "infolist", lambda self: [_HugeZipInfo()])

    workdir = tmp_path / "workdir"
    with pytest.raises(ValueError, match="decompress"):
        finetune.materialise_workdir(str(zip_path), workdir)


def test_materialise_workdir_deidentifies_and_flattens(tmp_path):
    src_dir = tmp_path / "src" / "nested"
    src_dir.mkdir(parents=True)
    _make_dicom_bytes(src_dir, "a.dcm")
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        tmp_path / "src" / "train.csv", index=False
    )
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(
        tmp_path / "src" / "val.csv", index=False
    )

    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tmp_path / "src" / "train.csv", arcname="train.csv")
        zf.write(tmp_path / "src" / "val.csv", arcname="val.csv")
        zf.write(src_dir / "a.dcm", arcname="nested/a.dcm")

    workdir = tmp_path / "workdir"
    finetune.materialise_workdir(str(zip_path), workdir)

    assert (workdir / "train.csv").exists()
    assert (workdir / "val.csv").exists()
    stripped_path = workdir / "processed" / "a.dcm"
    assert stripped_path.exists()
    stripped = pydicom.dcmread(str(stripped_path))
    assert "PatientName" not in stripped


def test_stream_finetune_epochs_yields_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    workdir = tmp_path / "workdir"
    processed = workdir / "processed"
    processed.mkdir(parents=True)

    image_ids = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    for image_id in image_ids:
        np.save(
            processed / f"{image_id}.npy",
            np.random.rand(32, 32).astype(np.float32),
        )
    df = pd.DataFrame({"image_id": image_ids, "label": labels})
    df.to_csv(workdir / "train.csv", index=False)
    df.to_csv(workdir / "val.csv", index=False)

    base_checkpoint = tmp_path / "base.pt"
    torch.save(build_model("baseline").state_dict(), base_checkpoint)

    epochs = list(
        finetune.stream_finetune_epochs(
            workdir,
            "baseline",
            base_checkpoint,
            epochs=2,
            lr=1e-4,
            freeze_backbone=False,
        )
    )

    assert [e["epoch"] for e in epochs] == [0, 1]
    assert all("val_auc" in e and "train_loss" in e for e in epochs)
    assert (workdir / "adapter.pt").exists()
