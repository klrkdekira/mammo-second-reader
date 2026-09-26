"""Tests for the INbreast web upload archive builder."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.make_finetune_archive import verify_archive
from src.data.make_inbreast_archives import (
    EVAL_MANIFEST,
    patient_split,
    verify_eval_archive,
    write_eval_archive,
    write_finetune_archive,
)


def _inbreast(tmp_path, n_patients=12, per_patient=4):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rows = []
    for patient in range(n_patients):
        for view in range(per_patient):
            image_id = f"{patient:08d}_{patient:04x}abcd_MG_L_V{view}_ANON"
            np.save(cache_dir / f"{image_id}.npy", np.zeros((16, 16), np.float32))
            rows.append(
                {
                    "image_id": image_id,
                    "label": int(patient % 3 == 0 and view < 2),
                    "patient_id": f"{patient:04x}abcd",
                    "birads_density": 2,
                    "birads_raw": "4a" if patient % 3 == 0 else "2",
                    "lesion_type": "mass",
                }
            )
    return pd.DataFrame(rows), cache_dir


def test_patient_split_is_disjoint_and_keeps_both_classes(tmp_path):
    frame, _ = _inbreast(tmp_path)
    train, val = patient_split(frame, val_fraction=0.25, seed=1)

    assert not set(train["patient_id"]) & set(val["patient_id"])
    assert len(train) + len(val) == len(frame)
    assert set(train["label"]) == {0, 1}
    assert set(val["label"]) == {0, 1}


def test_patient_split_rejects_single_malignant_patient(tmp_path):
    frame, _ = _inbreast(tmp_path, n_patients=3)
    with pytest.raises(ValueError, match="at least 2 patients"):
        patient_split(frame, val_fraction=0.25, seed=1)


def test_eval_archive_passes_web_loader(tmp_path):
    frame, cache_dir = _inbreast(tmp_path)
    output = tmp_path / "eval.zip"
    summary = write_eval_archive(
        output,
        frame,
        cache_dir=cache_dir,
        raw_root=cache_dir,
        source="npy",
        n_images=None,
        seed=1,
    )

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert names.count(EVAL_MANIFEST) == 1
    assert sum(name.endswith(".csv") for name in names) == 1
    assert summary["n"] == len(frame)
    assert verify_eval_archive(output) == len(frame)


def test_eval_archive_skips_missing_images(tmp_path):
    frame, cache_dir = _inbreast(tmp_path)
    (cache_dir / f"{frame.loc[1, 'image_id']}.npy").unlink()
    output = tmp_path / "eval.zip"
    summary = write_eval_archive(
        output,
        frame,
        cache_dir=cache_dir,
        raw_root=cache_dir,
        source="npy",
        n_images=None,
        seed=1,
    )
    assert summary["n"] == len(frame) - 1


def test_finetune_archive_passes_web_loader(tmp_path):
    frame, cache_dir = _inbreast(tmp_path)
    output = tmp_path / "finetune.zip"
    summary = write_finetune_archive(
        output,
        frame,
        cache_dir=cache_dir,
        raw_root=cache_dir,
        source="npy",
        val_fraction=0.25,
        n_train=10,
        n_val=None,
        seed=1,
    )

    assert summary["n_train"] == 10
    checked = verify_archive(output)
    assert checked == {"train_rows": 10, "val_rows": summary["n_val"]}
