"""Tests for CBIS-DDSM image collapse and canonical patient-disjoint splits."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.data.manifest import assert_patient_disjoint, read_split_frames
from src.data.splits import (
    collapse_to_image_level,
    quarantine_test_overlaps,
    write_split_bundle,
)

COLUMNS = [
    "image_id",
    "patient_id",
    "dataset",
    "pathology",
    "label",
    "birads_density",
    "lesion_type",
    "subtlety",
    "roi_mask_id",
]


def _row(image_id, label, lesion_type, roi_mask_id, patient_id="P1"):
    return {
        "image_id": image_id,
        "patient_id": patient_id,
        "dataset": "cbis_ddsm",
        "pathology": "MALIGNANT" if label else "BENIGN",
        "label": label,
        "birads_density": 2,
        "lesion_type": lesion_type,
        "subtlety": 3,
        "roi_mask_id": roi_mask_id,
    }


def _df(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_single_abnormality_images_are_untouched():
    df = _df([_row("img_a", 0, "mass", "roi_a"), _row("img_b", 1, "calc", "roi_b")])
    out = collapse_to_image_level(df)
    assert len(out) == 2
    assert set(out["image_id"]) == {"img_a", "img_b"}


def test_two_abnormalities_collapse_to_one_row():
    df = _df([_row("img_a", 0, "mass", "roi_1"), _row("img_a", 0, "calc", "roi_2")])
    out = collapse_to_image_level(df)
    assert len(out) == 1
    assert out.iloc[0]["image_id"] == "img_a"


def test_malignant_if_any():
    df = _df([_row("img_a", 0, "mass", "roi_1"), _row("img_a", 1, "calc", "roi_2")])
    out = collapse_to_image_level(df)
    assert out.iloc[0]["label"] == 1
    assert out.iloc[0]["roi_mask_id"] == "roi_2"


def test_mixed_lesion_type_marked_mixed():
    df = _df([_row("img_a", 1, "mass", "roi_1"), _row("img_a", 1, "calc", "roi_2")])
    out = collapse_to_image_level(df)
    assert out.iloc[0]["lesion_type"] == "mixed"


def test_same_lesion_type_preserved():
    df = _df([_row("img_a", 1, "mass", "roi_1"), _row("img_a", 0, "mass", "roi_2")])
    out = collapse_to_image_level(df)
    assert out.iloc[0]["lesion_type"] == "mass"


def test_empty_frame_is_returned_unchanged():
    df = _df([])
    out = collapse_to_image_level(df)
    assert out.empty


def _split_frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_mass_train_calc_test_collision_uses_locked_test_precedence():
    collision = "P_cross_family"
    frames = {
        "train": _split_frame(
            [_row("Mass-Training/case", 0, "mass", "roi_1", collision)]
        ),
        "val": _split_frame([_row("Mass-Training/val", 1, "mass", "roi_2", "P_val")]),
        "test": _split_frame(
            [_row("Calc-Test/case", 1, "calcification", "roi_3", collision)]
        ),
    }

    clean, ledger = quarantine_test_overlaps(frames)

    assert clean["train"].empty
    assert set(clean["test"]["patient_id"]) == {collision}
    assert ledger.to_dict("records") == [
        {
            "patient_id": collision,
            "development_split": "train",
            "n_train_images": 1,
            "n_val_images": 0,
            "n_test_images": 1,
            "reason": "patient_id_also_present_in_locked_test",
        }
    ]
    assert_patient_disjoint(clean)


def test_train_validation_collision_has_no_precedence():
    frames = {
        "train": _split_frame([_row("train", 0, "mass", "r1", "P_shared")]),
        "val": _split_frame([_row("val", 1, "mass", "r2", "P_shared")]),
        "test": _split_frame([_row("test", 0, "mass", "r3", "P_test")]),
    }

    with pytest.raises(ValueError, match="train and val"):
        quarantine_test_overlaps(frames)


@pytest.mark.parametrize(
    ("left", "right"),
    [("train", "val"), ("train", "test"), ("val", "test")],
)
def test_preflight_rejects_every_pairwise_patient_collision(left, right):
    frames = {
        "train": _split_frame([_row("train", 0, "mass", "r1", "P_train")]),
        "val": _split_frame([_row("val", 1, "mass", "r2", "P_val")]),
        "test": _split_frame([_row("test", 0, "mass", "r3", "P_test")]),
    }
    frames[right].loc[0, "patient_id"] = frames[left].loc[0, "patient_id"]

    with pytest.raises(ValueError, match=f"{left} and {right}"):
        assert_patient_disjoint(frames)


def test_split_bundle_is_deterministic_and_preflighted(tmp_path):
    frames = {
        "train": _split_frame([_row("train", 0, "mass", "r1", "P_train")]),
        "val": _split_frame([_row("val", 1, "mass", "r2", "P_val")]),
        "test": _split_frame([_row("test", 0, "mass", "r3", "P_test")]),
    }
    clean, ledger = quarantine_test_overlaps(frames)
    outputs = write_split_bundle(clean, ledger, tmp_path)
    first = {name: path.read_bytes() for name, path in outputs.items()}
    write_split_bundle(clean, ledger, tmp_path)

    assert first == {name: path.read_bytes() for name, path in outputs.items()}
    assert_patient_disjoint(read_split_frames(tmp_path))


def test_frozen_canonical_manifests_match_protocol():
    expected = {
        "train": (
            "add9c8b8bc95fe86e21673d7438f8179c2cde344ff5b0cf439690145cfdb8d18",
            2147,
            1091,
            {0: 1180, 1: 967},
        ),
        "val": (
            "9c8dbd5b413d2c2d74cdec93c8ba0bdf6f3a9ae81c952f4705928b87f7b8ea5f",
            247,
            126,
            {0: 135, 1: 112},
        ),
        "test": (
            "225241c53968e10c18e4040c67edef304009e4bd4a3c5f73e67f7958a8e85634",
            645,
            349,
            {0: 381, 1: 264},
        ),
    }
    root = Path("manifests/cbis-ddsm")
    frames = read_split_frames(root)

    for split, (digest, rows, patients, labels) in expected.items():
        path = root / f"{split}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert len(frames[split]) == rows
        assert frames[split]["patient_id"].nunique() == patients
        assert frames[split]["label"].value_counts().sort_index().to_dict() == labels
    assert_patient_disjoint(frames)
