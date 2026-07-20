"""Tests for image-level collapse of CBIS-DDSM per-abnormality rows."""

import pandas as pd

from src.data.splits import collapse_to_image_level

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
    # representative fields come from the malignant abnormality
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
