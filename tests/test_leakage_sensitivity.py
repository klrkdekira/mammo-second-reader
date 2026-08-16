"""Tests for the post-hoc leakage sensitivity analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.leakage_sensitivity import (
    contaminated_patients,
    split_frame,
)


def _ledger(tmp_path, patient_ids):
    path = tmp_path / "excluded.csv"
    pd.DataFrame(
        {
            "patient_id": patient_ids,
            "development_split": ["train"] * len(patient_ids),
            "n_train_images": [2] * len(patient_ids),
            "n_val_images": [0] * len(patient_ids),
            "n_test_images": [2] * len(patient_ids),
            "reason": ["patient_id_also_present_in_locked_test"] * len(patient_ids),
        }
    ).to_csv(path, index=False)
    return path


def _predictions(patient_ids):
    return pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(len(patient_ids))],
            "patient_id": patient_ids,
            "label": [i % 2 for i in range(len(patient_ids))],
        }
    )


def test_ledger_patient_ids_are_read_as_strings(tmp_path):
    path = _ledger(tmp_path, ["P_00016", "P_00041"])
    assert contaminated_patients(path) == {"P_00016", "P_00041"}


def test_clean_subset_drops_exactly_the_contaminated_patients():
    frame = _predictions(["P_00016", "P_00041", "P_00099", "P_00100"])
    full, clean = split_frame(frame, {"P_00016", "P_00041"})

    assert len(full) == 4
    assert set(clean["patient_id"]) == {"P_00099", "P_00100"}
    assert len(clean) == 2


def test_every_clean_row_survives_unchanged():
    """The subset must be a filter, never a recomputation."""
    frame = _predictions(["P_00016", "P_00099", "P_00100"])
    _, clean = split_frame(frame, {"P_00016"})

    expected = frame[frame["patient_id"] != "P_00016"].reset_index(drop=True)
    pd.testing.assert_frame_equal(clean, expected)


def test_an_empty_ledger_leaves_the_frame_intact():
    frame = _predictions(["P_00099", "P_00100"])
    full, clean = split_frame(frame, set())

    pd.testing.assert_frame_equal(clean, full)


def test_all_patients_contaminated_yields_an_empty_subset():
    frame = _predictions(["P_00016", "P_00041"])
    _, clean = split_frame(frame, {"P_00016", "P_00041"})

    assert clean.empty


@pytest.mark.parametrize("excluded", [{"P_00016"}, {"P_00016", "P_00041"}, set()])
def test_filtering_is_deterministic(excluded):
    frame = _predictions(["P_00016", "P_00041", "P_00099"])
    first = split_frame(frame, excluded)[1]
    second = split_frame(frame, excluded)[1]

    pd.testing.assert_frame_equal(first, second)
