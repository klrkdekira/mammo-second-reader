"""Synthetic geometry and leakage tests for Stage 0 patch extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import patch_manifest as pm


def _config(**overrides):
    values = {
        "seed": 42,
        "patch_size": 8,
        "lesion_patches_per_roi": 3,
        "background_patches_per_roi": 3,
        "min_roi_overlap": 0.9,
        "min_background_tissue_fraction": 0.5,
        "max_sampling_attempts": 1000,
        "use_clahe": False,
    }
    values.update(overrides)
    return pm.PatchExtractionConfig(**values)


def test_overlap_is_roi_coverage_or_patch_purity():
    small = np.zeros((32, 32), dtype=np.uint8)
    small[14:18, 14:18] = 1
    box = pm.PatchBox(10, 10, 22, 22)
    overlap = pm.overlap_with_roi(small, box)
    assert overlap.roi_coverage == 1.0
    assert overlap.patch_fraction == pytest.approx(16 / 144)
    assert overlap.score == 1.0

    large = np.ones((32, 32), dtype=np.uint8)
    overlap = pm.overlap_with_roi(large, box)
    assert overlap.roi_coverage == pytest.approx(144 / 1024)
    assert overlap.patch_fraction == 1.0
    assert overlap.score == 1.0


def test_lesion_sampling_is_deterministic_and_in_bounds():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[15:25, 20:30] = 1
    config = _config()

    first = pm.sample_lesion_boxes(mask, config, np.random.default_rng(123))
    second = pm.sample_lesion_boxes(mask, config, np.random.default_rng(123))

    assert first == second
    assert len(first) == config.lesion_patches_per_roi
    assert all(box.shape == (8, 8) for box, _, _ in first)
    assert all(0 <= box.y0 < box.y1 <= 40 for box, _, _ in first)
    assert all(0 <= box.x0 < box.x1 <= 50 for box, _, _ in first)
    assert all(overlap.score >= config.min_roi_overlap for _, overlap, _ in first)


def test_impossible_overlap_is_explicitly_marked_not_silently_relaxed():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[::2, ::2] = 1
    config = _config(
        lesion_patches_per_roi=2,
        min_roi_overlap=0.99,
        max_sampling_attempts=200,
    )
    sampled = pm.sample_lesion_boxes(mask, config, np.random.default_rng(8))

    assert len(sampled) == 2
    assert all(overlap.score < 0.99 for _, overlap, _ in sampled)
    assert all(reason.startswith("insufficient_overlap_0.990") for _, _, reason in sampled)


def test_background_sampling_has_zero_union_roi_overlap():
    union = np.zeros((48, 48), dtype=np.uint8)
    union[18:30, 18:30] = 1
    tissue = np.ones_like(union)
    config = _config()

    first = pm.sample_background_boxes(
        union, tissue, 12, config, np.random.default_rng(99)
    )
    second = pm.sample_background_boxes(
        union, tissue, 12, config, np.random.default_rng(99)
    )

    assert first == second
    assert len(first) == 12
    assert len({box for box, _ in first}) == 12
    for box, tissue_fraction in first:
        assert union[box.y0 : box.y1, box.x0 : box.x1].sum() == 0
        assert tissue_fraction == 1.0


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "image_a",
                "patient_id": "patient_train",
                "pathology": "MALIGNANT",
                "label": 1,
                "lesion_type": "mass",
                "roi_mask_id": "roi_mass",
                "split": "train",
            },
            {
                "image_id": "image_a",
                "patient_id": "patient_train",
                "pathology": "BENIGN",
                "label": 0,
                "lesion_type": "calc",
                "roi_mask_id": "roi_calc",
                "split": "train",
            },
        ]
    )


def test_end_to_end_generation_is_balanced_and_repeatable(tmp_path, monkeypatch):
    image = np.linspace(0, 1, 48 * 48, dtype=np.float32).reshape(48, 48)
    tissue = np.ones((48, 48), dtype=np.uint8)
    masks = {
        "roi_mass": np.pad(np.ones((12, 12), np.uint8), ((8, 28), (8, 28))),
        "roi_calc": np.pad(np.ones((12, 12), np.uint8), ((28, 8), (28, 8))),
    }

    monkeypatch.setattr(pm, "_find_dicom", lambda root, image_id: root / f"{image_id}.dcm")
    monkeypatch.setattr(pm, "load_dicom", lambda path: image)
    monkeypatch.setattr(
        pm,
        "preprocess_aligned_array",
        lambda raw, use_clahe: (image, tissue, (0, 48, 0, 48)),
    )
    monkeypatch.setattr(
        pm,
        "_load_roi",
        lambda path, source_shape, crop_box: masks[Path(path).stem],
    )
    config = _config()

    first, first_summary = pm.generate_patch_manifests(
        _source(), tmp_path / "raw", tmp_path / "first", config
    )
    second, second_summary = pm.generate_patch_manifests(
        _source(), tmp_path / "raw", tmp_path / "second", config
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    assert (first["patch_kind"] == "lesion").sum() == 6
    assert (first["patch_kind"] == "background").sum() == 6
    assert set(first["patch_class"]) == {
        "malignant_mass",
        "benign_calcification",
        "background",
    }
    assert (first.loc[first["patch_kind"] == "background", "union_roi_overlap_px"] == 0).all()
    assert not first["fallback_reason"].fillna("").astype(bool).any()

    for row in first.itertuples():
        a = np.load(tmp_path / "first" / row.patch_path)
        b = np.load(tmp_path / "second" / row.patch_path)
        np.testing.assert_array_equal(a, b)


def _valid_manifest_row(**overrides):
    row = {
        "patch_id": "p1",
        "patch_path": "patches/train/background/p1.npy",
        "patient_id": "patient_train",
        "image_id": "image_a",
        "roi_mask_id": "roi_a",
        "split": "train",
        "patch_class": "background",
        "class_id": pm.CLASS_TO_ID["background"],
        "patch_kind": "background",
        "sample_index": 0,
        "y0": 0,
        "x0": 0,
        "y1": 8,
        "x1": 8,
        "source_height": 32,
        "source_width": 32,
        "breast_y0": 1,
        "breast_x0": 2,
        "extraction_scale": 1.0,
        "roi_overlap": 0.0,
        "roi_coverage": 0.0,
        "patch_lesion_fraction": 0.0,
        "union_roi_overlap_px": 0,
        "tissue_fraction": 1.0,
        "fallback_reason": "",
    }
    row.update(overrides)
    return row


def test_manifest_rejects_test_rows_and_patient_leakage():
    test_row = pd.DataFrame([_valid_manifest_row(split="test")])
    with pytest.raises(ValueError, match="only train and val"):
        pm.validate_patch_manifest(test_row)

    leaked = pd.DataFrame(
        [
            _valid_manifest_row(),
            _valid_manifest_row(patch_id="p2", patch_path="p2.npy", split="val"),
        ]
    )
    with pytest.raises(ValueError, match="more than one patch split"):
        pm.validate_patch_manifest(leaked)


def test_manifest_rejects_background_roi_overlap_and_bad_geometry():
    overlap = pd.DataFrame([_valid_manifest_row(union_roi_overlap_px=1)])
    with pytest.raises(ValueError, match="background patch overlaps"):
        pm.validate_patch_manifest(overlap)

    outside = pd.DataFrame([_valid_manifest_row(y1=33)])
    with pytest.raises(ValueError, match="outside source-image bounds"):
        pm.validate_patch_manifest(outside)


def _locked_splits(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, patient, image in (
        ("train", "patient_train", "image_train"),
        ("val", "patient_val", "image_val"),
        ("test", "patient_test", "image_test"),
    ):
        pd.DataFrame([{"patient_id": patient, "image_id": image, "label": 0}]).to_csv(
            splits / f"{split}.csv", index=False
        )
    return splits


def test_source_manifest_retains_all_rois_but_excludes_test(tmp_path, monkeypatch):
    splits = _locked_splits(tmp_path)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for kind in ("mass", "calc"):
        (metadata / f"{kind}_case_description_train_set.csv").touch()
    dicom_root = tmp_path / "dicoms"
    dicom_root.mkdir()

    monkeypatch.setattr(pm, "DICOMPathResolver", lambda root: object())

    def fake_build(path, root, resolver):
        lesion = "mass" if path.name.startswith("mass") else "calc"
        rows = [
            {
                "image_id": "image_train",
                "patient_id": "patient_train",
                "pathology": "BENIGN",
                "label": 0,
                "lesion_type": lesion,
                "roi_mask_id": f"roi_train_{lesion}",
            },
            {
                "image_id": "image_val",
                "patient_id": "patient_val",
                "pathology": "MALIGNANT",
                "label": 1,
                "lesion_type": lesion,
                "roi_mask_id": f"roi_val_{lesion}",
            },
            {
                "image_id": "image_test",
                "patient_id": "patient_test",
                "pathology": "MALIGNANT",
                "label": 1,
                "lesion_type": lesion,
                "roi_mask_id": f"roi_test_{lesion}",
            },
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr(pm, "_build_dataframe", fake_build)
    source = pm.build_lesion_source_manifest(metadata, dicom_root, splits)

    assert len(source) == 4
    assert set(source["split"]) == {"train", "val"}
    assert set(source["patient_id"]) == {"patient_train", "patient_val"}
    assert not source["roi_mask_id"].str.contains("test").any()
    assert source.groupby("image_id")["roi_mask_id"].nunique().eq(2).all()


def test_locked_split_patient_overlap_is_rejected(tmp_path):
    splits = _locked_splits(tmp_path)
    val = pd.read_csv(splits / "val.csv")
    val.loc[0, "patient_id"] = "patient_train"
    val.to_csv(splits / "val.csv", index=False)

    with pytest.raises(ValueError, match="Patient leakage"):
        pm._read_split_assignments(splits)


def test_test_overlap_patient_is_quarantined_and_recorded(tmp_path):
    splits = _locked_splits(tmp_path)
    train = pd.read_csv(splits / "train.csv")
    train.loc[0, "patient_id"] = "patient_test"
    train.to_csv(splits / "train.csv", index=False)

    frames = pm._read_locked_split_frames(splits)
    ledger = pm.test_overlap_exclusion_ledger(frames)
    assignments, image_ids = pm._read_split_assignments(splits)

    assert ledger.to_dict("records") == [
        {
            "patient_id": "patient_test",
            "development_split": "train",
            "n_train_images": 1,
            "n_val_images": 0,
            "n_test_images": 1,
            "reason": "patient_id_also_present_in_locked_test",
        }
    ]
    assert "patient_test" not in assignments
    assert "image_train" not in image_ids["train"]
    assert "image_test" in image_ids["test"]


def test_quarantined_whole_image_splits_are_patient_disjoint(tmp_path):
    splits = _locked_splits(tmp_path)
    train = pd.read_csv(splits / "train.csv")
    train.loc[0, "patient_id"] = "patient_test"
    train.to_csv(splits / "train.csv", index=False)
    frames = pm._read_locked_split_frames(splits)
    ledger = pm.test_overlap_exclusion_ledger(frames)

    outputs = pm._write_quarantined_whole_image_splits(
        frames, ledger, tmp_path / "stage0"
    )

    assert {path.name for path in outputs} == {"train.csv", "val.csv", "test.csv"}
    clean_train = pd.read_csv(tmp_path / "stage0/whole_image_splits/train.csv")
    clean_test = pd.read_csv(tmp_path / "stage0/whole_image_splits/test.csv")
    assert clean_train.empty
    assert set(clean_test["patient_id"]) == {"patient_test"}


def test_stage0_config_rejects_unknown_sampling_knob(tmp_path):
    config = tmp_path / "stage0.toml"
    config.write_text(
        """
[paths]
metadata_dir = "metadata"
splits_dir = "splits"
raw_root = "raw"
out_dir = "out"

[extraction]
seed = 42
overlap_typo = 0.9
""".strip()
    )
    with pytest.raises(ValueError, match="Unknown Stage 0 extraction keys"):
        pm.load_patch_extraction_config(config)


def test_class_coverage_requires_all_five_classes_in_both_splits():
    rows = []
    patch_number = 0
    for split in ("train", "val"):
        for patch_class in pm.PATCH_CLASSES:
            for _ in range(2):
                rows.append(
                    _valid_manifest_row(
                        patch_id=f"p{patch_number}",
                        split=split,
                        patient_id=f"patient_{split}",
                        patch_class=patch_class,
                        class_id=pm.CLASS_TO_ID[patch_class],
                        patch_kind="background"
                        if patch_class == "background"
                        else "lesion",
                    )
                )
                patch_number += 1
    manifest = pd.DataFrame(rows)
    pm.validate_class_coverage(manifest, min_examples_per_class=4)

    missing = manifest[manifest["patch_class"] != "malignant_mass"]
    with pytest.raises(ValueError, match="missing classes"):
        pm.validate_class_coverage(missing, min_examples_per_class=1)
