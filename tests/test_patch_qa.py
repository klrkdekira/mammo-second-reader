"""Synthetic tests for supplementary Stage 0 visual QA."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import patch_qa as qa
from src.data.patch_manifest import CLASS_TO_ID, PATCH_CLASSES, overlap_with_roi


def _manifest_row(
    patch_class: str,
    split: str,
    unit: int,
    *,
    fallback: bool = False,
    metric: float = 1.0,
) -> dict[str, object]:
    kind = "background" if patch_class == "background" else "lesion"
    image_id = f"image_{patch_class}_{split}_{unit}"
    roi_id = f"roi_{patch_class}_{split}_{unit}"
    return {
        "patch_id": f"patch_{patch_class}_{split}_{unit}_{fallback}",
        "patch_path": f"patches/{split}/{patch_class}/{unit}.npy",
        "patient_id": f"patient_{split}_{unit}",
        "image_id": image_id,
        "roi_mask_id": roi_id,
        "split": split,
        "patch_class": patch_class,
        "class_id": CLASS_TO_ID[patch_class],
        "patch_kind": kind,
        "sample_index": 0,
        "y0": 0,
        "x0": 0,
        "y1": 8,
        "x1": 8,
        "source_height": 16,
        "source_width": 16,
        "breast_y0": 0,
        "breast_x0": 0,
        "extraction_scale": 1.0,
        "roi_overlap": 0.0 if kind == "background" else metric,
        "roi_coverage": 0.0 if kind == "background" else metric,
        "patch_lesion_fraction": 0.0 if kind == "background" else metric,
        "union_roi_overlap_px": 0 if kind == "background" else 16,
        "tissue_fraction": metric if kind == "background" else 1.0,
        "fallback_reason": "insufficient_overlap_0.900:0/10" if fallback else "",
    }


def _selection_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patch_class in PATCH_CLASSES:
        for split in ("train", "val"):
            for unit in range(8):
                rows.append(
                    _manifest_row(
                        patch_class,
                        split,
                        unit,
                        metric=0.91 + unit / 100,
                    )
                )
                if patch_class != "background":
                    rows.append(
                        _manifest_row(
                            patch_class,
                            split,
                            unit + 20,
                            fallback=True,
                            metric=0.4 + unit / 20,
                        )
                    )
    return pd.DataFrame(rows)


def test_review_selection_is_deterministic_unique_and_split_covered():
    manifest = _selection_manifest()
    config = qa.PatchQAConfig(seed=42, n_per_group=4)

    first = qa.select_review_cases(manifest, config)
    second = qa.select_review_cases(manifest, config)

    pd.testing.assert_frame_equal(first, second)
    assert first["audit_group"].nunique() == 10
    assert first.groupby("audit_group").size().eq(4).all()
    assert first.groupby("audit_group")["split"].nunique().eq(2).all()
    background = first[first["patch_class"] == "background"]
    assert background["image_id"].nunique() == len(background)
    for audit_group, group in first.groupby("audit_group"):
        if str(audit_group).startswith("background/"):
            assert group["image_id"].nunique() == len(group)
        else:
            assert group["roi_mask_id"].nunique() == len(group)
        if str(audit_group).endswith("/fallback"):
            assert group["fallback_reason"].ne("").all()
        if str(audit_group).endswith("/representative"):
            assert group["fallback_reason"].eq("").all()


def _case_row(
    patch_path: Path,
    *,
    patch_kind: str,
    box: tuple[int, int, int, int],
    target: np.ndarray,
    union: np.ndarray,
) -> pd.Series:
    y0, x0, y1, x1 = box
    overlap = overlap_with_roi(target, qa.PatchBox(y0, x0, y1, x1))
    return pd.Series(
        {
            "patch_id": f"{patch_kind}-patch",
            "patch_path": patch_path.as_posix(),
            "roi_mask_id": "target",
            "patch_kind": patch_kind,
            "y0": y0,
            "x0": x0,
            "y1": y1,
            "x1": x1,
            "roi_overlap": overlap.score if patch_kind == "lesion" else 0.0,
            "roi_coverage": overlap.roi_coverage if patch_kind == "lesion" else 0.0,
            "patch_lesion_fraction": overlap.patch_fraction
            if patch_kind == "lesion"
            else 0.0,
            "union_roi_overlap_px": int(union[y0:y1, x0:x1].sum()),
        }
    )


def test_reconstructed_lesion_case_matches_frozen_pixels_and_geometry(tmp_path):
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 255
    target = np.zeros_like(image, dtype=np.uint8)
    target[4:8, 4:8] = 1
    other = np.zeros_like(target)
    other[12:14, 12:14] = 1
    source = qa.ReviewSource(image, {"target": target, "other": other}, (0, 16, 0, 16))
    relative = Path("patches/train/lesion.npy")
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    np.save(destination, image[3:9, 3:9])
    row = _case_row(
        relative,
        patch_kind="lesion",
        box=(3, 3, 9, 9),
        target=target,
        union=source.union_mask,
    )

    result = qa.validate_reconstructed_case(row, source, tmp_path)

    assert result["exact_pixel_match"] is True
    assert result["roi_overlap"] == pytest.approx(1.0)
    assert result["union_roi_overlap_px"] == 16


def test_reconstructed_case_rejects_pixel_drift_and_background_overlap(tmp_path):
    image = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.uint8)
    target[10:14, 10:14] = 1
    source = qa.ReviewSource(image, {"target": target}, (0, 16, 0, 16))
    relative = Path("patches/train/background.npy")
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    np.save(destination, np.ones((4, 4), dtype=np.float32))
    row = _case_row(
        relative,
        patch_kind="background",
        box=(0, 0, 4, 4),
        target=target,
        union=source.union_mask,
    )
    with pytest.raises(ValueError, match="does not reconstruct exactly"):
        qa.validate_reconstructed_case(row, source, tmp_path)

    np.save(destination, image[10:14, 10:14])
    overlapping = _case_row(
        relative,
        patch_kind="background",
        box=(10, 10, 14, 14),
        target=target,
        union=source.union_mask,
    )
    with pytest.raises(ValueError, match="overlaps a known ROI"):
        qa.validate_reconstructed_case(overlapping, source, tmp_path)


def test_generator_refuses_to_overwrite_existing_review(tmp_path):
    output = tmp_path / "qa_review"
    output.mkdir()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        qa.generate_patch_qa(
            tmp_path / "raw",
            tmp_path / "data",
            output,
            False,
            qa.PatchQAConfig(n_per_group=1),
        )


def test_generator_writes_locked_atomic_review_package(tmp_path, monkeypatch):
    rows: list[dict[str, object]] = []
    for patch_class in PATCH_CLASSES:
        rows.append(_manifest_row(patch_class, "train", 1, metric=1.0))
        if patch_class != "background":
            rows.append(
                _manifest_row(
                    patch_class,
                    "train",
                    21,
                    fallback=True,
                    metric=0.5,
                )
            )
        else:
            rows.append(_manifest_row(patch_class, "train", 2, metric=0.8))
    manifest = pd.DataFrame(rows)
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in ("manifest-lock.json", "train.csv", "val.csv"):
        (data_root / name).write_text(name + "\n")

    monkeypatch.setattr(
        qa,
        "load_frozen_manifest",
        lambda root: (manifest, {"patch_tree_sha256": "frozen-tree"}),
    )

    def fake_source(image_rows, raw_root, *, use_clahe):
        del image_rows, raw_root, use_clahe
        image = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
        target = np.zeros((16, 16), dtype=np.uint8)
        target[2:8, 2:8] = 1
        return qa.ReviewSource(
            image, {"roi_background_train_1": target}, (0, 16, 0, 16)
        )

    monkeypatch.setattr(qa, "_load_review_source", fake_source)
    monkeypatch.setattr(
        qa,
        "validate_reconstructed_case",
        lambda row, source, root: {
            "patch_id": str(row["patch_id"]),
            "exact_pixel_match": True,
            "union_roi_overlap_px": 0,
            **(
                {"roi_overlap": float(row["roi_overlap"])}
                if row["patch_kind"] == "lesion"
                else {}
            ),
        },
    )
    monkeypatch.setattr(qa, "_render_case", lambda *args, **kwargs: None)
    output = data_root / "qa_review"

    qa.generate_patch_qa(
        tmp_path / "raw",
        data_root,
        output,
        False,
        qa.PatchQAConfig(n_per_group=1),
    )

    assert (output / "selection.csv").is_file()
    assert (output / "reconstruction-checks.csv").is_file()
    assert len(list((output / "grids").glob("*.png"))) == 10
    summary = json.loads((output / "qa-review-summary.json").read_text())
    assert summary["status"] == "pending_manual_review"
    assert summary["stage_a_training_authorised"] is False
    assert summary["automated_reconstruction"]["n_cases"] == 10
    lock = json.loads((output / "qa-review-lock.json").read_text())
    assert lock["stage0_patch_tree_sha256"] == "frozen-tree"
    assert "selection.csv" in lock["outputs"]
    assert "reconstruction-checks.csv" in lock["outputs"]
