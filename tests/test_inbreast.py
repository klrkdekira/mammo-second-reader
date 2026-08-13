"""Tests for the INbreast ingest and its locked label rule."""

import json
import plistlib

import numpy as np
import pandas as pd
import pytest

from src.data.inbreast import (
    BENIGN_ASSESSMENTS,
    MALIGNANT_ASSESSMENTS,
    build_manifest,
    lesion_present,
    lesion_type_for,
    read_rois,
    roi_mask_id,
    write_manifest,
)
from src.data.inbreast_roi import rasterise

# (file name, laterality, view, ACR density, raw BI-RADS)
_CASES = (
    ("100", "R", "CC", "2", "1"),
    ("101", "L", "MLO", "3", "2"),
    ("102", "R", "CC", "4", "3"),
    ("103", "L", "MLO", "1", "4a"),
    ("104", "R", "MLO", "2", "4c"),
    ("105", "L", "CC", "", "5"),
    ("106", "R", "CC", "3", "6"),
)
_PATIENTS = {
    "100": "aaaa000000000001",
    "101": "aaaa000000000001",
    "102": "bbbb000000000002",
    "103": "bbbb000000000002",
    "104": "cccc000000000003",
    "105": "cccc000000000003",
    "106": "dddd000000000004",
}


def _write_plist(path, rois):
    document = {
        "Images": [
            {
                "ImageIndex": 0,
                "NumberOfROIs": len(rois),
                "ROIs": [
                    {
                        "Name": name,
                        "NumberOfPoints": len(points),
                        "Point_px": [f"({x:.6f}, {y:.6f})" for x, y in points],
                    }
                    for name, points in rois
                ],
            }
        ]
    }
    with path.open("wb") as stream:
        plistlib.dump(document, stream)


@pytest.fixture
def release(tmp_path):
    """A miniature INbreast release directory."""
    root = tmp_path / "inbreast"
    dicom_dir = root / "AllDICOMs"
    xml_dir = root / "AllXML"
    dicom_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    lines = [
        "Patient ID;Patient age;Laterality;View;Acquisition date;File Name;ACR;Bi-Rads"
    ]
    for file_name, laterality, view, acr, birads in _CASES:
        # Real releases pad the blank ACR field with whitespace.
        padded = acr if acr else "     "
        lines.append(
            f"removed;removed;{laterality};{view};201001;{file_name};{padded};{birads}"
        )
        stem = f"{file_name}_{_PATIENTS[file_name]}_MG_{laterality}_{view}_ANON"
        (dicom_dir / f"{stem}.dcm").write_bytes(b"")

    (root / "INbreast.csv").write_text("\n".join(lines) + "\n")

    # BI-RADS 1 carries no annotation, matching the real release.
    _write_plist(xml_dir / "101.xml", [("Calcification", [(10.0, 10.0)])])
    _write_plist(xml_dir / "102.xml", [("Mass", [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)])])
    _write_plist(
        xml_dir / "103.xml",
        [("Mass", [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)]), ("Cluster", [(9.0, 9.0)])],
    )
    _write_plist(xml_dir / "104.xml", [("Calcifications", [(2.0, 3.0)])])
    _write_plist(xml_dir / "105.xml", [("Assymetry", [(4.0, 4.0)])])
    _write_plist(xml_dir / "106.xml", [("Calcification", [(6.0, 6.0)])])
    return root


def test_label_rule_matches_the_pre_registered_decision(release):
    frame = build_manifest(release)

    labels = dict(zip(frame["file_name"], frame["label"]))
    assert labels == {
        "100": 0,  # BI-RADS 1
        "101": 0,  # BI-RADS 2
        "102": 0,  # BI-RADS 3 folded into benign, not excluded
        "103": 1,  # 4a
        "104": 1,  # 4c
        "105": 1,  # 5
        "106": 1,  # 6
    }
    assert len(frame) == len(_CASES), "no image may be excluded under this rule"
    assert set(MALIGNANT_ASSESSMENTS).isdisjoint(BENIGN_ASSESSMENTS)


def test_patient_id_comes_from_the_filename_not_the_csv(release):
    frame = build_manifest(release)

    assert "removed" not in set(frame["patient_id"])
    assert frame["patient_id"].nunique() == 4
    assert (
        frame.loc[frame["file_name"] == "100", "patient_id"].iloc[0]
        == "aaaa000000000001"
    )


def test_blank_density_becomes_missing_not_zero(release):
    frame = build_manifest(release)

    density = frame.set_index("file_name")["birads_density"]
    assert pd.isna(density["105"])
    assert density["103"] == 1
    assert str(frame["birads_density"].dtype) == "Int64"


def test_unmapped_assessment_is_rejected(release):
    text = (release / "INbreast.csv").read_text().replace(";3\n", ";9\n", 1)
    (release / "INbreast.csv").write_text(text)

    with pytest.raises(ValueError, match="Unmapped BI-RADS"):
        build_manifest(release)


def test_csv_and_dicom_mismatch_is_rejected(release):
    next(iter((release / "AllDICOMs").glob("100_*.dcm"))).unlink()

    with pytest.raises(ValueError, match="disagree"):
        build_manifest(release)


def test_lesion_type_only_uses_clean_single_family_images(release):
    frame = build_manifest(release).set_index("file_name")

    assert pd.isna(frame.loc["100", "lesion_type"]), "unannotated normal"
    assert frame.loc["101", "lesion_type"] == "calcification"
    assert frame.loc["102", "lesion_type"] == "mass"
    assert frame.loc["103", "lesion_type"] == "mixed", "mass plus calcification"
    assert frame.loc["105", "lesion_type"] == "other", "asymmetry is neither family"


def test_lesion_type_for_handles_typo_vocabulary():
    from src.data.inbreast import Roi

    assert lesion_type_for([Roi("Calcifications", [(0.0, 0.0)])]) == "calcification"
    assert lesion_type_for([Roi("Cluster", [(0.0, 0.0)])]) == "calcification"
    assert lesion_type_for([Roi("Espiculated Region", [(0.0, 0.0)])]) == "other"
    assert lesion_type_for([]) is None


def test_roi_mask_id_is_set_only_for_annotated_images(release):
    frame = build_manifest(release).set_index("file_name")

    assert pd.isna(frame.loc["100", "roi_mask_id"])
    assert frame.loc["101", "roi_mask_id"] == roi_mask_id("101")


def test_lesion_present_subset_drops_only_birads_one(release):
    frame = build_manifest(release)
    subset = lesion_present(frame)

    assert len(subset) == len(frame) - 1
    assert "100" not in set(subset["file_name"])
    assert subset["roi_mask_id"].notna().all()


def test_write_manifest_records_the_lock(release, tmp_path):
    primary, secondary, lock = write_manifest(release, tmp_path / "manifest")

    assert primary.name == "test.csv", "named so dicom_to_png can cache it unchanged"
    record = json.loads(lock.read_text())
    assert record["primary"]["n_images"] == 7
    assert record["primary"]["n_malignant"] == 4
    assert record["secondary"]["n_images"] == 6
    assert record["label"]["excluded"] == []
    assert "not_pathological" in record["label"]["source"]
    assert record["primary"]["lesion_type"]["unannotated"] == 1
    saved = pd.read_csv(secondary)
    assert len(saved) == 6


def test_read_rois_parses_points_and_skips_pointless_rois(tmp_path):
    path = tmp_path / "roi.xml"
    _write_plist(path, [("Mass", [(1.5, 2.5)]), ("Unnamed", [])])

    rois = read_rois(path)

    assert len(rois) == 1
    assert rois[0].points == [(1.5, 2.5)]


def test_read_rois_rejects_a_corrupt_file(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("<plist>not really")

    with pytest.raises(ValueError, match="Cannot parse"):
        read_rois(path)


class _FakeRoi:
    def __init__(self, points):
        self.points = points


def test_rasterise_fills_polygons_and_dots_single_points():
    polygon = rasterise([_FakeRoi([(1, 1), (8, 1), (8, 8), (1, 8)])], (20, 20))
    point = rasterise([_FakeRoi([(10, 10)])], (20, 20), point_radius=2)

    assert polygon[5, 5] == 1
    assert polygon[15, 15] == 0
    assert point[10, 10] == 1
    assert point.sum() < polygon.sum(), "a speck must be smaller than a mass contour"
    assert set(np.unique(point)).issubset({0, 1})


def test_rasterise_clips_out_of_frame_points_instead_of_dropping_them():
    mask = rasterise([_FakeRoi([(-5, -5)])], (10, 10), point_radius=3)

    assert mask.any(), "an annotation rounded outside the frame must not vanish"
