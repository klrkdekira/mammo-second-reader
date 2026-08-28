"""Build the INbreast external-test manifest."""

from __future__ import annotations

import json
import logging
import plistlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd

from src.config import setup_logging
from src.data import manifest as _manifest

LOGGER = logging.getLogger(__name__)

INGEST_VERSION = 1

MALIGNANT_ASSESSMENTS = ("4a", "4b", "4c", "5", "6")
BENIGN_ASSESSMENTS = ("1", "2", "3")
LABEL_RULE = "birads_4a4b4c5and6_malignant__birads_123_benign__no_exclusions"

NORMAL_ASSESSMENT = "1"

_DICOM_STEM = re.compile(
    r"^(?P<file>\d+)_(?P<patient>[0-9a-f]+)_MG_(?P<laterality>[LR])_(?P<view>[A-Z]+)_ANON$"
)

_CALCIFICATION_NAMES = frozenset({"calcification", "calcifications", "cluster"})
_MASS_NAMES = frozenset({"mass"})

_POINT = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")


@dataclass(frozen=True)
class Roi:
    """One annotated region from an INbreast OsiriX XML export."""

    name: str
    points: list[tuple[float, float]]

    @property
    def normalised_name(self) -> str:
        return self.name.strip().lower()

    @property
    def family(self) -> str:
        """Coarse lesion family, comparable with the CBIS-DDSM strata."""
        name = self.normalised_name
        if name in _CALCIFICATION_NAMES:
            return "calcification"
        if name in _MASS_NAMES:
            return "mass"
        return "other"


def read_rois(xml_path: str | Path) -> list[Roi]:
    """Parse pixel coordinates from an INbreast OsiriX XML export."""
    path = Path(xml_path)
    with path.open("rb") as stream:
        try:
            document = plistlib.load(stream)
        except Exception as exc:  # plistlib raises several unrelated types
            raise ValueError(f"Cannot parse INbreast XML {path}: {exc}") from exc

    rois: list[Roi] = []
    for image in document.get("Images", []) or []:
        for roi in image.get("ROIs", []) or []:
            points: list[tuple[float, float]] = []
            for raw in roi.get("Point_px", []) or []:
                match = _POINT.search(str(raw))
                if match:
                    points.append((float(match.group(1)), float(match.group(2))))
            if points:
                rois.append(Roi(name=str(roi.get("Name", "")), points=points))
    return rois


def lesion_type_for(rois: Sequence[Roi]) -> str | None:
    """Map ROI annotations to a subgroup lesion type."""
    families = {roi.family for roi in rois}
    has_mass = "mass" in families
    has_calc = "calcification" in families
    if has_mass and has_calc:
        return "mixed"
    if has_mass:
        return "mass"
    if has_calc:
        return "calcification"
    return "other" if families else None


def _read_metadata(csv_path: Path) -> pd.DataFrame:
    """Read `INbreast.csv`, which is semicolon-delimited with padded fields."""
    frame = pd.read_csv(csv_path, sep=";", dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    expected = {"Laterality", "View", "File Name", "ACR", "Bi-Rads"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def _index_dicoms(dicom_dir: Path) -> dict[str, dict[str, str]]:
    """Map each `File Name` to its DICOM stem and anonymised patient hash."""
    index: dict[str, dict[str, str]] = {}
    for path in sorted(dicom_dir.glob("*.dcm")):
        match = _DICOM_STEM.match(path.stem)
        if match is None:
            LOGGER.warning("Skipping unrecognised DICOM filename: %s", path.name)
            continue
        fields = match.groupdict()
        key = fields["file"]
        if key in index:
            raise ValueError(f"Duplicate INbreast File Name on disk: {key}")
        index[key] = {"stem": path.stem, **fields}
    return index


def _assessment(raw: str) -> tuple[int, int]:
    """Return (numeric BI-RADS, binary label) for one raw assessment string."""
    value = raw.strip().lower()
    if value in MALIGNANT_ASSESSMENTS:
        label = 1
    elif value in BENIGN_ASSESSMENTS:
        label = 0
    else:
        raise ValueError(
            f"Unmapped BI-RADS assessment {raw!r}. Add it explicitly to "
            "MALIGNANT_ASSESSMENTS or BENIGN_ASSESSMENTS."
        )
    digits = re.match(r"^(\d)", value)
    if digits is None:
        raise ValueError(f"BI-RADS assessment {raw!r} has no leading category digit.")
    return int(digits.group(1)), label


def _density(raw: str) -> int | None:
    """ACR density 1-4, or None when the field is blank."""
    value = raw.strip()
    return int(value) if value in {"1", "2", "3", "4"} else None


def build_manifest(root: Path, *, xml_dir: Path | None = None) -> pd.DataFrame:
    """Build the full INbreast manifest from the release directory."""
    root = Path(root)
    dicom_dir = root / "AllDICOMs"
    xml_dir = Path(xml_dir) if xml_dir is not None else root / "AllXML"
    metadata = _read_metadata(root / "INbreast.csv")
    dicoms = _index_dicoms(dicom_dir)

    csv_keys = set(metadata["File Name"])
    disk_keys = set(dicoms)
    if csv_keys != disk_keys:
        raise ValueError(
            "INbreast.csv and AllDICOMs disagree: "
            f"{len(csv_keys - disk_keys)} row(s) without a DICOM, "
            f"{len(disk_keys - csv_keys)} DICOM(s) without a row."
        )

    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        key = str(record["File Name"])
        entry = dicoms[key]
        birads_raw = str(record["Bi-Rads"])
        assessment, label = _assessment(birads_raw)
        xml_path = xml_dir / f"{key}.xml"
        rois = read_rois(xml_path) if xml_path.is_file() else []
        rows.append(
            {
                "image_id": entry["stem"],
                "label": label,
                "patient_id": entry["patient"],
                "dataset": "inbreast",
                "birads_density": _density(str(record["ACR"])),
                "birads_assessment": assessment,
                "birads_raw": birads_raw,
                "roi_mask_id": roi_mask_id(key) if rois else None,
                "lesion_type": lesion_type_for(rois),
                "view": str(record["View"]),
                "laterality": str(record["Laterality"]),
                "n_annotated_rois": len(rois),
                "file_name": key,
            }
        )

    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    frame["birads_density"] = frame["birads_density"].astype("Int64")
    frame["birads_assessment"] = frame["birads_assessment"].astype("Int64")
    frame["file_name"] = frame["file_name"].astype(str)
    _manifest.validate(frame, source="inbreast ingest")
    return frame


def roi_mask_id(file_name: str) -> str:
    """Cache stem for the rasterised lesion mask of one INbreast image."""
    return f"{file_name}_roi"


def lesion_present(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the pre-registered subset that excludes BI-RADS 1 normals."""
    keep = frame["birads_raw"].astype(str).str.strip() != NORMAL_ASSESSMENT
    return frame.loc[keep].reset_index(drop=True)


def _tally(values: pd.Series) -> dict[str, int]:
    """Value counts with missing entries labelled, not rendered as "nan"."""
    counts = values.value_counts(dropna=False).sort_index(na_position="last")
    labels = pd.Index(counts.index).astype(object).fillna("unannotated")
    return {str(label): int(count) for label, count in zip(labels, counts.to_numpy())}


def _counts(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "n_images": len(frame),
        "n_patients": int(frame["patient_id"].nunique()),
        "n_malignant": int((frame["label"] == 1).sum()),
        "n_benign": int((frame["label"] == 0).sum()),
        "prevalence": round(float((frame["label"] == 1).mean()), 6),
        "birads_raw": _tally(frame["birads_raw"]),
        "birads_density": _tally(frame["birads_density"]),
        "lesion_type": _tally(frame["lesion_type"]),
        "view": _tally(frame["view"]),
        "n_with_roi_mask": int(frame["roi_mask_id"].notna().sum()),
    }


def build_lock(full: pd.DataFrame, subset: pd.DataFrame) -> dict[str, object]:
    """Pre-registration record for the locked external manifest."""
    return {
        "version": INGEST_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "inbreast",
        "role": "cold_external_test",
        "label": {
            "source": "birads_assessment_radiological_not_pathological",
            "rule": LABEL_RULE,
            "malignant": list(MALIGNANT_ASSESSMENTS),
            "benign": list(BENIGN_ASSESSMENTS),
            "excluded": [],
            "note": (
                "CBIS-DDSM training labels are biopsy-confirmed pathology. "
                "INbreast labels are the reporting radiologist's assessment, so "
                "the external target is a related but different construct."
            ),
        },
        "patient_id": {
            "source": "dicom_filename_field_2",
            "reason": "the INbreast.csv patient column is the literal string 'removed'",
        },
        "primary": {"subset": "full", **_counts(full)},
        "secondary": {
            "subset": "lesion_present_birads_ge_2",
            "reason": (
                "CBIS-DDSM has no normal images; dropping BI-RADS 1 separates "
                "imaging domain shift from case-mix shift."
            ),
            **_counts(subset),
        },
    }


def write_manifest(
    root: Path,
    out_dir: Path,
    *,
    xml_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write the locked manifest, the lesion-present subset, and the lock file.

    The primary manifest is named `test.csv` so `src.data.dicom_to_png` and
    `src.data.cache_roi_masks` can cache it with no changes: both discover work
    by reading `train.csv`/`val.csv`/`test.csv` from a splits directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = build_manifest(root, xml_dir=xml_dir)
    subset = lesion_present(full)

    primary_path = out_dir / "test.csv"
    subset_path = out_dir / "test_lesion_present.csv"
    lock_path = out_dir / "manifest-lock.json"
    full.to_csv(primary_path, index=False)
    subset.to_csv(subset_path, index=False)
    lock_path.write_text(json.dumps(build_lock(full, subset), indent=2) + "\n")
    return primary_path, subset_path, lock_path


def main(root: Path, out_dir: Path, xml_dir: Path | None = None) -> None:
    setup_logging()
    primary, subset, lock = write_manifest(root, out_dir, xml_dir=xml_dir)
    record = json.loads(lock.read_text())
    LOGGER.info(
        "Locked INbreast manifest: %d images / %d patients, prevalence %.3f",
        record["primary"]["n_images"],
        record["primary"]["n_patients"],
        record["primary"]["prevalence"],
    )
    LOGGER.info(
        "Lesion-present subset: %d images / %d patients, prevalence %.3f",
        record["secondary"]["n_images"],
        record["secondary"]["n_patients"],
        record["secondary"]["prevalence"],
    )
    LOGGER.info("Wrote %s, %s and %s", primary, subset, lock)


@click.command()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/inbreast"),
    show_default=True,
    help="INbreast Release 1.0 directory (contains AllDICOMs and INbreast.csv).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/inbreast/manifest"),
    show_default=True,
)
@click.option(
    "--xml-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Annotation directory. Defaults to <root>/AllXML.",
)
def cli(root: Path, out_dir: Path, xml_dir: Path | None) -> None:
    try:
        main(root, out_dir, xml_dir)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()


__all__ = [
    "BENIGN_ASSESSMENTS",
    "LABEL_RULE",
    "MALIGNANT_ASSESSMENTS",
    "Roi",
    "build_lock",
    "build_manifest",
    "lesion_present",
    "lesion_type_for",
    "read_rois",
    "roi_mask_id",
    "write_manifest",
]
