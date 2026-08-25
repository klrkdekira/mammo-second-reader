"""Shared manifest schema for all mammography datasets.

Every CSV that drives MammogramDataset must conform to this schema.
Required columns are `image_id` and `label`. All others are optional
and may be absent or NaN for datasets that do not carry them.

Column glossary
---------------
image_id        : str  - path stem relative to image_root. The loader appends
                         `.npy` (cached) or `.dcm` (raw).
label           : int  - 0 = benign, 1 = malignant.
patient_id      : str  - for patient-level leakage checks across splits.
dataset         : str  - source identifier, e.g. "cbis_ddsm", "inbreast".
birads_density  : int  - BI-RADS breast-density category 1-4.
birads_assessment: int - BI-RADS assessment category 1-6 (INbreast primary).
roi_mask_id     : str  - path stem of the binary lesion-mask file. Same
                         root convention as image_id.
pathology       : str  - raw string label before binary collapse (CBIS-DDSM).
lesion_type     : str  - "mass" or "calcification" (CBIS-DDSM).
subtlety        : int  - radiologist subtlety rating 1-5 (CBIS-DDSM).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

REQUIRED: tuple[str, ...] = ("image_id", "label")

# Maps column name -> Python type used for coercion.
# Int64 (capital I) is pandas' nullable integer - preserves NaN for optional
# integer columns rather than forcing a float cast.
SCHEMA: dict[str, str] = {
    "image_id": "str",
    "label": "int",
    "patient_id": "str",
    "dataset": "str",
    "birads_density": "Int64",
    "birads_assessment": "Int64",
    "roi_mask_id": "str",
    "pathology": "str",
    "lesion_type": "str",
    "subtlety": "Int64",
}


def validate(df: pd.DataFrame, source: str = "") -> None:
    """Raise `ValueError` if the required columns are missing or labels invalid.

    Call once at manifest load time so problems surface immediately rather than mid-epoch.
    """
    tag = f" (from {source})" if source else ""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest{tag} is missing required columns: {missing}")
    valid_labels = {0, 1}
    actual = set(df["label"].dropna().unique())
    if not actual.issubset(valid_labels):
        raise ValueError(
            f"Manifest{tag} label column must contain only 0 and 1, "
            f"got unexpected values: {actual - valid_labels}"
        )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dtype coercion for known columns, leaving unknown ones alone."""
    df = df.copy()
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            continue
        if dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == "str":
            df[col] = df[col].where(df[col].notna(), other=pd.NA).astype(str)
            df[col] = df[col].replace("nan", pd.NA).replace("None", pd.NA)
    return df


def read(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV, validate it, and coerce column types.

    Use this instead of `pd.read_csv` everywhere a manifest is loaded so
    schema errors are caught at read time.
    """
    path = Path(path)
    df = pd.read_csv(path)
    validate(df, source=str(path))
    return _coerce(df)


def assert_patient_disjoint(frames: Mapping[str, pd.DataFrame]) -> None:
    """Reject patient leakage between any pair of named manifest frames."""
    patient_sets: dict[str, set[str]] = {}
    for split, frame in frames.items():
        if "patient_id" not in frame.columns:
            raise ValueError(f"{split} manifest is missing required patient_id column")
        if frame["patient_id"].isna().any():
            raise ValueError(f"{split} manifest contains missing patient_id values")
        patient_sets[split] = set(frame["patient_id"].astype(str))

    names = list(patient_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Patient leakage between {left} and {right} manifests: "
                    f"{len(overlap)} overlapping patient(s); examples: {examples}"
                )


def read_split_frames(
    splits_dir: str | Path,
    names: Sequence[str] = ("train", "val", "test"),
) -> dict[str, pd.DataFrame]:
    """Read named split manifests from one directory using the shared schema."""
    splits_dir = Path(splits_dir)
    frames: dict[str, pd.DataFrame] = {}
    for split in names:
        path = splits_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required split manifest not found: {path}")
        frame = read(path)
        if "patient_id" not in frame.columns:
            raise ValueError(f"{path} is missing required patient_id column")
        frames[split] = frame
    return frames


def validate_split_paths(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> None:
    """Preflight three configured manifests and reject patient overlap."""
    paths = {
        "train": Path(train_csv),
        "val": Path(val_csv),
        "test": Path(test_csv),
    }
    frames = {name: read(path) for name, path in paths.items()}
    assert_patient_disjoint(frames)
