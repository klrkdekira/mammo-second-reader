"""Shared manifest schema for all mammography datasets.

Every CSV that drives MammogramDataset must conform to this schema.
Required columns are `image_id` and `label`; all others are optional
and may be absent or NaN for datasets that do not carry them.

Column glossary
---------------
image_id        : str  - path stem relative to image_root; loader appends
                         `.npy` (cached) or `.dcm` (raw).
label           : int  - 0 = benign, 1 = malignant.
patient_id      : str  - for patient-level leakage checks across splits.
dataset         : str  - source identifier, e.g. "cbis_ddsm", "inbreast".
birads_density  : int  - BI-RADS breast-density category 1-4.
birads_assessment: int - BI-RADS assessment category 1-6 (INbreast primary).
roi_mask_id     : str  - path stem of the binary lesion-mask file; same
                         root convention as image_id.
pathology       : str  - raw string label before binary collapse (CBIS-DDSM).
mass_or_calc    : str  - "mass" or "calc" (CBIS-DDSM).
subtlety        : int  - radiologist subtlety rating 1-5 (CBIS-DDSM).
"""

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
    "mass_or_calc": "str",
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
