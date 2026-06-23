"""Build train, val, and test CSVs from the official CBIS-DDSM partition."""

import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import setup_logging
from src.data.cbis_ddsm import DICOMPathResolver

LOGGER = logging.getLogger(__name__)

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}


def _collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["pathology"].map(LABEL_MAP)
    return df.dropna(subset=["label"]).astype({"label": int})


def _path_to_id(path: Path | None, dicom_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(dicom_dir).with_suffix("").as_posix()
    except ValueError:
        return None


def _build_dataframe(
    raw_csv: Path, dicom_dir: Path, resolver: DICOMPathResolver
) -> pd.DataFrame:
    """Read an official CBIS-DDSM CSV and return the project schema.

    Output columns:
    - image_id
    - patient_id
    - pathology
    - label
    - birads_density
    - mass_or_calc
    - subtlety
    - roi_mask_id

    image_id and roi_mask_id are dicom_dir-relative paths without extension.
    """
    df = pd.read_csv(raw_csv)
    df = df.rename(
        columns={
            "breast_density": "birads_density",  # mass CSVs use underscore
            "breast density": "birads_density",  # calc CSVs use space
            "abnormality type": "mass_or_calc",
        }
    )
    if "subtlety" not in df.columns:
        df["subtlety"] = pd.NA

    resolver.resolve_dataframe(df)

    df["image_id"] = df["full_image_path"].apply(lambda p: _path_to_id(p, dicom_dir))
    df["roi_mask_id"] = df["roi_mask_path"].apply(lambda p: _path_to_id(p, dicom_dir))

    missing = df["image_id"].isna().sum()
    if missing:
        LOGGER.warning(
            "Dropping %d rows with unresolved image paths from %s",
            missing,
            raw_csv.name,
        )
    df = df.dropna(subset=["image_id"])

    df = _collapse_labels(df)
    df["dataset"] = "cbis_ddsm"
    keep = [
        "image_id",
        "patient_id",
        "dataset",
        "pathology",
        "label",
        "birads_density",
        "mass_or_calc",
        "subtlety",
        "roi_mask_id",
    ]
    return df[keep]


def carve_validation(
    train_df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-disjoint, label-stratified val carve-out on the training fold."""
    n_splits = max(round(1 / val_frac), 2)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        sgkf.split(train_df, train_df["label"], groups=train_df["patient_id"])
    )
    return (
        train_df.iloc[train_idx].reset_index(drop=True),
        train_df.iloc[val_idx].reset_index(drop=True),
    )


def main(
    raw_dir: Path,
    dicom_dir: Path,
    splits_dir: Path,
    val_frac: float,
    seed: int,
) -> None:
    setup_logging()
    raw_dir = Path(raw_dir)
    dicom_dir = Path(dicom_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building DICOM path index from %s ...", dicom_dir)
    resolver = DICOMPathResolver(dicom_dir)

    train_parts, test_parts = [], []
    for kind in ("mass", "calc"):
        train_csv = raw_dir / f"{kind}_case_description_train_set.csv"
        test_csv = raw_dir / f"{kind}_case_description_test_set.csv"
        if not train_csv.exists() or not test_csv.exists():
            LOGGER.warning("Skipping %s: %s or %s missing.", kind, train_csv, test_csv)
            continue
        LOGGER.info("Processing %s ...", kind)
        train_parts.append(_build_dataframe(train_csv, dicom_dir, resolver))
        test_parts.append(_build_dataframe(test_csv, dicom_dir, resolver))

    if not train_parts:
        raise RuntimeError(
            f"No CBIS-DDSM CSVs found in {raw_dir}. Download the metadata "
            "from TCIA and place mass_case_description_*.csv plus "
            "calc_case_description_*.csv into the raw directory."
        )

    train_full = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df, val_df = carve_validation(train_full, val_frac, seed)

    for name, df in (
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test.csv", test_df),
    ):
        out = splits_dir / name
        df.to_csv(out, index=False)
        LOGGER.info("Wrote %d rows to %s", len(df), out)


@click.command()
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm"),
    show_default=True,
    help="Directory containing the official CBIS-DDSM CSV files.",
)
@click.option(
    "--dicom-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/cbis_ddsm"),
    show_default=True,
    help="Root of the raw DICOM tree.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/training"),
    show_default=True,
    help="Output directory for train/val/test CSVs.",
)
@click.option(
    "--val-frac",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of training fold reserved for validation.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for stratified split.",
)
def cli(
    raw_dir: Path, dicom_dir: Path, splits_dir: Path, val_frac: float, seed: int
) -> None:
    main(raw_dir, dicom_dir, splits_dir, val_frac, seed)


if __name__ == "__main__":
    cli()
