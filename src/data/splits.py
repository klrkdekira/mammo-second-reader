"""Build train, val, and test CSVs from the official CBIS-DDSM partition."""

import logging
import os
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import setup_logging
from src.data.cbis_ddsm import DICOMPathResolver
from src.data.manifest import assert_patient_disjoint

LOGGER = logging.getLogger(__name__)

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
SPLIT_NAMES = ("train", "val", "test")
EXCLUSION_COLUMNS = (
    "patient_id",
    "development_split",
    "n_train_images",
    "n_val_images",
    "n_test_images",
    "reason",
)


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
    """Map an official CBIS-DDSM CSV to the project schema."""
    df = pd.read_csv(raw_csv)
    df = df.rename(
        columns={
            "breast_density": "birads_density",  # mass CSVs use underscore
            "breast density": "birads_density",  # calc CSVs use space
            "abnormality type": "lesion_type",
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
        "lesion_type",
        "subtlety",
        "roi_mask_id",
    ]
    return df[keep]


def collapse_to_image_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse abnormality rows to one row per image."""
    if df.empty:
        return df
    n_before = len(df)
    rows = []
    for _, g in df.groupby("image_id", sort=False):
        malignant = int(g["label"].max())
        rep = g[g["label"] == malignant].iloc[0].to_dict()
        rep["label"] = malignant
        types = sorted(g["lesion_type"].dropna().astype(str).str.lower().unique())
        rep["lesion_type"] = types[0] if len(types) == 1 else "mixed"
        rows.append(rep)
    out = pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)
    collapsed = n_before - len(out)
    if collapsed:
        LOGGER.info(
            "Collapsed %d duplicate abnormality rows to image level (%d rows -> %d images)",
            collapsed,
            n_before,
            len(out),
        )
    return out


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


def test_overlap_exclusion_ledger(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Record development patients removed by test-set precedence."""
    patients = {
        split: set(frame["patient_id"].astype(str)) for split, frame in frames.items()
    }
    train_val = patients["train"] & patients["val"]
    if train_val:
        examples = ", ".join(sorted(train_val)[:5])
        raise ValueError("Patient leakage between train and val manifests: " + examples)

    rows: list[dict[str, object]] = []
    test_patients = patients["test"]
    for development_split in ("train", "val"):
        for patient_id in sorted(patients[development_split] & test_patients):
            rows.append(
                {
                    "patient_id": patient_id,
                    "development_split": development_split,
                    "n_train_images": int(
                        (frames["train"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "n_val_images": int(
                        (frames["val"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "n_test_images": int(
                        (frames["test"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "reason": "patient_id_also_present_in_locked_test",
                }
            )
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


def quarantine_test_overlaps(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Remove test-overlap patients from development and validate the result."""
    ledger = test_overlap_exclusion_ledger(frames)
    excluded = set(ledger["patient_id"].astype(str))
    clean: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        frame = frames[split]
        if split in ("train", "val") and excluded:
            frame = frame[~frame["patient_id"].astype(str).isin(excluded)]
        clean[split] = frame.reset_index(drop=True)
    assert_patient_disjoint(clean)
    return clean, ledger


def write_split_bundle(
    frames: dict[str, pd.DataFrame],
    ledger: pd.DataFrame,
    splits_dir: Path,
) -> dict[str, Path]:
    """Atomically write one validated canonical split bundle and its ledger."""
    assert_patient_disjoint(frames)
    splits_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    payloads: list[tuple[str, pd.DataFrame]] = [
        *((split, frames[split]) for split in SPLIT_NAMES),
        ("excluded-test-overlap-patients", ledger),
    ]
    for name, frame in payloads:
        path = splits_dir / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
        outputs[name] = path
        if name in SPLIT_NAMES:
            labels = frame["label"].value_counts().to_dict()
            LOGGER.info(
                "Wrote %s: rows=%d patients=%d benign=%d malignant=%d path=%s",
                name,
                len(frame),
                frame["patient_id"].nunique(),
                labels.get(0, 0),
                labels.get(1, 0),
                path,
            )
        else:
            LOGGER.info(
                "Wrote %s: rows=%d patients=%d path=%s",
                name,
                len(frame),
                frame["patient_id"].nunique(),
                path,
            )
    return outputs


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

    train_full = collapse_to_image_level(pd.concat(train_parts, ignore_index=True))
    test_df = collapse_to_image_level(pd.concat(test_parts, ignore_index=True))
    train_df, val_df = carve_validation(train_full, val_frac, seed)
    clean, ledger = quarantine_test_overlaps(
        {"train": train_df, "val": val_df, "test": test_df}
    )
    LOGGER.warning(
        "Locked-test precedence quarantined %d patient(s) from development",
        ledger["patient_id"].nunique(),
    )
    write_split_bundle(clean, ledger, splits_dir)


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
    default=Path("manifests/cbis-ddsm"),
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
