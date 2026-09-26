"""Build INbreast upload archives for testing the web evaluation and fine-tuning flows.

Two ZIPs are written from the locked INbreast manifest:

* an evaluation archive holding one `manifest.csv` and its images, for the
  Batch Evaluation tab
* a fine-tuning archive holding patient-disjoint `train.csv` and `val.csv`
  and their images, for the Fine-tune tab

INbreast is the pre-registered cold external test set. These archives exist to
exercise the web app only. Scores from them must not be reported, and a model
fine-tuned on them is no longer cold for INbreast.

INbreast image IDs are already flat DICOM stems, so no renaming is needed.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import click
import numpy as np
import pandas as pd

from src.config import setup_logging
from src.data import manifest as _manifest
from src.data.inbreast import lesion_present
from src.data.make_finetune_archive import _stratified_sample, verify_archive
from src.web.archive import MAX_ARCHIVE_BYTES, MAX_EXTRACTED_BYTES

LOGGER = logging.getLogger(__name__)

CARRY_COLUMNS = (
    "patient_id",
    "birads_density",
    "birads_assessment",
    "lesion_type",
    "view",
    "laterality",
)
EVAL_MANIFEST = "manifest.csv"


def _source_file(image_id: str, cache_dir: Path, raw_root: Path, source: str) -> Path:
    suffix = ".npy" if source == "npy" else ".dcm"
    root = cache_dir if source == "npy" else raw_root
    return root / f"{image_id}{suffix}"


def build_missing_cache(
    frame: pd.DataFrame, *, cache_dir: Path, raw_root: Path, image_size: int
) -> int:
    """Cache any missing arrays from the raw DICOMs, as `dicom_to_png` would.

    `dicom_to_png` expects train, val and test manifests, and INbreast has only
    a test manifest, so the cache is filled here instead.
    """
    from src.data.preprocessing import preprocess

    built = 0
    for image_id in frame["image_id"].astype(str):
        target = cache_dir / f"{image_id}.npy"
        dcm = raw_root / f"{image_id}.dcm"
        if target.is_file() or not dcm.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, preprocess(dcm, image_size=image_size))
        built += 1
    return built


def _keep_available(
    frame: pd.DataFrame, *, cache_dir: Path, raw_root: Path, source: str, name: str
) -> pd.DataFrame:
    """Drop rows whose image is missing, and fail if either class disappears."""
    present = (
        frame["image_id"]
        .astype(str)
        .map(
            lambda image_id: _source_file(
                image_id, cache_dir, raw_root, source
            ).is_file()
        )
    )
    missing = int((~present).sum())
    if missing:
        LOGGER.warning("%s: %d image(s) have no %s file", name, missing, source)
    kept = frame.loc[present].reset_index(drop=True)
    counts = kept["label"].value_counts()
    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise ValueError(
            f"{name}: no usable images for one class ({counts.to_dict()}). Check "
            "--cache-dir/--raw-root and that the INbreast cache has been built."
        )
    return kept


def _manifest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["image_id", "label", *(c for c in CARRY_COLUMNS if c in frame.columns)]
    return frame.loc[:, columns].reset_index(drop=True)


def _write_zip(
    output: Path,
    manifests: dict[str, pd.DataFrame],
    *,
    cache_dir: Path,
    raw_root: Path,
    source: str,
) -> tuple[int, int]:
    """Zip the manifests and their images, enforcing the web upload limits."""
    output.parent.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(
        {str(i) for frame in manifests.values() for i in frame["image_id"]}
    )
    images = [_source_file(i, cache_dir, raw_root, source) for i in image_ids]
    uncompressed = sum(path.stat().st_size for path in images)
    if uncompressed > MAX_EXTRACTED_BYTES:
        raise ValueError(
            f"{output.name} would decompress to {uncompressed / 1e6:.0f} MB, over "
            f"the {MAX_EXTRACTED_BYTES / 1e6:.0f} MB web limit. Use --source npy or "
            "lower the image counts."
        )
    with tempfile.TemporaryDirectory(prefix="mammo-inbreast-fixture-") as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, frame in manifests.items():
                csv_path = staging / name
                frame.to_csv(csv_path, index=False)
                archive.write(csv_path, arcname=name)
            for path in images:
                archive.write(path, arcname=path.name)
    size = output.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        output.unlink()
        raise ValueError(
            f"{output.name} is {size / 1e6:.0f} MB, over the "
            f"{MAX_ARCHIVE_BYTES / 1e6:.0f} MB upload limit. Lower the image counts."
        )
    return size, uncompressed


def patient_split(
    frame: pd.DataFrame, *, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by patient, stratified on whether a patient has any malignant image."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    patients = frame.groupby(frame["patient_id"].astype(str))["label"].max()
    rng = np.random.default_rng(seed)
    val_patients: set[str] = set()
    for label in (0, 1):
        group = patients.index[patients == label].to_numpy()
        if len(group) < 2:
            raise ValueError(
                f"Need at least 2 patients with patient-level label {label} to "
                "put that class in both splits."
            )
        rng.shuffle(group)
        n_val = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        val_patients.update(group[:n_val])
    in_val = frame["patient_id"].astype(str).isin(val_patients)
    train = frame.loc[~in_val].reset_index(drop=True)
    val = frame.loc[in_val].reset_index(drop=True)
    for name, split in (("train", train), ("val", val)):
        if split["label"].nunique() < 2:
            raise ValueError(f"{name}: patient split left a single class.")
    _manifest.assert_patient_disjoint({"train": train, "val": val})
    return train, val


def _limit(frame: pd.DataFrame, n: int | None, seed: int, name: str) -> pd.DataFrame:
    if n is None or n >= len(frame):
        return frame
    return _stratified_sample(frame, n, seed, name)


def write_eval_archive(
    output: Path,
    frame: pd.DataFrame,
    *,
    cache_dir: Path,
    raw_root: Path,
    source: str,
    n_images: int | None,
    seed: int,
) -> dict[str, object]:
    """Write the Batch Evaluation archive and report what went into it."""
    frame = _keep_available(
        frame, cache_dir=cache_dir, raw_root=raw_root, source=source, name="eval"
    )
    selected = _manifest_rows(_limit(frame, n_images, seed, "eval"))
    size, uncompressed = _write_zip(
        output,
        {EVAL_MANIFEST: selected},
        cache_dir=cache_dir,
        raw_root=raw_root,
        source=source,
    )
    return {
        "archive": str(output),
        "archive_bytes": size,
        "uncompressed_bytes": uncompressed,
        "n": len(selected),
        "labels": selected["label"].value_counts().to_dict(),
        "patients": int(selected["patient_id"].nunique()),
    }


def write_finetune_archive(
    output: Path,
    frame: pd.DataFrame,
    *,
    cache_dir: Path,
    raw_root: Path,
    source: str,
    val_fraction: float,
    n_train: int | None,
    n_val: int | None,
    seed: int,
) -> dict[str, object]:
    """Write the Fine-tune archive with patient-disjoint splits."""
    frame = _keep_available(
        frame, cache_dir=cache_dir, raw_root=raw_root, source=source, name="finetune"
    )
    train, val = patient_split(frame, val_fraction=val_fraction, seed=seed)
    train = _manifest_rows(_limit(train, n_train, seed, "train"))
    val = _manifest_rows(_limit(val, n_val, seed + 100, "val"))
    size, uncompressed = _write_zip(
        output,
        {"train.csv": train, "val.csv": val},
        cache_dir=cache_dir,
        raw_root=raw_root,
        source=source,
    )
    return {
        "archive": str(output),
        "archive_bytes": size,
        "uncompressed_bytes": uncompressed,
        "n_train": len(train),
        "n_val": len(val),
        "train_labels": train["label"].value_counts().to_dict(),
        "val_labels": val["label"].value_counts().to_dict(),
        "train_patients": int(train["patient_id"].nunique()),
        "val_patients": int(val["patient_id"].nunique()),
    }


def verify_eval_archive(output: Path) -> int:
    """Validate an evaluation archive through the web loader."""
    from src.web.evaluation import _extract_batch

    with tempfile.TemporaryDirectory(prefix="mammo-inbreast-verify-") as tmp:
        workdir = Path(tmp)
        manifest_csv = _extract_batch(str(output), workdir)
        frame = _manifest.read(manifest_csv)
        missing = [
            image_id
            for image_id in frame["image_id"].astype(str)
            if not any(
                (workdir / f"{image_id}{suffix}").is_file()
                for suffix in (".npy", ".dcm")
            )
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} manifest row(s) have no extracted image, "
                f"starting with {missing[0]!r}."
            )
        return len(frame)


def main(
    manifest_dir: Path,
    subset: str,
    cache_dir: Path,
    raw_root: Path,
    source: str,
    eval_output: Path,
    finetune_output: Path,
    n_eval: int | None,
    n_train: int | None,
    n_val: int | None,
    val_fraction: float,
    seed: int,
    image_size: int,
    verify: bool,
) -> None:
    setup_logging()
    frame = _manifest.read(manifest_dir / "test.csv")
    if subset == "lesion_present":
        frame = lesion_present(frame)
    LOGGER.info("INbreast %s subset: %d images", subset, len(frame))
    if source == "npy":
        built = build_missing_cache(
            frame, cache_dir=cache_dir, raw_root=raw_root, image_size=image_size
        )
        if built:
            LOGGER.info("Cached %d missing array(s) into %s", built, cache_dir)

    evaluation = write_eval_archive(
        eval_output,
        frame,
        cache_dir=cache_dir,
        raw_root=raw_root,
        source=source,
        n_images=n_eval,
        seed=seed,
    )
    LOGGER.info(
        "Wrote %s (%.1f MB): %d images %s from %d patients",
        evaluation["archive"],
        cast(int, evaluation["archive_bytes"]) / 1e6,
        evaluation["n"],
        evaluation["labels"],
        evaluation["patients"],
    )

    finetune = write_finetune_archive(
        finetune_output,
        frame,
        cache_dir=cache_dir,
        raw_root=raw_root,
        source=source,
        val_fraction=val_fraction,
        n_train=n_train,
        n_val=n_val,
        seed=seed,
    )
    LOGGER.info(
        "Wrote %s (%.1f MB): %d train %s from %d patients, %d val %s from %d patients",
        finetune["archive"],
        cast(int, finetune["archive_bytes"]) / 1e6,
        finetune["n_train"],
        finetune["train_labels"],
        finetune["train_patients"],
        finetune["n_val"],
        finetune["val_labels"],
        finetune["val_patients"],
    )

    if verify:
        n = verify_eval_archive(eval_output)
        LOGGER.info("Verified evaluation archive: %d rows resolve.", n)
        checked = verify_archive(finetune_output)
        LOGGER.info(
            "Verified fine-tuning archive: %d train and %d val rows resolve.",
            checked["train_rows"],
            checked["val_rows"],
        )
    LOGGER.info(
        "These archives are for app testing only. INbreast is the cold external "
        "set, so do not report their scores or evaluate a model fine-tuned on them "
        "as an external result."
    )


@click.command()
@click.option(
    "--manifest-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/inbreast/manifest"),
    show_default=True,
    help="Directory holding the locked INbreast test.csv.",
)
@click.option(
    "--subset",
    type=click.Choice(["full", "lesion_present"]),
    default="full",
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path("data/inbreast/cache_448"),
    show_default=True,
    help="Directory holding cached .npy arrays (used when --source npy).",
)
@click.option(
    "--raw-root",
    type=click.Path(path_type=Path),
    default=Path("data/inbreast/AllDICOMs"),
    show_default=True,
    help="Raw DICOM directory. Used for --source dcm and to fill a missing npy cache.",
)
@click.option(
    "--source",
    type=click.Choice(["npy", "dcm"]),
    default="npy",
    show_default=True,
    help="npy fits the full set. dcm exercises de-identification but each "
    "INbreast DICOM is large, so set small --n-eval/--n-train/--n-val.",
)
@click.option(
    "--eval-output",
    type=click.Path(path_type=Path),
    default=Path("results/fixtures/inbreast-eval.zip"),
    show_default=True,
)
@click.option(
    "--finetune-output",
    type=click.Path(path_type=Path),
    default=Path("results/fixtures/inbreast-finetune.zip"),
    show_default=True,
)
@click.option(
    "--n-eval", type=click.IntRange(min=2), default=None, help="Default: all images."
)
@click.option(
    "--n-train", type=click.IntRange(min=2), default=None, help="Default: all."
)
@click.option("--n-val", type=click.IntRange(min=2), default=None, help="Default: all.")
@click.option(
    "--val-fraction",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    default=0.25,
    show_default=True,
    help="Share of patients per class placed in val.csv.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--image-size",
    type=int,
    default=448,
    show_default=True,
    help="Size for arrays cached from DICOM when the npy cache is incomplete.",
)
@click.option("--verify/--no-verify", default=True, show_default=True)
def cli(
    manifest_dir: Path,
    subset: str,
    cache_dir: Path,
    raw_root: Path,
    source: str,
    eval_output: Path,
    finetune_output: Path,
    n_eval: int | None,
    n_train: int | None,
    n_val: int | None,
    val_fraction: float,
    seed: int,
    image_size: int,
    verify: bool,
) -> None:
    try:
        main(
            manifest_dir,
            subset,
            cache_dir,
            raw_root,
            source,
            eval_output,
            finetune_output,
            n_eval,
            n_train,
            n_val,
            val_fraction,
            seed,
            image_size,
            verify,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
