"""Build a small upload archive for testing the web fine-tuning flow.

The upload format is flat, while CBIS-DDSM image IDs are nested paths. Selected
rows receive unique flat IDs, with the original IDs retained for traceability.
Both splits are required to contain benign and malignant images.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import click
import pandas as pd

from src.config import setup_logging
from src.data.dicom_to_png import _find_dicom
from src.web.archive import MAX_ARCHIVE_BYTES, MAX_EXTRACTED_BYTES

LOGGER = logging.getLogger(__name__)

CARRY_COLUMNS = ("patient_id", "birads_density", "lesion_type", "pathology")


def _stratified_sample(
    frame: pd.DataFrame, n: int, seed: int, split: str
) -> pd.DataFrame:
    """Take `n` rows with both classes represented as evenly as available."""
    if n < 2:
        raise ValueError(f"{split}: need at least 2 images to cover both classes.")
    positives = frame[frame["label"] == 1]
    negatives = frame[frame["label"] == 0]
    if positives.empty or negatives.empty:
        raise ValueError(
            f"{split}: the source split has only one class, so no usable "
            "fine-tuning fixture can be built from it."
        )
    n_pos = min(len(positives), max(1, n // 2))
    n_neg = min(len(negatives), max(1, n - n_pos))
    chosen = pd.concat(
        [
            positives.sample(n=n_pos, random_state=seed),
            negatives.sample(n=n_neg, random_state=seed + 1),
        ]
    )
    return chosen.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def _source_file(
    image_id: str, cache_dir: Path, raw_root: Path, source: str
) -> Path | None:
    if source == "npy":
        candidate = cache_dir / f"{image_id}.npy"
        return candidate if candidate.is_file() else None
    return _find_dicom(raw_root, image_id)


def build_split(
    frame: pd.DataFrame,
    *,
    split: str,
    n: int,
    seed: int,
    cache_dir: Path,
    raw_root: Path,
    source: str,
    staging: Path,
) -> pd.DataFrame:
    """Copy one split's images into `staging` under flat names and return its manifest."""
    selected = _stratified_sample(frame, n, seed, split)
    suffix = ".npy" if source == "npy" else ".dcm"
    rows: list[dict[str, object]] = []
    for index, record in enumerate(selected.to_dict(orient="records")):
        original = str(record["image_id"])
        found = _source_file(original, cache_dir, raw_root, source)
        if found is None:
            LOGGER.warning("%s: no %s file for %s", split, source, original)
            continue
        flat_id = f"{split}_{index:04d}"
        shutil.copyfile(found, staging / f"{flat_id}{suffix}")
        row: dict[str, object] = {
            "image_id": flat_id,
            "label": int(record["label"]),
            "source_image_id": original,
        }
        for column in CARRY_COLUMNS:
            if column in selected.columns:
                row[column] = record.get(column)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(
            f"{split}: no source images were found. Check --cache-dir/--raw-root "
            "and that the dataset is present on this machine."
        )
    counts = manifest["label"].value_counts()
    if counts.get(1, 0) == 0 or counts.get(0, 0) == 0:
        raise ValueError(
            f"{split}: ended up single-class ({counts.to_dict()}) because some "
            f"source images were missing. Increase --n-{split} or repair the cache."
        )
    return manifest


def write_archive(
    output: Path,
    *,
    splits_dir: Path,
    cache_dir: Path,
    raw_root: Path,
    n_train: int,
    n_val: int,
    seed: int,
    source: str,
) -> dict[str, object]:
    """Assemble the fine-tuning fixture archive and report what went into it."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    train_source = pd.read_csv(Path(splits_dir) / "train.csv")
    val_source = pd.read_csv(Path(splits_dir) / "val.csv")

    with tempfile.TemporaryDirectory(prefix="mammo-finetune-fixture-") as tmp:
        staging = Path(tmp)
        train = build_split(
            train_source,
            split="train",
            n=n_train,
            seed=seed,
            cache_dir=cache_dir,
            raw_root=raw_root,
            source=source,
            staging=staging,
        )
        val = build_split(
            val_source,
            split="val",
            n=n_val,
            seed=seed + 100,
            cache_dir=cache_dir,
            raw_root=raw_root,
            source=source,
            staging=staging,
        )
        train.to_csv(staging / "train.csv", index=False)
        val.to_csv(staging / "val.csv", index=False)

        members = sorted(path for path in staging.iterdir() if path.is_file())
        uncompressed = sum(path.stat().st_size for path in members)
        if uncompressed > MAX_EXTRACTED_BYTES:
            raise ValueError(
                f"Fixture would decompress to {uncompressed / 1e6:.0f} MB, over the "
                f"{MAX_EXTRACTED_BYTES / 1e6:.0f} MB web limit. Lower --n-train/--n-val."
            )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, arcname=path.name)

    size = output.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Fixture is {size / 1e6:.0f} MB, over the "
            f"{MAX_ARCHIVE_BYTES / 1e6:.0f} MB upload limit. Lower --n-train/--n-val."
        )
    return {
        "archive": str(output),
        "archive_bytes": size,
        "uncompressed_bytes": uncompressed,
        "n_train": len(train),
        "n_val": len(val),
        "train_labels": train["label"].value_counts().to_dict(),
        "val_labels": val["label"].value_counts().to_dict(),
        "source": source,
    }


def verify_archive(output: Path) -> dict[str, int]:
    """Unpack the fixture exactly as the web app does, to prove it is loadable."""
    from src.web.finetune import materialise_workdir

    with tempfile.TemporaryDirectory(prefix="mammo-finetune-verify-") as tmp:
        workdir = materialise_workdir(str(output), Path(tmp) / "work")
        train = pd.read_csv(workdir / "train.csv")
        val = pd.read_csv(workdir / "val.csv")
        processed = workdir / "processed"
        missing = [
            image_id
            for image_id in pd.concat([train, val])["image_id"].astype(str)
            if not any(
                (processed / f"{image_id}{suffix}").is_file()
                for suffix in (".npy", ".dcm")
            )
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} manifest row(s) have no extracted image, "
                f"starting with {missing[0]!r}."
            )
        return {"train_rows": len(train), "val_rows": len(val)}


def main(
    output: Path,
    splits_dir: Path,
    cache_dir: Path,
    raw_root: Path,
    n_train: int,
    n_val: int,
    seed: int,
    source: str,
    verify: bool,
) -> None:
    setup_logging()
    summary = write_archive(
        output,
        splits_dir=splits_dir,
        cache_dir=cache_dir,
        raw_root=raw_root,
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        source=source,
    )
    LOGGER.info(
        "Wrote %s (%.1f MB): %d train %s, %d val %s, source=%s",
        summary["archive"],
        cast(int, summary["archive_bytes"]) / 1e6,
        summary["n_train"],
        summary["train_labels"],
        summary["n_val"],
        summary["val_labels"],
        summary["source"],
    )
    if verify:
        checked = verify_archive(Path(output))
        LOGGER.info(
            "Verified through materialise_workdir: %d train and %d val rows resolve.",
            checked["train_rows"],
            checked["val_rows"],
        )
    LOGGER.info(
        "Upload this ZIP in the Fine-tune tab with a base checkpoint from models/. "
        "Use models/resnet50_imagenet.pt with model name 'resnet50_imagenet': "
        "freezing its backbone leaves ~0.5M trainable parameters, so an epoch is "
        "quick even on CPU. models/baseline.pt is faster still (~0.1M, no backbone "
        "to freeze). Avoid vgg16_imagenet for a demo: its classifier head alone is "
        "~120M parameters, so 'freeze backbone' does not make it cheap."
    )


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("results/fixtures/finetune-testset.zip"),
    show_default=True,
)
@click.option(
    "--splits-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/cbis-ddsm/training"),
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/cbis_ddsm"),
    show_default=True,
    help="Directory holding cached .npy arrays (used when --source npy).",
)
@click.option(
    "--raw-root",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/cbis_ddsm"),
    show_default=True,
    help="Raw DICOM root (used when --source dcm).",
)
@click.option("--n-train", type=click.IntRange(min=2), default=24, show_default=True)
@click.option("--n-val", type=click.IntRange(min=2), default=12, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--source",
    type=click.Choice(["npy", "dcm"]),
    default="npy",
    show_default=True,
    help="npy is fast; dcm also exercises the de-identification path on upload.",
)
@click.option("--verify/--no-verify", default=True, show_default=True)
def cli(
    output: Path,
    splits_dir: Path,
    cache_dir: Path,
    raw_root: Path,
    n_train: int,
    n_val: int,
    seed: int,
    source: str,
    verify: bool,
) -> None:
    try:
        main(
            output,
            splits_dir,
            cache_dir,
            raw_root,
            n_train,
            n_val,
            seed,
            source,
            verify,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
