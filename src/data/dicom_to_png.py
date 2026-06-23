"""Pre-convert raw DICOMs into compact .npy tensors."""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.preprocessing import preprocess

LOGGER = logging.getLogger(__name__)


def _find_dicom(raw_root: Path, image_id: str) -> Path | None:
    # image_id is already a relative path from raw_root (no extension)
    direct = raw_root / f"{image_id}.dcm"
    if direct.exists():
        return direct
    # Fallback: search by stem in case the directory nesting changed
    stem = Path(image_id).name
    hits = list(raw_root.rglob(f"{stem}.dcm"))
    return hits[0] if hits else None


def main(
    splits_dir: Path,
    raw_root: Path,
    out_dir: Path,
    image_size: int,
    use_clahe: bool,
) -> None:
    setup_logging()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for name in ("train.csv", "val.csv", "test.csv"):
        path = splits_dir / name
        if not path.exists():
            continue
        seen.update(pd.read_csv(path)["image_id"].astype(str).tolist())

    LOGGER.info("Caching %d images to %s", len(seen), out_dir)
    skipped = 0
    for image_id in tqdm(sorted(seen)):
        out_path = out_dir / f"{image_id}.npy"
        if out_path.exists():
            continue
        dcm = _find_dicom(raw_root, image_id)
        if dcm is None:
            LOGGER.warning("No DICOM found for %s", image_id)
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = preprocess(dcm, image_size=image_size, use_clahe=use_clahe)
        np.save(out_path, arr)
    LOGGER.info("Done. Skipped %d images with no matching DICOM.", skipped)


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/training"),
    show_default=True,
)
@click.option(
    "--raw-root",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/cbis_ddsm"),
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/cbis-ddsm/cbis_ddsm"),
    show_default=True,
)
@click.option("--image-size", type=int, default=224, show_default=True)
@click.option(
    "--no-clahe", is_flag=True, default=False, help="Skip CLAHE contrast enhancement."
)
def cli(
    splits_dir: Path, raw_root: Path, out_dir: Path, image_size: int, no_clahe: bool
) -> None:
    main(splits_dir, raw_root, out_dir, image_size, not no_clahe)


if __name__ == "__main__":
    cli()
