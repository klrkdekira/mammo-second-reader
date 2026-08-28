"""Rasterise INbreast OsiriX annotations into cached binary masks.

Contours are filled as polygons. Point annotations are drawn as fixed-radius
discs because the source data does not record their extent. Masks include all
annotated findings on an image, including benign findings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.dicom_to_png import _cache_has_shape, _find_dicom
from src.data.inbreast import read_rois, roi_mask_id
from src.data.preprocessing import breast_bbox, breast_mask, load_dicom

LOGGER = logging.getLogger(__name__)

POINT_RADIUS_PX = 7
MIN_POLYGON_POINTS = 3


def rasterise(
    rois, shape: tuple[int, int], *, point_radius: int = POINT_RADIUS_PX
) -> np.ndarray:
    """Rasterise ROIs into a binary ``uint8`` mask with shape ``(H, W)``."""
    import cv2

    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for roi in rois:
        points = np.asarray(roi.points, dtype=np.float64)
        if points.size == 0:
            continue
        xs = np.clip(np.rint(points[:, 0]), 0, width - 1).astype(np.int32)
        ys = np.clip(np.rint(points[:, 1]), 0, height - 1).astype(np.int32)
        if len(xs) >= MIN_POLYGON_POINTS:
            cv2.fillPoly(mask, [np.stack([xs, ys], axis=1)], 1)
        else:
            for x, y in zip(xs, ys):
                cv2.circle(mask, (int(x), int(y)), int(point_radius), 1, -1)
    return mask


def _manifest_rows(splits_dir: Path) -> pd.DataFrame:
    """Collect manifest rows that declare a rasterisable ROI mask."""
    frames = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = Path(splits_dir) / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError(f"No manifest CSV found in {splits_dir}")
    frame = pd.concat(frames, ignore_index=True)
    for column in ("image_id", "file_name", "roi_mask_id"):
        if column not in frame.columns:
            raise ValueError(f"INbreast manifest is missing the {column} column.")
    frame = frame[frame["roi_mask_id"].notna()]
    return frame.drop_duplicates(subset="image_id").reset_index(drop=True)


def main(
    splits_dir: Path,
    raw_root: Path,
    xml_dir: Path,
    out_dir: Path,
    image_size: int,
    point_radius: int,
) -> None:
    import cv2

    setup_logging()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _manifest_rows(splits_dir)
    LOGGER.info("Rasterising %d INbreast ROI masks to %s", len(rows), out_dir)

    written = skipped = empty = 0
    for row in tqdm(rows.to_dict(orient="records")):
        image_id = str(row["image_id"])
        file_name = str(row["file_name"])
        expected_id = roi_mask_id(file_name)
        if str(row["roi_mask_id"]) != expected_id:
            raise ValueError(
                f"Manifest roi_mask_id {row['roi_mask_id']!r} does not match the "
                f"ingest convention {expected_id!r} for {image_id}."
            )
        out_path = out_dir / f"{expected_id}.npy"
        if _cache_has_shape(out_path, image_size):
            continue
        xml_path = Path(xml_dir) / f"{file_name}.xml"
        if not xml_path.is_file():
            LOGGER.warning("No XML annotation for %s", file_name)
            skipped += 1
            continue
        image_dcm = _find_dicom(Path(raw_root), image_id)
        if image_dcm is None:
            LOGGER.warning("No image DICOM found for %s", image_id)
            skipped += 1
            continue

        image = load_dicom(image_dcm)
        mask = rasterise(read_rois(xml_path), image.shape, point_radius=point_radius)
        if not mask.any():
            LOGGER.warning("Annotation for %s rasterised to an empty mask", file_name)
            empty += 1
        y0, y1, x0, x1 = breast_bbox(breast_mask(image))
        mask = mask[y0:y1, x0:x1]
        mask = cv2.resize(
            mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
        )
        np.save(out_path, mask)
        written += 1

    LOGGER.info(
        "Done. Wrote %d masks, skipped %d with no annotation or image, "
        "%d rasterised empty.",
        written,
        skipped,
        empty,
    )


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/inbreast/manifest"),
    show_default=True,
)
@click.option(
    "--raw-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/inbreast/AllDICOMs"),
    show_default=True,
)
@click.option(
    "--xml-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/inbreast/AllXML"),
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/inbreast/cache_448"),
    show_default=True,
)
@click.option("--image-size", type=int, default=448, show_default=True)
@click.option(
    "--point-radius",
    type=click.IntRange(min=1),
    default=POINT_RADIUS_PX,
    show_default=True,
    help="Disc radius in full-resolution pixels for single-point calcifications.",
)
def cli(
    splits_dir: Path,
    raw_root: Path,
    xml_dir: Path,
    out_dir: Path,
    image_size: int,
    point_radius: int,
) -> None:
    try:
        main(splits_dir, raw_root, xml_dir, out_dir, image_size, point_radius)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
