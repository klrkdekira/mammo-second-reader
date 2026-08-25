"""Pre-convert ROI lesion-mask DICOMs into compact .npy tensors.

Cached masks are cropped to the same breast bounding box as their source
image and resized to the cache resolution, matching the geometry of the
cached images. `MammogramDataset.load_roi` can then skip the expensive
per-call DICOM re-segmentation.

The source image is segmented once per image, not once per mask.
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.dicom_to_png import _cache_has_shape, _find_dicom
from src.data.manifest import assert_patient_disjoint, read_split_frames
from src.data.preprocessing import breast_bbox, breast_mask, load_dicom

LOGGER = logging.getLogger(__name__)

_MISSING = ("", "nan", "<NA>", "None")


def _collect_pairs(splits_dir: Path) -> dict[str, set[str]]:
    """Map each image_id to the set of roi_mask_ids referencing it."""
    pairs: dict[str, set[str]] = {}
    frames = read_split_frames(splits_dir)
    assert_patient_disjoint(frames)
    for df in frames.values():
        if "roi_mask_id" not in df.columns:
            continue
        for image_id, rid in zip(df["image_id"].astype(str), df["roi_mask_id"]):
            if pd.isna(rid) or str(rid) in _MISSING:
                continue
            pairs.setdefault(image_id, set()).add(str(rid))
    return pairs


def main(splits_dir: Path, raw_root: Path, out_dir: Path, image_size: int) -> None:
    import cv2
    import pydicom

    setup_logging()
    out_dir = Path(out_dir)
    pairs = _collect_pairs(splits_dir)
    n_masks = sum(len(v) for v in pairs.values())
    LOGGER.info(
        "Caching %d ROI masks for %d images to %s", n_masks, len(pairs), out_dir
    )

    skipped = 0
    for image_id, mask_ids in tqdm(sorted(pairs.items())):
        todo = [
            rid
            for rid in sorted(mask_ids)
            if not _cache_has_shape(out_dir / f"{rid}.npy", image_size)
        ]
        if not todo:
            continue
        img_dcm = _find_dicom(raw_root, image_id)
        if img_dcm is None:
            LOGGER.warning("No image DICOM found for %s", image_id)
            skipped += len(todo)
            continue
        img = load_dicom(img_dcm)
        y0, y1, x0, x1 = breast_bbox(breast_mask(img))
        for rid in todo:
            mask_dcm = _find_dicom(raw_root, rid)
            if mask_dcm is None:
                LOGGER.warning("No mask DICOM found for %s", rid)
                skipped += 1
                continue
            mask = pydicom.dcmread(str(mask_dcm)).pixel_array
            mask = (np.asarray(mask) > 0).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask.any(axis=0).astype(np.uint8)
            if mask.shape != img.shape:
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask[y0:y1, x0:x1]
            mask = cv2.resize(
                mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
            )
            out_path = out_dir / f"{rid}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, mask)
    LOGGER.info("Done. Skipped %d masks with no matching DICOM.", skipped)


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=Path("manifests/cbis-ddsm"),
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
def cli(splits_dir: Path, raw_root: Path, out_dir: Path, image_size: int) -> None:
    main(splits_dir, raw_root, out_dir, image_size)


if __name__ == "__main__":
    cli()
