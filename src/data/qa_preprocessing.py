"""Dataset-wide QA sweep of the preprocessing pipeline.

Runs segmentation and artefact removal over every image in the split CSVs,
records mechanical health metrics, and flags outliers:

- breast-mask fraction outside a plausible range (segmentation collapsed or
  leaked over the frame),
- crop box nearly full-frame while the mask touches all four edges (frame
  leak) or suspiciously small,
- saturated pixels surviving near the image edge after artefact removal
  (film border remnant),
- a black stripe at the crop edge (crop box extends onto zeroed pixels),
- ROI lesion masks clipped by the breast crop box.

Writes qa_metrics.csv (every image), qa_flags.csv (flagged only), and
contact-sheet PNGs of the flagged images' final model inputs for eyeballing.
"""

import logging
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.dicom_to_png import _find_dicom
from src.data.manifest import assert_patient_disjoint, read_split_frames
from src.data.preprocessing import (
    artifact_mask,
    breast_bbox,
    load_dicom,
    preprocess_array,
    segment_breast,
)

LOGGER = logging.getLogger(__name__)

_MISSING = ("", "nan", "<NA>", "None")

MASK_FRAC_RANGE = (0.10, 0.90)
BBOX_FRAC_MIN = 0.15
SAT_EDGE_MAX_PX = 50
EDGE_ZONE_PX = 100
ROI_COVERAGE_MIN = 0.999
STRIPE_BAND_PX = 5


def _collect_images(splits_dir: Path) -> dict[str, set[str]]:
    """Map each image_id to the set of roi_mask_ids referencing it."""
    images: dict[str, set[str]] = {}
    frames = read_split_frames(splits_dir)
    assert_patient_disjoint(frames)
    for df in frames.values():
        rids = df.get("roi_mask_id", pd.Series([None] * len(df)))
        for image_id, rid in zip(df["image_id"].astype(str), rids):
            masks = images.setdefault(image_id, set())
            if not (pd.isna(rid) or str(rid) in _MISSING):
                masks.add(str(rid))
    return images


def _edges_touched(mask: np.ndarray) -> int:
    return sum(
        bool(side.any()) for side in (mask[0], mask[-1], mask[:, 0], mask[:, -1])
    )


def _black_stripe(clean: np.ndarray, breast: np.ndarray, box) -> bool:
    """True if a crop-edge band is mostly masked-in yet near-black."""
    y0, y1, x0, x1 = box
    c, m = clean[y0:y1, x0:x1], breast[y0:y1, x0:x1]
    b = STRIPE_BAND_PX
    for cs, ms in (
        (c[:b], m[:b]),
        (c[-b:], m[-b:]),
        (c[:, :b], m[:, :b]),
        (c[:, -b:], m[:, -b:]),
    ):
        inside = ms > 0
        if inside.mean() > 0.2 and float(cs[inside].mean()) < 0.02:
            return True
    return False


def _roi_coverage(mask_dcm: Path, shape: tuple[int, int], box) -> float:
    import pydicom

    roi_arr = np.asarray(pydicom.dcmread(str(mask_dcm)).pixel_array) > 0
    if roi_arr.shape != shape:
        roi_arr = (
            cv2.resize(
                roi_arr.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    total = int(roi_arr.sum())
    if total == 0:
        return 1.0
    y0, y1, x0, x1 = box
    return float(roi_arr[y0:y1, x0:x1].sum()) / total


def _qa_image(image_id: str, img_dcm: Path, mask_ids: set[str], raw_root: Path) -> dict:
    arr = load_dicom(img_dcm)
    h, w = arr.shape
    breast = segment_breast(arr)
    art = artifact_mask(arr, breast)
    clean = arr.copy()
    clean[art > 0] = 0.0
    box = breast_bbox(breast)
    y0, y1, x0, x1 = box

    sat = clean > 0.85
    ys, xs = np.where(sat)
    if len(ys):
        depth = np.minimum.reduce([ys, h - 1 - ys, xs, w - 1 - xs])
        sat_edge_px = int((depth < EDGE_ZONE_PX).sum())
    else:
        sat_edge_px = 0

    coverages = {}
    for rid in sorted(mask_ids):
        mask_dcm = _find_dicom(raw_root, rid)
        if mask_dcm is None:
            LOGGER.warning("No mask DICOM found for %s", rid)
            continue
        coverages[rid] = _roi_coverage(mask_dcm, arr.shape, box)

    row = {
        "image_id": image_id,
        "mask_frac": float(breast.mean()),
        "bbox_frac": (y1 - y0) * (x1 - x0) / (h * w),
        "edges_touched": _edges_touched(breast),
        "sat_edge_px": sat_edge_px,
        "black_stripe": _black_stripe(clean, breast, box),
        "roi_min_coverage": min(coverages.values()) if coverages else np.nan,
        "n_rois": len(coverages),
    }

    reasons = []
    if not MASK_FRAC_RANGE[0] <= row["mask_frac"] <= MASK_FRAC_RANGE[1]:
        reasons.append(f"mask_frac={row['mask_frac']:.2f}")
    if row["bbox_frac"] < BBOX_FRAC_MIN:
        reasons.append(f"bbox_frac={row['bbox_frac']:.2f}")
    if row["bbox_frac"] > 0.98 and row["edges_touched"] == 4:
        reasons.append("bbox full-frame, mask touches all edges")
    if sat_edge_px > SAT_EDGE_MAX_PX:
        reasons.append(f"sat_edge_px={sat_edge_px}")
    if row["black_stripe"]:
        reasons.append("black stripe at crop edge")
    clipped = {rid: c for rid, c in coverages.items() if c < ROI_COVERAGE_MIN}
    if clipped:
        worst = min(clipped, key=lambda k: clipped[k])
        reasons.append(f"ROI clipped: {worst}={clipped[worst]:.3f}")
    row["flags"] = "; ".join(reasons)
    return row


def _contact_sheets(
    flagged: list[str], raw_root: Path, out_dir: Path, thumb: int, cols: int = 6
) -> None:
    tiles = []
    for image_id in flagged:
        dcm = _find_dicom(raw_root, image_id)
        if dcm is None:
            continue
        tile = preprocess_array(load_dicom(dcm), image_size=thumb)
        tile = (np.clip(tile, 0, 1) * 255).astype(np.uint8)
        cv2.putText(
            tile,
            Path(image_id).name[-28:],
            (4, thumb - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            255,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    rows_per_sheet = cols * 5
    for s in range(0, len(tiles), rows_per_sheet):
        chunk = tiles[s : s + rows_per_sheet]
        while len(chunk) % cols:
            chunk.append(np.zeros((thumb, thumb), np.uint8))
        grid = np.vstack(
            [np.hstack(chunk[r : r + cols]) for r in range(0, len(chunk), cols)]
        )
        cv2.imwrite(str(out_dir / f"flagged_{s // rows_per_sheet:02d}.png"), grid)


def main(
    splits_dir: Path, raw_root: Path, out_dir: Path, thumb: int, limit: int
) -> None:
    setup_logging()
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _collect_images(splits_dir)
    items = sorted(images.items())
    if limit:
        items = items[:limit]
    LOGGER.info("QA sweep over %d images -> %s", len(items), out_dir)

    rows, skipped = [], 0
    for image_id, mask_ids in tqdm(items):
        img_dcm = _find_dicom(raw_root, image_id)
        if img_dcm is None:
            LOGGER.warning("No image DICOM found for %s", image_id)
            skipped += 1
            continue
        try:
            rows.append(_qa_image(image_id, img_dcm, mask_ids, raw_root))
        except Exception:
            LOGGER.exception("QA failed for %s", image_id)
            rows.append({"image_id": image_id, "flags": "processing error"})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qa_metrics.csv", index=False)
    flagged = df[df["flags"].astype(str).str.len() > 0]
    flagged.to_csv(out_dir / "qa_flags.csv", index=False)
    LOGGER.info(
        "Done: %d images, %d flagged, %d skipped. Reports in %s",
        len(df),
        len(flagged),
        skipped,
        out_dir,
    )
    if len(flagged):
        _contact_sheets(flagged["image_id"].tolist(), raw_root, out_dir, thumb)
        LOGGER.info("Contact sheets written for %d flagged images.", len(flagged))


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
    default=Path("results/qa_preprocessing"),
    show_default=True,
)
@click.option("--thumb", type=int, default=224, show_default=True)
@click.option("--limit", type=int, default=0, show_default=True, help="0 = all")
def cli(
    splits_dir: Path, raw_root: Path, out_dir: Path, thumb: int, limit: int
) -> None:
    main(splits_dir, raw_root, out_dir, thumb, limit)


if __name__ == "__main__":
    cli()
