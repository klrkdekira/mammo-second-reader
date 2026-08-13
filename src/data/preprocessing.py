"""DICOM to model-input tensor pipeline.

Segment breast tissue, remove film/view artifacts, apply optional CLAHE,
and crop/resize to model dimensions.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pydicom

import cv2
import numpy as np


def dicom_to_array(ds: "pydicom.Dataset | pydicom.FileDataset") -> np.ndarray:
    """Normalise DICOM pixel values to float32 in [0, 1].

    Inverts MONOCHROME1 images so high values represent bright tissue.
    """
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return cast(np.ndarray, (arr - arr.min()) / (arr.max() - arr.min() + 1e-8))


def load_dicom(path: str | Path) -> np.ndarray:
    """Read DICOM file as float32 array in [0, 1]."""
    import pydicom

    return dicom_to_array(pydicom.dcmread(str(path)))


def apply_clahe(
    arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to a float32 array in [0, 1]."""
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return cast(np.ndarray, clahe.apply(u8).astype(np.float32) / 255.0)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary uint8 mask."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _not_air(
    arr: np.ndarray,
    air_thresh: float = 0.03,
    blur_ksize: int = 5,
    open_ksize: int = 15,
) -> np.ndarray:
    """Binary uint8 mask for non-air pixels."""
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    mask = (cv2.medianBlur(u8, blur_ksize) > round(air_thresh * 255)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(open_ksize))


def _film_border(
    arr: np.ndarray,
    sat_thresh: float = 0.85,
    max_thick: int = 15,
    edge_px: int = 5,
    band_frac: float = 0.01,
    glow_thresh: float = 0.03,
    grow_ksize: int = 9,
) -> np.ndarray:
    """Binary uint8 mask of outer film border lines and glow."""

    def _edge_ids(labels: np.ndarray) -> np.ndarray:
        edge = np.concatenate(
            [
                labels[:edge_px].ravel(),
                labels[-edge_px:].ravel(),
                labels[:, :edge_px].ravel(),
                labels[:, -edge_px:].ravel(),
            ]
        )
        ids = np.unique(edge)
        return ids[ids != 0]

    band = max(1, round(max(arr.shape) * band_frac))
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    border = (cv2.medianBlur(u8, 5) > round(glow_thresh * 255)).astype(np.uint8)
    border[band:-band, band:-band] = 0

    sat = (arr >= sat_thresh).astype(np.uint8)
    k = 2 * max_thick + 1
    thin = sat - cv2.morphologyEx(sat, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels = cv2.connectedComponents(thin, connectivity=8)
    if n > 1:
        border |= np.isin(labels, _edge_ids(labels)).astype(np.uint8)
    confine = max(3 * max_thick, 2 * band)
    n, labels = cv2.connectedComponents(sat, connectivity=8)
    if n > 1:
        inland = np.unique(labels[confine:-confine, confine:-confine])
        ids = np.setdiff1d(_edge_ids(labels), inland)
        border |= np.isin(labels, ids).astype(np.uint8)
    if not border.any():
        return border
    return cast(np.ndarray, cv2.dilate(border, _ellipse(grow_ksize)))


def segment_breast(
    arr: np.ndarray,
    air_thresh: float = 0.03,
    edge_margin_frac: float = 0.01,
    close_ksize: int = 25,
) -> np.ndarray:
    """Segment largest non-air component as the breast region mask."""
    border = _film_border(arr)
    notair = _not_air(arr, air_thresh) & (1 - border)
    margin = max(1, round(max(arr.shape) * edge_margin_frac))
    inner = notair.copy()
    inner[:margin], inner[-margin:] = 0, 0
    inner[:, :margin], inner[:, -margin:] = 0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
    if n <= 1:
        return np.ones(arr.shape, np.float32)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8)
    mask = cv2.dilate(mask, _ellipse(2 * margin + 1)) & notair
    mask = cv2.morphologyEx(_fill_holes(mask), cv2.MORPH_CLOSE, _ellipse(close_ksize))
    mask = _fill_holes(mask) & (1 - border)
    return cast(np.ndarray, mask.astype(np.float32))


def artifact_mask(
    arr: np.ndarray, breast: np.ndarray | None = None, grow_ksize: int = 31
) -> np.ndarray:
    """Bright non-breast blobs: view marker, film frame, scanner specks.

    Grown so the faint halo around burned-in text goes with it, but never
    into the breast itself. The frame's white line is the one exception: it
    is film, not tissue, so it is zeroed even where it overlaps the breast
    mask at the image edge.
    """
    if breast is None:
        breast = segment_breast(arr)
    keep = (breast > 0).astype(np.uint8)
    inv_keep = (1 - keep).astype(np.uint8)
    art_initial = _not_air(arr) & inv_keep
    dilated = np.asarray(cv2.dilate(art_initial, _ellipse(grow_ksize)), dtype=np.uint8)
    art = dilated & inv_keep
    return np.maximum(art, _film_border(arr)).astype(np.float32)


def breast_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the (y0, y1, x0, x1) bounding box of the mask's set pixels."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def breast_mask(arr: np.ndarray) -> np.ndarray:
    """Hole-free breast mask for `arr`."""
    return segment_breast(arr)


def breast_crop_box(path: str | Path) -> tuple[int, int, int, int]:
    """Breast bounding box, matching `preprocess`'s crop."""
    return breast_bbox(breast_mask(load_dicom(path)))


def resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize to a fixed square. 224 matches the ImageNet backbone input."""
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_aligned_array(
    arr: np.ndarray, use_clahe: bool = True
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Clean and breast-crop an image without resizing it.

    Returns ``(image, breast_mask, source_bbox)``.  The image and mask have
    identical geometry, and ``source_bbox`` maps their coordinates back to the
    original DICOM array.  Patch extraction uses this path so lesion masks are
    aligned before any whole-image downscaling.
    """
    breast = segment_breast(arr)
    clean = arr.copy()
    clean[artifact_mask(arr, breast) > 0] = 0.0
    y0, y1, x0, x1 = breast_bbox(breast)
    clean = clean[y0:y1, x0:x1]
    breast = breast[y0:y1, x0:x1]
    if use_clahe:
        fill = float(clean[breast > 0].mean()) if breast.any() else 0.5
        clean = np.where(
            breast > 0,
            apply_clahe(np.where(breast > 0, clean, fill)),
            clean,
        )
    return clean.astype(np.float32), breast.astype(np.uint8), (y0, y1, x0, x1)


def normalise(arr: np.ndarray, mean: float = 0.485, std: float = 0.229) -> np.ndarray:
    """Normalise `arr` using the ImageNet red-channel mean and std.

    These are the single-channel ImageNet statistics (0.485 / 0.229), applied
    to the grayscale mammogram; `ThreeChannelWrapper` then repeats that one
    normalised channel to three, so all three backbone inputs share the
    red-channel stats rather than the true per-channel means/stds. This is a
    deliberate simplification for single-channel medical images and is applied
    identically in training (`augment.A.Normalize`) and inference, so it is a
    consistent convention, not a train/serve skew.
    """
    return (arr - mean) / std


def preprocess_array(
    arr: np.ndarray, image_size: int = 224, use_clahe: bool = True
) -> np.ndarray:
    """Zero artefacts, crop to the breast, optionally CLAHE, resize.

    Only artefact pixels are altered; breast and air keep their original
    values (air is near-black already), so no tissue is ever deleted.
    """
    arr, _, _ = preprocess_aligned_array(arr, use_clahe=use_clahe)
    return resize(arr, image_size).astype(np.float32)


def preprocess(
    path: str | Path, image_size: int = 224, use_clahe: bool = True
) -> np.ndarray:
    """Full pipeline: DICOM -> breast crop -> optional CLAHE -> resize -> float32."""
    return preprocess_array(load_dicom(path), image_size, use_clahe)
