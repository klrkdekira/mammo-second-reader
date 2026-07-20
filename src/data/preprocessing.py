"""DICOM to model-input tensor pipeline.

The scans are digitised film: a flat, near-black air background plus three
kinds of bright artefact, the film frame, the burned-in view marker, and
scanner specks. The breast is found as the largest above-air component and
bright pixels *outside* it are zeroed. Breast pixels are never masked out,
so a segmentation error cannot carve cavities into faint peripheral tissue.
"""

from pathlib import Path

import cv2
import numpy as np


def dicom_to_array(ds: "object") -> np.ndarray:
    """Normalise a read DICOM dataset to a float32 array in [0, 1].

    Handles `PhotometricInterpretation == "MONOCHROME1"`, where high stored
    values are *dark*. CBIS-DDSM is a conversion of the film-based DDSM and
    some series carry this convention; without the flip such an image reaches
    the network inverted (breast dark, air bright), which also defeats the
    air-threshold segmentation in `_not_air`. The flip is a no-op for the
    usual MONOCHROME2 series.

    Rescale slope/intercept is deliberately not applied: min-max normalisation
    is invariant to a positive affine transform, so it would change nothing.
    """
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def load_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM as a float32 array scaled to [0, 1]."""
    import pydicom

    return dicom_to_array(pydicom.dcmread(str(path)))


def apply_clahe(
    arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on a [0, 1] array."""
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(u8).astype(np.float32) / 255.0


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill fully-enclosed holes, keeping the (concave) outline. Returns uint8."""
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
    """Binary uint8 mask of everything brighter than film air.

    The background is flat and near zero, so a low fixed threshold traces the
    faint skin line; Otsu's bimodal split lands mid-tissue (~0.3 on these
    scans) and cuts the breast outline. Opening breaks the thin bridges
    between breast, film frame, and marker halos.
    """
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
    """Binary uint8 mask of the film frame's white border and its glow.

    Three rules, layered:
    - anything above-air in the outer `band_frac` strip is film edge, the
      frame line and its blurred glow live there, and tissue that far out
      sits behind them and is unreadable anyway;
    - thin saturated residue connected to the edge: the frame line where it
      fuses with dense tissue past the strip (an opening keeps thick blobs
      so only the line remains);
    - saturated components that touch the edge but stay confined near it:
      thick frame corners reaching past the strip.
    Saturated tissue survives: it is thick and extends far inland, and
    bright specks in the tissue are thin but never edge-touching. The whole
    mask is dilated a little so any remaining glow goes with it.
    """

    def _edge_ids(labels: np.ndarray) -> np.ndarray:
        edge = np.concatenate([
            labels[:edge_px].ravel(), labels[-edge_px:].ravel(),
            labels[:, :edge_px].ravel(), labels[:, -edge_px:].ravel(),
        ])
        ids = np.unique(edge)
        return ids[ids != 0]

    # Same blur and threshold arithmetic as `_not_air` so that, within the
    # band, border is a superset of not-air and the breast mask cannot keep
    # a sliver the artefact pass then blacks out.
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
    return cv2.dilate(border, _ellipse(grow_ksize))


def segment_breast(
    arr: np.ndarray,
    air_thresh: float = 0.03,
    edge_margin_frac: float = 0.01,
    close_ksize: int = 25,
) -> np.ndarray:
    """Binary breast mask: largest above-air component, holes filled.

    The film frame runs along the image edge and can touch the breast, so its
    white line is subtracted and a thin edge margin is cleared before picking
    the largest component. The mask is then grown back over that margin , 
    chest-wall tissue usually reaches the edge, but only onto above-air
    pixels, so the frame stays out. The border is re-subtracted at the end
    because closing regrows the mask into it, which would drag the crop box
    onto zeroed pixels and leave a black stripe at the crop edge.
    """
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
    return mask.astype(np.float32)


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
    art = _not_air(arr) & (1 - keep)
    art = cv2.dilate(art, _ellipse(grow_ksize)) & (1 - keep)
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
    breast = segment_breast(arr)
    arr = arr.copy()
    arr[artifact_mask(arr, breast) > 0] = 0.0
    y0, y1, x0, x1 = breast_bbox(breast)
    arr, breast = arr[y0:y1, x0:x1], breast[y0:y1, x0:x1]
    if use_clahe:
        # Equalise inside the breast only; fill outside with the tissue mean
        # so CLAHE tiles at the skin line are not skewed by the black edge.
        fill = float(arr[breast > 0].mean()) if breast.any() else 0.5
        arr = np.where(breast > 0, apply_clahe(np.where(breast > 0, arr, fill)), arr)
    return resize(arr, image_size).astype(np.float32)


def preprocess(
    path: str | Path, image_size: int = 224, use_clahe: bool = True
) -> np.ndarray:
    """Full pipeline: DICOM -> breast crop -> optional CLAHE -> resize -> float32."""
    return preprocess_array(load_dicom(path), image_size, use_clahe)
