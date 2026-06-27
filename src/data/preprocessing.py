"""DICOM to tensor pipeline."""

from pathlib import Path

import numpy as np


def load_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM and return a float32 array normalised to the unit range."""
    import pydicom

    arr = pydicom.dcmread(str(path)).pixel_array.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def apply_clahe(
    arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE)."""
    import cv2

    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(arr_u8).astype(np.float32) / 255.0


def strip_border(
    arr: np.ndarray, sat_thresh: float = 0.85, edge_px: int = 5
) -> np.ndarray:
    """Zero the bright film-digitisation frame left on CBIS-DDSM scans."""
    import cv2

    # 0.85, not 0.9, so bright calcifications near the border still count as
    # tissue after min-max normalisation.
    sat = (arr >= sat_thresh).astype(np.uint8)
    n, labels = cv2.connectedComponents(sat, connectivity=8)
    if n <= 1:
        return arr.copy()
    # edge_px deep, since some scans have 1-2 background pixels before the frame.
    edge_labels = (
        set(labels[:edge_px, :].ravel())
        | set(labels[-edge_px:, :].ravel())
        | set(labels[:, :edge_px].ravel())
        | set(labels[:, -edge_px:].ravel())
    )
    edge_labels.discard(np.intp(0))
    if not edge_labels:
        return arr.copy()
    out = arr.copy()
    out[np.isin(labels, list(edge_labels))] = 0.0
    return out


def segment_breast(
    arr: np.ndarray,
    blur_ksize: int = 5,
    open_ksize: int = 20,
    close_ksize: int = 25,
    margin_frac: float = 0.02,
    remove_border: bool = True,
) -> np.ndarray:
    """Return a float32 {0., 1.} breast mask via Otsu and the largest component."""
    import cv2

    # Strip film-scan bands first so they cannot merge with the breast component.
    if remove_border:
        arr = strip_border(arr)
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    blur = cv2.medianBlur(u8, blur_ksize)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.ones(arr.shape, np.float32)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8)
    # Fill interior holes (dark fatty regions below the Otsu threshold) so the
    # mask does not cut holes in breast tissue when applied.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8)
    )

    # Grow the mask past the skin line so the border gradient is not clipped.
    margin = int(round(max(arr.shape) * margin_frac))
    if margin > 0:
        filled = cv2.dilate(filled, np.ones((margin, margin), np.uint8))
    return filled.astype(np.float32)


def breast_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the (y0, y1, x0, x1) bounding box of the mask's set pixels."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _segment(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strip the film border and return the (stripped image, breast mask)."""
    arr = strip_border(arr)
    mask = segment_breast(arr, remove_border=False)
    # Intersect with arr > 0 so the dilation margin cannot reintroduce the
    # zeroed border or feed CLAHE a steep zero-to-tissue gradient.
    mask = mask * (arr > 1e-6).astype(np.float32)
    return arr, mask


def breast_crop_box(path: str | Path) -> tuple[int, int, int, int]:
    """Return the breast bounding box using the same pipeline as `preprocess`."""
    _, mask = _segment(load_dicom(path))
    return breast_bbox(mask)


def resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize to a fixed square. 224 matches the ImageNet backbone input."""
    import cv2

    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def normalise(arr: np.ndarray, mean: float = 0.485, std: float = 0.229) -> np.ndarray:
    """Normalise `arr` using ImageNet mean and std."""
    return (arr - mean) / std


def preprocess_array(
    arr: np.ndarray, image_size: int = 224, use_clahe: bool = True
) -> np.ndarray:
    """Segmented crop -> optional CLAHE -> resize, from a unit-range array."""
    arr, mask = _segment(arr)
    y0, y1, x0, x1 = breast_bbox(mask)
    mask_crop = mask[y0:y1, x0:x1]
    arr = arr[y0:y1, x0:x1]
    if use_clahe:
        tissue_mean = float(np.mean(arr[mask_crop > 0])) if mask_crop.any() else 0.5
        arr_filled = np.where(mask_crop > 0, arr, tissue_mean)
        arr = apply_clahe(arr_filled) * mask_crop
    else:
        arr = arr * mask_crop
    arr = resize(arr, image_size)
    return arr.astype(np.float32)


def preprocess(
    path: str | Path, image_size: int = 224, use_clahe: bool = True
) -> np.ndarray:
    """Full pipeline: DICOM -> segmented crop -> optional CLAHE -> resize -> float32."""
    return preprocess_array(load_dicom(path), image_size, use_clahe)
