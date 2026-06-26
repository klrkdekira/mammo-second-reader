"""Quantitative Grad-CAM evaluation against CBIS-DDSM ROIs.

Metrics:

- Pointing Game accuracy: fraction of cases where the heatmap argmax
  pixel lies inside the lesion ROI.
- Energy-based pointing game: fraction of total heatmap energy that falls
  inside the ROI (Wang et al. 2020, Score-CAM). Unlike the single-pixel
  pointing game and the area-sensitive IoU, this stays meaningful when the
  lesion is a tiny fraction of the image - which CBIS-DDSM lesions are
  (typically well under 1% of pixels). Compare it against `roi_area_frac`,
  the score a model that ignored the lesion entirely would get by chance.
- Heatmap-ROI IoU: IoU of (cam >= 0.5) and the ROI mask. Reported for
  completeness, but structurally small here: a broad heatmap region cannot
  overlap a sub-1% ROI by much, so read this alongside the energy metric.
- Centroid distance: Euclidean distance between cam centroid and ROI
  centroid, normalised by the image diagonal.
"""

import numpy as np


def pointing_game(cam: np.ndarray, roi: np.ndarray) -> bool:
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    return bool(roi[y, x])


def energy_pointing_game(cam: np.ndarray, roi: np.ndarray) -> float:
    """Fraction of total Grad-CAM energy lying inside the ROI.

    A perfect localiser approaches 1.0; a model that ignores the lesion scores
    roughly the ROI's area fraction. Robust to tiny lesions, where the
    single-pixel pointing game is noisy and IoU is structurally near-zero.
    """
    total = float(cam.sum())
    if total <= 0.0:
        return 0.0
    return float((cam * (roi > 0)).sum() / total)


def heatmap_roi_iou(cam: np.ndarray, roi: np.ndarray, thr: float = 0.5) -> float:
    cam_bin = cam >= thr
    intersection = int(np.logical_and(cam_bin, roi).sum())
    union = int(np.logical_or(cam_bin, roi).sum())
    return float(intersection / union) if union else 0.0


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=float)


def centroid_distance(cam: np.ndarray, roi: np.ndarray) -> float:
    cam_bin = cam >= cam.max() * 0.5
    c_cam = _centroid(cam_bin)
    c_roi = _centroid(roi.astype(bool))
    if c_cam is None or c_roi is None:
        return float("nan")
    h, w = cam.shape
    return float(np.linalg.norm(c_cam - c_roi) / np.hypot(h, w))


def grad_cam_subset_stats(
    cams: list[np.ndarray], rois: list[np.ndarray], mask: np.ndarray
) -> dict:
    pg = np.array([pointing_game(c, r) for c, r in zip(cams, rois)])[mask]
    ebpg = np.array([energy_pointing_game(c, r) for c, r in zip(cams, rois)])[mask]
    iou = np.array([heatmap_roi_iou(c, r) for c, r in zip(cams, rois)])[mask]
    cd = np.array([centroid_distance(c, r) for c, r in zip(cams, rois)])[mask]
    area = np.array([float((r > 0).mean()) for r in rois])[mask]
    return {
        "n": int(mask.sum()),
        "pointing_game": float(pg.mean()) if pg.size else 0.0,
        "energy_pointing_game": float(ebpg.mean()) if ebpg.size else 0.0,
        "roi_area_frac": float(area.mean()) if area.size else 0.0,
        "iou_mean": float(np.nanmean(iou)) if iou.size else 0.0,
        "iou_sd": float(np.nanstd(iou)) if iou.size else 0.0,
        "centroid_mean": float(np.nanmean(cd)) if cd.size else 0.0,
        "centroid_sd": float(np.nanstd(cd)) if cd.size else 0.0,
    }
