"""Regression tests for preprocessing pipeline using synthetic scan images."""

import cv2
import numpy as np
import pytest

from src.data.preprocessing import (
    artifact_mask,
    breast_bbox,
    dicom_to_array,
    preprocess_array,
    segment_breast,
)


class _FakeDicom:
    def __init__(self, pixel_array, photometric):
        self.pixel_array = pixel_array
        self.PhotometricInterpretation = photometric


def test_monochrome1_is_inverted():
    ramp = np.tile(np.arange(256, dtype=np.uint16), (4, 1))
    mono2 = dicom_to_array(_FakeDicom(ramp, "MONOCHROME2"))
    mono1 = dicom_to_array(_FakeDicom(ramp, "MONOCHROME1"))
    assert mono2[0, 0] == 0.0 and mono2[0, -1] == pytest.approx(1.0)
    assert mono1[0, 0] == pytest.approx(1.0) and mono1[0, -1] == 0.0
    np.testing.assert_allclose(mono1, 1.0 - mono2, atol=1e-6)


def test_missing_photometric_defaults_to_no_flip():
    ramp = np.tile(np.arange(256, dtype=np.uint16), (4, 1))
    plain = dicom_to_array(_FakeDicom(ramp, ""))
    assert plain[0, 0] == 0.0 and plain[0, -1] == pytest.approx(1.0)


H, W = 2400, 1400
BAND = 24  # matches segment_breast's 1% edge margin on this size


def _synthetic_scan() -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic scan and its breast mask."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    r = np.sqrt(((yy - 1200) / 900) ** 2 + (xx / 650) ** 2)
    tissue = r < 1.0
    img = np.where(tissue, 0.64 * np.clip(1 - r, 0, 1) ** 0.7 + 0.06, 0.01)

    frame = np.zeros((H, W), np.float32)
    frame[6:18, 6:-6] = frame[-18:-6, 6:-6] = 1.0
    frame[6:-6, 6:18] = frame[6:-6, -18:-6] = 1.0
    glow = cv2.GaussianBlur(frame, (31, 31), 0) * 0.15
    img = np.maximum(img, np.maximum(frame, glow))

    cv2.circle(img, (0, 0), 40, 1.0, -1)

    img[300:390, 1050:1250] = 0.95

    cv2.circle(img, (400, 1200), 4, 0.95, -1)

    return img.astype(np.float32), tissue


@pytest.fixture(scope="module")
def scan():
    img, tissue = _synthetic_scan()
    breast = segment_breast(img)
    art = artifact_mask(img, breast)
    clean = img.copy()
    clean[art > 0] = 0.0
    return img, tissue, breast, art, clean


def test_marker_removed(scan):
    _, _, _, _, clean = scan
    assert clean[300:390, 1050:1250].max() < 0.05


def test_frame_removed_on_air_edges(scan):
    _, _, _, _, clean = scan
    zone = 40
    assert clean[:, -zone:].max() < 0.1
    assert clean[:zone, 600:].max() < 0.1
    assert clean[-zone:, 600:].max() < 0.1


def test_frame_removed_where_it_crosses_tissue(scan):
    _, _, _, _, clean = scan
    assert (clean[:, :40] > 0.85).sum() == 0


def test_faint_skin_line_kept(scan):
    _, tissue, breast, _, _ = scan
    inner = np.zeros_like(tissue)
    pad = 2 * BAND
    inner[pad:-pad, pad:-pad] = tissue[pad:-pad, pad:-pad]
    covered = (breast > 0) & inner
    assert covered.sum() / inner.sum() > 0.97


def test_no_tissue_deleted(scan):
    img, tissue, _, _, clean = scan
    inner = np.zeros_like(tissue)
    pad = 2 * BAND
    inner[pad:-pad, pad:-pad] = tissue[pad:-pad, pad:-pad]
    changed = (clean != img) & inner
    assert changed.sum() / inner.sum() < 0.005


def test_interior_speck_preserved(scan):
    _, _, _, _, clean = scan
    assert clean[1196:1204, 396:404].max() > 0.9


def test_crop_box_has_no_black_stripe(scan):
    _, _, breast, _, clean = scan
    y0, y1, x0, x1 = breast_bbox(breast)
    crop, mask = clean[y0:y1, x0:x1], breast[y0:y1, x0:x1]
    for band, inside in (
        (crop[:5], mask[:5] > 0),
        (crop[-5:], mask[-5:] > 0),
        (crop[:, :5], mask[:, :5] > 0),
        (crop[:, -5:], mask[:, -5:] > 0),
    ):
        if inside.mean() > 0.2:
            assert float(band[inside].mean()) > 0.02


def test_preprocess_array_output(scan):
    img, _, _, _, _ = scan
    out = preprocess_array(img, image_size=224)
    assert out.shape == (224, 224)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-5
    assert out.max() > 0.3  # tissue actually present
