"""Unit tests for _gradcam_roi_panel's handling of degenerate Grad-CAM output."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate import _gradcam_roi_panel


class _FakeDataset:
    """Minimal stand-in for MammogramDataset: no image files touched."""

    def __init__(self, n: int, rois: list[np.ndarray | None]):
        self.df = pd.DataFrame({"label": [1] * n})
        self._rois = rois

    def __getitem__(self, idx: int):
        return torch.zeros(3, 8, 8), torch.tensor(1.0)

    def load_roi(self, idx: int, out_shape) -> np.ndarray | None:
        return self._rois[idx]


def _roi(shape=(8, 8)) -> np.ndarray:
    roi = np.zeros(shape, dtype=bool)
    roi[2, 2] = True
    return roi


def test_all_degenerate_cams_returns_none_with_reason():
    n = 3
    ds = _FakeDataset(n, rois=[_roi()] * n)
    zero_cams = [np.zeros((8, 8)) for _ in range(n)]

    with patch("src.evaluation.gradcam.compute_gradcam", side_effect=zero_cams):
        result, reason = _gradcam_roi_panel(
            model=None,
            test_ds=ds,
            model_name="vgg16",
            y_prob=np.ones(n),
            threshold=0.5,
            device=torch.device("cpu"),
        )

    assert result is None
    assert "zero-energy" in reason


def test_degenerate_cams_excluded_but_usable_cases_still_scored():
    n = 3
    ds = _FakeDataset(n, rois=[_roi()] * n)
    cams = [np.zeros((8, 8)), np.ones((8, 8)), np.ones((8, 8))]

    with patch("src.evaluation.gradcam.compute_gradcam", side_effect=cams):
        result, reason = _gradcam_roi_panel(
            model=None,
            test_ds=ds,
            model_name="vgg16",
            y_prob=np.ones(n),
            threshold=0.5,
            device=torch.device("cpu"),
        )

    assert reason is None
    assert result["n_malignant_scored"] == 2
    assert result["n_degenerate_excluded"] == 1
    assert result["n_no_roi_excluded"] == 0


def test_missing_roi_mask_excluded_and_counted():
    n = 3
    ds = _FakeDataset(n, rois=[_roi(), None, _roi()])
    cams = [np.ones((8, 8)) for _ in range(n)]

    with patch("src.evaluation.gradcam.compute_gradcam", side_effect=cams):
        result, reason = _gradcam_roi_panel(
            model=None,
            test_ds=ds,
            model_name="vgg16",
            y_prob=np.ones(n),
            threshold=0.5,
            device=torch.device("cpu"),
        )

    assert reason is None
    assert result["n_malignant_scored"] == 2
    assert result["n_no_roi_excluded"] == 1


def test_unknown_model_name_returns_none_with_reason():
    ds = _FakeDataset(1, rois=[_roi()])

    result, reason = _gradcam_roi_panel(
        model=None,
        test_ds=ds,
        model_name="not_a_real_model",
        y_prob=np.ones(1),
        threshold=0.5,
        device=torch.device("cpu"),
    )

    assert result is None
    assert "target layer" in reason
