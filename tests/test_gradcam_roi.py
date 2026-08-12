"""Unit tests for Grad-CAM-vs-ROI localisation metrics."""

import numpy as np
import pytest

from src.evaluation.gradcam_roi import (
    energy_pointing_game,
    is_degenerate,
    pointing_game,
)


def test_is_degenerate_flags_zero_energy_heatmap():
    cam = np.zeros((8, 8))
    assert is_degenerate(cam)


def test_is_degenerate_false_for_normal_heatmap():
    cam = np.zeros((8, 8))
    cam[3, 4] = 1.0
    assert not is_degenerate(cam)


def test_pointing_game_hit():
    cam = np.zeros((4, 4))
    cam[1, 2] = 1.0
    roi = np.zeros((4, 4), dtype=bool)
    roi[1, 2] = True
    assert pointing_game(cam, roi) is True


def test_pointing_game_miss():
    cam = np.zeros((4, 4))
    cam[0, 0] = 1.0
    roi = np.zeros((4, 4), dtype=bool)
    roi[3, 3] = True
    assert pointing_game(cam, roi) is False


def test_energy_pointing_game_uniform_heatmap_returns_roi_area_fraction():
    cam = np.ones((10, 10))
    roi = np.zeros((10, 10), dtype=bool)
    roi[:3, :4] = True  # 12 of 100 pixels

    result = energy_pointing_game(cam, roi)

    assert result == pytest.approx(0.12)


def test_energy_pointing_game_zero_energy_heatmap_returns_zero():
    cam = np.zeros((4, 4))
    roi = np.zeros((4, 4), dtype=bool)
    roi[0, 0] = True
    assert energy_pointing_game(cam, roi) == 0.0
