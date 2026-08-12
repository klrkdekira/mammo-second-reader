"""Checks that resolution-specific cache files cannot be mixed."""

import numpy as np

from src.data.dicom_to_png import _cache_has_shape


def test_cache_shape_must_match_requested_resolution(tmp_path):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((224, 224), dtype=np.float32))

    assert _cache_has_shape(path, 224)
    assert not _cache_has_shape(path, 448)


def test_invalid_cache_file_is_rejected(tmp_path):
    path = tmp_path / "image.npy"
    path.write_text("not a numpy array")

    assert not _cache_has_shape(path, 448)
