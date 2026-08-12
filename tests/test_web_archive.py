"""Security regressions for uploaded ZIP extraction."""

import zipfile

import pytest

from src.web.archive import extract_flat_archive


def test_archive_rejects_parent_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.dcm", b"not-a-dicom")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_flat_archive(archive, tmp_path / "work")
    assert not (tmp_path / "outside.dcm").exists()


def test_archive_rejects_duplicate_flattened_names(tmp_path):
    archive = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("first/a.npy", b"one")
        zf.writestr("second/a.npy", b"two")

    with pytest.raises(ValueError, match="duplicate basename"):
        extract_flat_archive(archive, tmp_path / "work")
