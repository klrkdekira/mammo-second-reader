"""Regression tests for deterministic CBIS-DDSM path resolution."""

from pathlib import Path

from src.data.cbis_ddsm import DICOMPathResolver


def test_equal_area_full_image_duplicates_use_path_tiebreaker(monkeypatch):
    resolver = object.__new__(DICOMPathResolver)
    folder = Path("series")
    first = folder / "a" / "image.dcm"
    second = folder / "b" / "image.dcm"

    monkeypatch.setattr(resolver, "_resolve_folder", lambda csv_path: folder)
    monkeypatch.setattr(resolver, "_dcms", lambda candidate: [second, first])
    monkeypatch.setattr(resolver, "_bare_folders", lambda csv_path: [])
    monkeypatch.setattr(resolver, "_area", lambda path: 100)
    monkeypatch.setattr(resolver, "_is_mask", lambda path: False)

    assert resolver._resolve_full_image("Mass-Training_P_00001_LEFT_CC") == first
