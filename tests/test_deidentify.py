"""Unit tests for DICOM identifying-tag stripping."""

import pydicom

from src.data.deidentify import strip_identifying_tags


def test_strip_identifying_tags_removes_patient_name():
    ds = pydicom.Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.Rows = 8

    strip_identifying_tags(ds)

    assert "PatientName" not in ds
    assert "PatientID" not in ds
    assert ds.Rows == 8


def test_strip_identifying_tags_tolerates_missing_tags():
    ds = pydicom.Dataset()
    ds.Rows = 8

    strip_identifying_tags(ds)

    assert "PatientName" not in ds
    assert ds.Rows == 8
