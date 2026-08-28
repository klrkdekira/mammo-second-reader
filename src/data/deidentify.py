"""Strip identifying DICOM tags from an uploaded dataset before any other code touches it."""

import pydicom

IDENTIFYING_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "AccessionNumber",
    "StudyID",
)


def strip_identifying_tags(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Remove identifying tags from a DICOM dataset in place, and return it."""
    for keyword in IDENTIFYING_KEYWORDS:
        if keyword in ds:
            delattr(ds, keyword)
    return ds
