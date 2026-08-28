"""Safely unpack uploaded mammogram ZIP files."""

from __future__ import annotations

import logging
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from src.data.deidentify import strip_identifying_tags

LOGGER = logging.getLogger(__name__)

MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000


def _safe_basename(info: zipfile.ZipInfo) -> str:
    """Return the filename if the archive path is safe."""
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts or not member.name:
        raise ValueError(f"Unsafe archive member path: {info.filename!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise ValueError(
            f"Symbolic links are not allowed in archives: {info.filename!r}"
        )
    return member.name


def extract_flat_archive(
    archive_path: str | Path,
    destination: Path,
    *,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    max_extracted_bytes: int = MAX_EXTRACTED_BYTES,
    max_files: int = MAX_ARCHIVE_FILES,
) -> list[Path]:
    """Safely extract a ZIP into one directory.

    Manifests use filenames, so nested folders are flattened. Duplicate
    filenames are rejected.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ValueError(f"Archive not found: {archive_path}")
    if archive_path.stat().st_size > max_archive_bytes:
        raise ValueError("Archive exceeds the compressed upload-size limit.")

    destination.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not open ZIP archive: {exc}") from exc

    with zf:
        infos = zf.infolist()
        declared_size = sum(max(int(info.file_size), 0) for info in infos)
        if declared_size > max_extracted_bytes:
            raise ValueError("Archive would decompress beyond the permitted size.")
        files = [info for info in infos if not info.is_dir()]
        if len(files) > max_files:
            raise ValueError(f"Archive contains more than {max_files} files.")

        extracted: list[Path] = []
        basenames: set[str] = set()
        actual_size = 0
        for info in files:
            basename = _safe_basename(info)
            if basename in basenames:
                raise ValueError(
                    f"Archive contains duplicate basename {basename!r}; "
                    "use unique image identifiers."
                )
            basenames.add(basename)
            target = destination / basename
            try:
                with zf.open(info) as source, target.open("xb") as sink:
                    while chunk := source.read(1024 * 1024):
                        actual_size += len(chunk)
                        if actual_size > max_extracted_bytes:
                            raise ValueError(
                                "Archive decompressed beyond the permitted size."
                            )
                        sink.write(chunk)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            extracted.append(target)
        return extracted


def deidentify_dicom_in_place(path: Path) -> bool:
    """Remove identifying tags from a DICOM file."""
    import pydicom

    try:
        ds = pydicom.dcmread(str(path))
        strip_identifying_tags(ds)
        ds.save_as(str(path), enforce_file_format=True)
        return True
    except Exception:  # noqa: BLE001
        LOGGER.warning("Could not de-identify DICOM %s; leaving it unchanged", path)
        return False


def move_file(source: Path, destination: Path) -> Path:
    """Move a file without overwriting another one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite extracted file: {destination.name}")
    shutil.move(str(source), str(destination))
    return destination
