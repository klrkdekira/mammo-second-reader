"""CBIS-DDSM DICOM path resolution.

Three folder-naming conventions appear across the four split CSVs:
1. Exact match      - CSV base folder name exists on disk as-is.
2. Suffix mismatch  - folder stored with a different _N suffix than the CSV.
3. Bare patient dir - P_xxxxx_VIEW without the lesion-type prefix.

Crop vs mask disambiguation uses pixel area (smaller = crop, larger = mask)
rather than CSV column order, which would swap them in roughly half of cases.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


class DICOMPathResolver:
    """Maps CBIS-DDSM CSV path strings to on-disk DICOM Paths.

    Build once, then call `resolve_dataframe` on each of the four split
    DataFrames. The instance caches header reads so repeated lookups for
    the same file are cheap.
    """

    _PREFIX_RE = re.compile(r"^[A-Za-z]+-(?:Test|Training)_")

    def __init__(self, dicom_dir: Path) -> None:
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.is_dir():
            raise ValueError(f"DICOM directory not found: {dicom_dir}")

        self.dicom_dir = dicom_dir
        self._folder_lookup: dict[str, dict[int, Path]] = defaultdict(dict)
        self._dcm_dir_lookup: dict[str, dict[int, Path]] = defaultdict(dict)
        self._area_cache: dict[Path, int] = {}
        self._mask_cache: dict[Path, bool] = {}
        self._build_lookups()

    def _build_lookups(self) -> None:
        for p in self.dicom_dir.iterdir():
            if not p.is_dir():
                continue
            if m := re.match(r"^(.+)_(\d+)$", p.name):
                self._folder_lookup[m.group(1)][int(m.group(2))] = p
            elif m := re.match(r"^(P_\d+_.+?)(?:_(\d+))?\.dcm$", p.name):
                self._dcm_dir_lookup[m.group(1)][
                    int(m.group(2)) if m.group(2) else 0
                ] = p

    def _resolve_folder(self, csv_path: str) -> Path | None:
        """Return the on-disk series folder for the first component of a CSV path."""
        base = Path(csv_path).parts[0]

        # Convention 1: exact match
        if (folder := self.dicom_dir / base).exists():
            return folder

        # Convention 2: numeric suffix mismatch, strip _N and pick lowest-numbered match
        core = re.sub(r"_\d+$", "", base)
        if series := self._folder_lookup.get(core):
            return series[min(series)]

        # Convention 3: bare patient folder, strip lesion-type prefix
        bare = self._PREFIX_RE.sub("", base)
        bare_series = self._dcm_dir_lookup.get(re.sub(r"_\d+$", "", bare), {})
        if not bare_series:
            return None
        n_hint = int(m.group(1)) if (m := re.search(r"_(\d+)$", bare)) else 0
        return bare_series.get(n_hint, bare_series[min(bare_series)])

    def _bare_folders(self, csv_path: str) -> list[Path]:
        """All bare P_xxxxx_VIEW folders matching this CSV path's patient/view."""
        bare_core = re.sub(
            r"_\d+$", "", self._PREFIX_RE.sub("", Path(csv_path).parts[0])
        )
        return list(self._dcm_dir_lookup.get(bare_core, {}).values())

    def _dcms(self, folder: Path | None) -> list[Path]:
        return sorted(
            p
            for p in (folder.rglob("*.dcm") if folder and folder.exists() else [])
            if p.is_file()
        )

    def _area(self, p: Path) -> int:
        if p not in self._area_cache:
            d = pydicom.dcmread(str(p), stop_before_pixels=True)
            self._area_cache[p] = int(getattr(d, "Rows", 0)) * int(
                getattr(d, "Columns", 0)
            )
        return self._area_cache[p]

    def _is_mask(self, p: Path) -> bool:
        if p not in self._mask_cache:
            self._mask_cache[p] = (
                int(np.unique(pydicom.dcmread(str(p)).pixel_array).size) <= 2
            )
        return self._mask_cache[p]

    def _resolve_crop_and_mask(
        self, cropped_csv: str | None, roi_csv: str | None
    ) -> tuple[Path | None, Path | None]:
        files = (
            self._dcms(self._resolve_folder(cropped_csv)) if cropped_csv else []
        ) or (self._dcms(self._resolve_folder(roi_csv)) if roi_csv else [])
        if len(files) >= 2:
            ranked = sorted(files, key=self._area)
            return ranked[0], ranked[-1]
        return (files[0], None) if files else (None, None)

    def _resolve_full_image(self, image_csv: str) -> Path | None:
        """Recover the full mammogram, falling back to bare folders for calc_test."""
        full_dcms = self._dcms(self._resolve_folder(image_csv))
        if len(full_dcms) == 1:
            return full_dcms[0]
        candidates = sorted(
            set(
                full_dcms
                + [f for b in self._bare_folders(image_csv) for f in self._dcms(b)]
            ),
            key=self._area,
            reverse=True,
        )
        return next(
            (p for p in candidates if not self._is_mask(p)),
            candidates[0] if candidates else None,
        )

    def resolve_case(
        self, image_csv: object, cropped_csv: object, roi_csv: object
    ) -> tuple[Path | None, Path | None, Path | None]:
        """Return (full_image, cropped, roi_mask) Paths for one annotation row."""
        full_image = (
            self._resolve_full_image(image_csv) if isinstance(image_csv, str) else None
        )
        cropped, mask = self._resolve_crop_and_mask(
            cropped_csv if isinstance(cropped_csv, str) else None,
            roi_csv if isinstance(roi_csv, str) else None,
        )
        return full_image, cropped, mask

    def resolve_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add full_image_path, cropped_path, roi_mask_path columns to df in-place."""
        resolved = df.apply(
            lambda r: self.resolve_case(
                r["image file path"],
                r["cropped image file path"],
                r["ROI mask file path"],
            ),
            axis=1,
        )
        df[["full_image_path", "cropped_path", "roi_mask_path"]] = pd.DataFrame(
            resolved.tolist(), index=df.index
        )
        return df
