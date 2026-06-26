"""Shared mammography PyTorch Dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import manifest as _manifest


class MammogramDataset(Dataset):
    """PyTorch dataset for mammography images."""

    def __init__(
        self,
        split_csv: str | Path,
        image_root: str | Path,
        transform: Any | None = None,
        cache_suffix: str = ".npy",
    ) -> None:
        self.df = _manifest.read(split_csv)
        self.image_root = Path(image_root)
        self.transform = transform
        self.cache_suffix = cache_suffix

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> np.ndarray:
        path = self.image_root / f"{image_id}{self.cache_suffix}"
        if path.exists():
            return np.load(path)
        from src.data.preprocessing import preprocess

        dcm_path = self.image_root / f"{image_id}.dcm"
        return preprocess(dcm_path)

    def load_roi(self, idx: int, out_shape: tuple[int, int]) -> np.ndarray | None:
        """Return a binary ROI mask for row `idx`, resized to `out_shape` (H, W).

        Cropped to the same breast bounding box used by `preprocess` so it
        aligns with Grad-CAM heatmap coordinates.

        Returns `None` if the manifest has no `roi_mask_id` column, the
        value is missing, or the mask file does not exist on disk.
        """
        if "roi_mask_id" not in self.df.columns:
            return None
        row = self.df.iloc[idx]
        rid = row["roi_mask_id"]
        if (
            rid is None
            or (isinstance(rid, float) and np.isnan(rid))
            or str(rid) in ("", "nan", "<NA>", "None")
        ):
            return None
        import cv2

        base = self.image_root / str(rid)
        npy = base.with_suffix(self.cache_suffix)
        dcm = base.with_suffix(".dcm")
        if npy.exists():
            mask = np.load(npy)
        elif dcm.exists():
            import pydicom

            mask = pydicom.dcmread(str(dcm)).pixel_array
        else:
            return None
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        mask = self._crop_roi_to_breast(str(row["image_id"]), mask)
        mask = cv2.resize(
            mask, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return (mask > 0).astype(np.uint8)

    def _crop_roi_to_breast(self, image_id: str, mask: np.ndarray) -> np.ndarray:
        """Crop an ROI mask to the breast bounding box of its source image.

        Recomputes the box from the raw DICOM, since the cached .npy is already
        cropped. Returns `mask` unchanged if the raw DICOM is unavailable.
        """
        import cv2

        from src.data.preprocessing import breast_crop_box, load_dicom

        dcm = self.image_root / f"{image_id}.dcm"
        if not dcm.exists():
            return mask
        full = load_dicom(dcm)
        if mask.shape != full.shape:
            mask = cv2.resize(
                mask, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        y0, y1, x0, x1 = breast_crop_box(dcm)
        return mask[y0:y1, x0:x1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        arr = self._load_image(str(row["image_id"]))
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor, label
