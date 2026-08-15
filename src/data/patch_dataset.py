"""PyTorch dataset for frozen five-class mammography patch manifests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.patch_manifest import CLASS_TO_ID, PATCH_CLASSES


class PatchDataset(Dataset):
    """Load native-resolution patch arrays using paths in a Stage 0 manifest."""

    def __init__(
        self,
        manifest_csv: str | Path,
        patch_root: str | Path,
        transform: Callable[..., dict[str, np.ndarray] | np.ndarray] | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.patch_root = Path(patch_root)
        self.transform = transform
        self.df = pd.read_csv(self.manifest_csv)
        required = {
            "patch_id",
            "patch_path",
            "patient_id",
            "split",
            "patch_class",
            "class_id",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Patch manifest is missing columns: {sorted(missing)}")
        if self.df.empty:
            raise ValueError("Patch manifest is empty")
        if not set(self.df["patch_class"]).issubset(PATCH_CLASSES):
            raise ValueError("Patch manifest contains an unknown class")
        expected = self.df["patch_class"].map(CLASS_TO_ID).astype(int)
        if not expected.equals(self.df["class_id"].astype(int)):
            raise ValueError("Patch class names and IDs do not match")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self.patch_root / str(row["patch_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Patch array not found: {path}")
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(
                f"Patch array must be two-dimensional, got {arr.shape}: {path}"
            )
        if self.transform is not None:
            transformed = self.transform(image=arr)
            arr = transformed["image"] if isinstance(transformed, dict) else transformed
        tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        label = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return tensor, label
