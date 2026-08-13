"""Build deterministic, patient-disjoint CBIS-DDSM lesion-patch manifests.

Stage 0 deliberately reads only the official *training* case-description CSVs.
The existing whole-image train/validation manifests provide the locked patient
assignment; their test patients are used only as an exclusion set.  Images are
cleaned and breast-cropped at native resolution before aligned 224-pixel crops
are extracted.

The overlap rule follows the cited Shen et al. reference implementation:
``max(intersection / ROI area, intersection / patch area) >= cutoff``.  Both
components are retained in the manifest so the rule is auditable.  Background
patches are stricter than that implementation and must contain zero pixels from
the union of every known ROI in the source mammogram.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import click
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from src.config import setup_logging
from src.data.cbis_ddsm import DICOMPathResolver
from src.data.dicom_to_png import _find_dicom
from src.data.preprocessing import load_dicom, preprocess_aligned_array
from src.data.splits import _build_dataframe

LOGGER = logging.getLogger(__name__)

PATCH_CLASSES = (
    "benign_calcification",
    "malignant_calcification",
    "benign_mass",
    "malignant_mass",
    "background",
)
CLASS_TO_ID = {name: index for index, name in enumerate(PATCH_CLASSES)}
MANIFEST_COLUMNS = (
    "patch_id",
    "patch_path",
    "patient_id",
    "image_id",
    "roi_mask_id",
    "split",
    "patch_class",
    "class_id",
    "patch_kind",
    "sample_index",
    "y0",
    "x0",
    "y1",
    "x1",
    "source_height",
    "source_width",
    "breast_y0",
    "breast_x0",
    "extraction_scale",
    "roi_overlap",
    "roi_coverage",
    "patch_lesion_fraction",
    "union_roi_overlap_px",
    "tissue_fraction",
    "fallback_reason",
)
EXCLUSION_COLUMNS = (
    "patient_id",
    "development_split",
    "n_train_images",
    "n_val_images",
    "n_test_images",
    "reason",
)


@dataclass(frozen=True)
class PatchExtractionConfig:
    """Registered Stage 0 extraction controls."""

    seed: int = 42
    patch_size: int = 224
    lesion_patches_per_roi: int = 10
    background_patches_per_roi: int = 10
    min_roi_overlap: float = 0.9
    min_background_tissue_fraction: float = 0.5
    max_sampling_attempts: int = 5000
    use_clahe: bool = True

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.lesion_patches_per_roi <= 0:
            raise ValueError("lesion_patches_per_roi must be positive")
        if self.background_patches_per_roi <= 0:
            raise ValueError("background_patches_per_roi must be positive")
        for name, value in (
            ("min_roi_overlap", self.min_roi_overlap),
            ("min_background_tissue_fraction", self.min_background_tissue_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_sampling_attempts <= 0:
            raise ValueError("max_sampling_attempts must be positive")


def load_patch_extraction_config(
    path: Path,
) -> tuple[Path, Path, Path, Path, PatchExtractionConfig]:
    """Load the registered Stage 0 paths and sampling controls from TOML."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    unknown_sections = set(raw) - {"paths", "extraction"}
    if unknown_sections:
        raise ValueError(f"Unknown Stage 0 config sections: {sorted(unknown_sections)}")
    paths = raw.get("paths", {})
    required_paths = {"metadata_dir", "splits_dir", "raw_root", "out_dir"}
    missing_paths = required_paths - set(paths)
    if missing_paths:
        raise ValueError(f"Stage 0 config is missing paths: {sorted(missing_paths)}")
    unknown_paths = set(paths) - required_paths
    if unknown_paths:
        raise ValueError(f"Unknown Stage 0 path keys: {sorted(unknown_paths)}")
    extraction = raw.get("extraction", {})
    allowed_extraction = set(PatchExtractionConfig.__dataclass_fields__)
    unknown_extraction = set(extraction) - allowed_extraction
    if unknown_extraction:
        raise ValueError(
            f"Unknown Stage 0 extraction keys: {sorted(unknown_extraction)}"
        )
    config = PatchExtractionConfig(**extraction)
    config.validate()
    return (
        Path(paths["metadata_dir"]),
        Path(paths["splits_dir"]),
        Path(paths["raw_root"]),
        Path(paths["out_dir"]),
        config,
    )


@dataclass(frozen=True, order=True)
class PatchBox:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


@dataclass(frozen=True)
class Overlap:
    score: float
    roi_coverage: float
    patch_fraction: float
    intersection_px: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _patch_id(
    seed: int,
    split: str,
    image_id: str,
    roi_mask_id: str,
    patch_kind: str,
    sample_index: int,
) -> str:
    value = f"{seed}|{split}|{image_id}|{roi_mask_id}|{patch_kind}|{sample_index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _box_from_center(
    cy: int, cx: int, patch_size: int, shape: tuple[int, int]
) -> PatchBox | None:
    height, width = shape
    if height < patch_size or width < patch_size:
        return None
    y0 = int(np.clip(cy - patch_size // 2, 0, height - patch_size))
    x0 = int(np.clip(cx - patch_size // 2, 0, width - patch_size))
    return PatchBox(y0, x0, y0 + patch_size, x0 + patch_size)


def overlap_with_roi(mask: np.ndarray, box: PatchBox) -> Overlap:
    """Return the registered patch/ROI overlap components for ``box``."""
    binary = np.asarray(mask) > 0
    roi_area = int(binary.sum())
    if roi_area == 0:
        return Overlap(0.0, 0.0, 0.0, 0)
    patch = binary[box.y0 : box.y1, box.x0 : box.x1]
    intersection = int(patch.sum())
    patch_area = int((box.y1 - box.y0) * (box.x1 - box.x0))
    coverage = intersection / roi_area
    fraction = intersection / patch_area
    return Overlap(max(coverage, fraction), coverage, fraction, intersection)


def _candidate_centres(
    mask: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> Iterable[tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if not len(ys):
        return
    # The centroid is a useful deterministic first candidate.  Subsequent
    # centres come from actual ROI pixels, as in the reference sampler.
    yield int(np.rint(ys.mean())), int(np.rint(xs.mean()))
    indices = rng.integers(0, len(ys), size=max(n - 1, 0))
    for index in indices:
        yield int(ys[index]), int(xs[index])


def sample_lesion_boxes(
    roi_mask: np.ndarray,
    config: PatchExtractionConfig,
    rng: np.random.Generator,
) -> list[tuple[PatchBox, Overlap, str]]:
    """Sample lesion boxes, retaining an explicit fallback when needed.

    Accepted boxes meet ``min_roi_overlap``.  If fewer unique qualifying boxes
    exist after the registered attempt budget, the best remaining candidates
    are used and marked rather than silently lowering the cutoff.
    """
    binary = np.asarray(roi_mask) > 0
    if not binary.any():
        return []
    if min(binary.shape) < config.patch_size:
        return []

    candidates: dict[PatchBox, Overlap] = {}
    for cy, cx in _candidate_centres(binary, rng, config.max_sampling_attempts):
        box = _box_from_center(cy, cx, config.patch_size, binary.shape)
        if box is not None and box not in candidates:
            candidates[box] = overlap_with_roi(binary, box)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1].score, item[0].y0, item[0].x0),
    )
    accepted = [item for item in ranked if item[1].score >= config.min_roi_overlap]
    selected = accepted[: config.lesion_patches_per_roi]
    reason = ""
    if len(selected) < config.lesion_patches_per_roi:
        reason = (
            f"insufficient_overlap_{config.min_roi_overlap:.3f}:"
            f"{len(selected)}/{config.lesion_patches_per_roi}"
        )
        selected_ids = {box for box, _ in selected}
        selected.extend(
            item
            for item in ranked
            if item[0] not in selected_ids
        )
        selected = selected[: config.lesion_patches_per_roi]
    return [(box, overlap, reason if overlap.score < config.min_roi_overlap else "") for box, overlap in selected]


def sample_background_boxes(
    union_roi_mask: np.ndarray,
    tissue_mask: np.ndarray,
    n_patches: int,
    config: PatchExtractionConfig,
    rng: np.random.Generator,
) -> list[tuple[PatchBox, float]]:
    """Sample unique breast-tissue boxes with zero overlap with every ROI."""
    if union_roi_mask.shape != tissue_mask.shape:
        raise ValueError("union ROI and tissue masks must have identical geometry")
    height, width = union_roi_mask.shape
    if min(height, width) < config.patch_size:
        return []

    boxes: dict[PatchBox, float] = {}
    for _ in range(config.max_sampling_attempts):
        if len(boxes) >= n_patches:
            break
        y0 = int(rng.integers(0, height - config.patch_size + 1))
        x0 = int(rng.integers(0, width - config.patch_size + 1))
        box = PatchBox(y0, x0, y0 + config.patch_size, x0 + config.patch_size)
        if box in boxes:
            continue
        roi_crop = union_roi_mask[box.y0 : box.y1, box.x0 : box.x1]
        if np.asarray(roi_crop).any():
            continue
        tissue_fraction = float(
            (tissue_mask[box.y0 : box.y1, box.x0 : box.x1] > 0).mean()
        )
        if tissue_fraction < config.min_background_tissue_fraction:
            continue
        boxes[box] = tissue_fraction
    return list(boxes.items())


def _read_locked_split_frames(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the three existing manifests without altering them."""
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required locked split manifest not found: {path}")
        frame = pd.read_csv(path)
        missing = {"patient_id", "image_id"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames[split] = frame
    return frames


def test_overlap_exclusion_ledger(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """List development patients quarantined because they also occur in test.

    Train/validation overlap remains a hard error because there is no valid
    precedence rule between those two development roles.  A train/test or
    validation/test collision is resolved conservatively by retaining the
    patient only in test and recording every removed development image.
    """
    patients = {
        split: set(frame["patient_id"].astype(str))
        for split, frame in frames.items()
    }
    train_val = patients["train"] & patients["val"]
    if train_val:
        examples = ", ".join(sorted(train_val)[:5])
        raise ValueError(
            "Patient leakage between locked train and val manifests: " + examples
        )

    rows: list[dict[str, object]] = []
    test_patients = patients["test"]
    for development_split in ("train", "val"):
        for patient_id in sorted(patients[development_split] & test_patients):
            rows.append(
                {
                    "patient_id": patient_id,
                    "development_split": development_split,
                    "n_train_images": int(
                        (frames["train"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "n_val_images": int(
                        (frames["val"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "n_test_images": int(
                        (frames["test"]["patient_id"].astype(str) == patient_id).sum()
                    ),
                    "reason": "patient_id_also_present_in_locked_test",
                }
            )
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


def _read_split_assignments(splits_dir: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return clean development assignments, quarantining test collisions."""
    frames = _read_locked_split_frames(splits_dir)
    exclusions = test_overlap_exclusion_ledger(frames)
    excluded = set(exclusions["patient_id"].astype(str))
    if excluded:
        LOGGER.warning(
            "Quarantining %d development patient(s) also present in locked test: %s",
            len(excluded),
            ", ".join(sorted(excluded)[:10]),
        )
    patient_split: dict[str, str] = {}
    image_ids: dict[str, set[str]] = {}
    for split, original in frames.items():
        frame = original
        if split in ("train", "val"):
            frame = frame[~frame["patient_id"].astype(str).isin(excluded)]
        image_ids[split] = set(frame["image_id"].astype(str))
        if split in ("train", "val"):
            for patient_id in frame["patient_id"].astype(str).unique():
                patient_split[patient_id] = split
    return patient_split, image_ids


def _write_quarantined_whole_image_splits(
    frames: dict[str, pd.DataFrame], exclusions: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """Write non-destructive whole-image splits for the later matched ablation."""
    excluded = set(exclusions["patient_id"].astype(str))
    clean_dir = out_dir / "whole_image_splits"
    clean_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    clean_frames: dict[str, pd.DataFrame] = {}
    for split, original in frames.items():
        frame = original.copy()
        if split in ("train", "val"):
            frame = frame[~frame["patient_id"].astype(str).isin(excluded)]
        frame = frame.reset_index(drop=True)
        clean_frames[split] = frame
        path = clean_dir / f"{split}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)
    clean_patients = {
        split: set(frame["patient_id"].astype(str))
        for split, frame in clean_frames.items()
    }
    if any(
        clean_patients[left] & clean_patients[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Quarantined whole-image splits are still not patient-disjoint")
    return outputs


def build_lesion_source_manifest(
    metadata_dir: Path, dicom_root: Path, splits_dir: Path
) -> pd.DataFrame:
    """Reconstruct all train/validation abnormalities under locked assignments."""
    patient_split, split_image_ids = _read_split_assignments(Path(splits_dir))
    resolver = DICOMPathResolver(Path(dicom_root))
    parts = []
    for kind in ("mass", "calc"):
        path = Path(metadata_dir) / f"{kind}_case_description_train_set.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Official training metadata not found: {path}")
        parts.append(_build_dataframe(path, Path(dicom_root), resolver))
    source = pd.concat(parts, ignore_index=True)
    source["split"] = source["patient_id"].astype(str).map(patient_split)
    source = source[source["split"].isin(("train", "val"))].copy()
    if source.empty:
        raise ValueError("No official ROI rows matched the locked train/val patients")
    wrong_split = source.apply(
        lambda row: str(row["image_id"]) not in split_image_ids[str(row["split"])],
        axis=1,
    )
    if wrong_split.any():
        examples = source.loc[wrong_split, ["patient_id", "image_id", "split"]].head()
        raise ValueError(
            "Official ROI metadata does not match the locked image assignments:\n"
            + examples.to_string(index=False)
        )
    if source["roi_mask_id"].isna().any():
        raise ValueError("One or more train/validation abnormality rows has no ROI mask")
    if source["roi_mask_id"].astype(str).duplicated().any():
        raise ValueError("ROI mask identifiers must be unique in the source manifest")
    return source.sort_values(["split", "patient_id", "image_id", "roi_mask_id"]).reset_index(drop=True)


def _class_name(pathology: object, lesion_type: object) -> str:
    malignant = str(pathology).upper() == "MALIGNANT"
    kind = str(lesion_type).lower()
    if kind == "calc":
        kind = "calcification"
    if kind not in ("mass", "calcification"):
        raise ValueError(f"Unsupported lesion type for patch learning: {lesion_type!r}")
    return f"{'malignant' if malignant else 'benign'}_{kind}"


def _load_roi(path: Path, source_shape: tuple[int, int], crop_box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.asarray(pydicom.dcmread(str(path)).pixel_array)
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != source_shape:
        mask = cv2.resize(
            mask,
            (source_shape[1], source_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    y0, y1, x0, x1 = crop_box
    return mask[y0:y1, x0:x1]


def _record(
    *,
    patch_id: str,
    patch_path: Path,
    source_row: pd.Series,
    patch_class: str,
    patch_kind: str,
    sample_index: int,
    box: PatchBox,
    source_shape: tuple[int, int],
    breast_box: tuple[int, int, int, int],
    overlap: Overlap,
    union_overlap_px: int,
    tissue_fraction: float,
    fallback_reason: str,
) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "patch_path": patch_path.as_posix(),
        "patient_id": str(source_row["patient_id"]),
        "image_id": str(source_row["image_id"]),
        "roi_mask_id": str(source_row["roi_mask_id"]),
        "split": str(source_row["split"]),
        "patch_class": patch_class,
        "class_id": CLASS_TO_ID[patch_class],
        "patch_kind": patch_kind,
        "sample_index": sample_index,
        "y0": box.y0,
        "x0": box.x0,
        "y1": box.y1,
        "x1": box.x1,
        "source_height": source_shape[0],
        "source_width": source_shape[1],
        "breast_y0": breast_box[0],
        "breast_x0": breast_box[2],
        "extraction_scale": 1.0,
        "roi_overlap": overlap.score,
        "roi_coverage": overlap.roi_coverage,
        "patch_lesion_fraction": overlap.patch_fraction,
        "union_roi_overlap_px": union_overlap_px,
        "tissue_fraction": tissue_fraction,
        "fallback_reason": fallback_reason,
    }


def validate_patch_manifest(
    manifest: pd.DataFrame,
    *,
    train_patients: set[str] | None = None,
    val_patients: set[str] | None = None,
) -> None:
    """Enforce the Stage 0 split, geometry, label, and overlap invariants."""
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"Patch manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Patch manifest is empty")
    if not set(manifest["split"]).issubset({"train", "val"}):
        raise ValueError("Patch manifest may contain only train and val rows")
    split_counts = manifest.groupby("patient_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("A patient occurs in more than one patch split")
    if train_patients is not None:
        actual = set(manifest.loc[manifest["split"] == "train", "patient_id"])
        if not actual.issubset(train_patients):
            raise ValueError("Patch train rows contain patients outside locked train")
    if val_patients is not None:
        actual = set(manifest.loc[manifest["split"] == "val", "patient_id"])
        if not actual.issubset(val_patients):
            raise ValueError("Patch val rows contain patients outside locked val")
    if not set(manifest["patch_class"]).issubset(PATCH_CLASSES):
        raise ValueError("Patch manifest contains an unknown class")
    expected_ids = manifest["patch_class"].map(CLASS_TO_ID)
    if not expected_ids.equals(manifest["class_id"].astype(int)):
        raise ValueError("Patch class names and IDs do not match")
    geometry_ok = (
        (manifest["y0"] >= 0)
        & (manifest["x0"] >= 0)
        & (manifest["y1"] <= manifest["source_height"])
        & (manifest["x1"] <= manifest["source_width"])
        & ((manifest["y1"] - manifest["y0"]) > 0)
        & ((manifest["x1"] - manifest["x0"]) > 0)
    )
    if not geometry_ok.all():
        raise ValueError("Patch coordinates fall outside source-image bounds")
    background = manifest["patch_kind"] == "background"
    if (manifest.loc[background, "union_roi_overlap_px"] != 0).any():
        raise ValueError("A background patch overlaps a known ROI")
    if manifest["patch_id"].duplicated().any():
        raise ValueError("Patch IDs must be unique")


def validate_class_coverage(
    manifest: pd.DataFrame, *, min_examples_per_class: int = 25
) -> None:
    """Require all five classes in both splits and enough rows for visual QA."""
    for split in ("train", "val"):
        classes = set(manifest.loc[manifest["split"] == split, "patch_class"])
        missing = set(PATCH_CLASSES) - classes
        if missing:
            raise ValueError(
                f"Patch {split} split is missing classes: {sorted(missing)}"
            )
    counts = manifest["patch_class"].value_counts()
    too_small = {
        patch_class: int(counts.get(patch_class, 0))
        for patch_class in PATCH_CLASSES
        if int(counts.get(patch_class, 0)) < min_examples_per_class
    }
    if too_small:
        raise ValueError(
            "Not enough patches for the required per-class QA grids: "
            f"{too_small}; need at least {min_examples_per_class}"
        )


def _write_qa_grids(manifest: pd.DataFrame, out_dir: Path, n_per_class: int = 25) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for patch_class in PATCH_CLASSES:
        subset = manifest[manifest["patch_class"] == patch_class].copy()
        if subset.empty:
            continue
        # Surface difficult examples: fallbacks first, then lowest overlap for
        # lesions or lowest tissue fraction for background.
        subset["has_fallback"] = subset["fallback_reason"].fillna("").ne("")
        metric = "tissue_fraction" if patch_class == "background" else "roi_overlap"
        subset = subset.sort_values(["has_fallback", metric], ascending=[False, True])
        subset = subset.head(n_per_class)
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), subset.iterrows()):
            patch = np.load(out_dir / str(row["patch_path"]))
            ax.imshow(patch, cmap="gray", vmin=0.0, vmax=1.0)
            score = row[metric]
            ax.set_title(f"{row['patch_id'][:8]}\n{metric}={score:.3f}", fontsize=7)
            if row["fallback_reason"]:
                ax.add_patch(
                    Rectangle(
                        (1, 1),
                        patch.shape[1] - 2,
                        patch.shape[0] - 2,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
            ax.axis("off")
        fig.suptitle(f"Stage 0 QA: {patch_class} (hardest available examples)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(qa_dir / f"{patch_class}.png", dpi=180)
        plt.close(fig)


def _summary(
    manifest: pd.DataFrame, issues: Sequence[dict[str, object]]
) -> dict[str, object]:
    counts_by_split_and_class: dict[str, int] = {}
    grouped = manifest.groupby(["split", "patch_class"]).size()
    for key, count in grouped.items():
        split, patch_class = cast(tuple[object, object], key)
        counts_by_split_and_class[f"{split}/{patch_class}"] = int(count)
    return {
        "n_patches": len(manifest),
        "n_patients": int(manifest["patient_id"].nunique()),
        "counts_by_split_and_class": counts_by_split_and_class,
        "patients_by_split": {
            str(split): int(count)
            for split, count in manifest.groupby("split")["patient_id"].nunique().items()
        },
        "n_fallback_patches": int(manifest["fallback_reason"].fillna("").ne("").sum()),
        "n_issues": len(issues),
        "issues": list(issues),
    }


def generate_patch_manifests(
    source: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    config: PatchExtractionConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract patches and return the combined manifest plus QA summary."""
    config.validate()
    out_dir = Path(out_dir)
    records: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []

    for image_id, image_rows in tqdm(
        source.groupby("image_id", sort=True), desc="Extracting patch images"
    ):
        image_rows = image_rows.sort_values("roi_mask_id").reset_index(drop=True)
        split = str(image_rows.iloc[0]["split"])
        if image_rows["split"].nunique() != 1:
            raise ValueError(f"Image {image_id} occurs in multiple splits")
        image_path = _find_dicom(Path(raw_root), str(image_id))
        if image_path is None:
            issues.append({"image_id": str(image_id), "reason": "missing_image_dicom"})
            continue
        raw = load_dicom(image_path)
        image, tissue, breast_box = preprocess_aligned_array(
            raw, use_clahe=config.use_clahe
        )
        roi_masks: dict[str, np.ndarray] = {}
        for _, row in image_rows.iterrows():
            roi_id = str(row["roi_mask_id"])
            roi_path = _find_dicom(Path(raw_root), roi_id)
            if roi_path is None:
                issues.append(
                    {"image_id": str(image_id), "roi_mask_id": roi_id, "reason": "missing_roi_dicom"}
                )
                continue
            mask = _load_roi(roi_path, raw.shape, breast_box)
            if not mask.any():
                issues.append(
                    {"image_id": str(image_id), "roi_mask_id": roi_id, "reason": "empty_roi_after_breast_crop"}
                )
                continue
            roi_masks[roi_id] = mask
        if not roi_masks:
            continue
        union_mask = np.logical_or.reduce(list(roi_masks.values()))

        valid_rows = image_rows[image_rows["roi_mask_id"].astype(str).isin(roi_masks)]
        for _, row in valid_rows.iterrows():
            roi_id = str(row["roi_mask_id"])
            patch_class = _class_name(row["pathology"], row["lesion_type"])
            sampled = sample_lesion_boxes(
                roi_masks[roi_id],
                config,
                _stable_rng(config.seed, split, image_id, roi_id, "lesion"),
            )
            if len(sampled) < config.lesion_patches_per_roi:
                issues.append(
                    {
                        "image_id": str(image_id),
                        "roi_mask_id": roi_id,
                        "reason": "insufficient_unique_lesion_patches",
                        "found": len(sampled),
                        "required": config.lesion_patches_per_roi,
                    }
                )
            for sample_index, (box, overlap, fallback_reason) in enumerate(sampled):
                patch_id = _patch_id(config.seed, split, str(image_id), roi_id, "lesion", sample_index)
                relative = Path("patches") / split / patch_class / f"{patch_id}.npy"
                destination = out_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, image[box.y0 : box.y1, box.x0 : box.x1].astype(np.float32))
                union_overlap = int(union_mask[box.y0 : box.y1, box.x0 : box.x1].sum())
                tissue_fraction = float((tissue[box.y0 : box.y1, box.x0 : box.x1] > 0).mean())
                records.append(
                    _record(
                        patch_id=patch_id,
                        patch_path=relative,
                        source_row=row,
                        patch_class=patch_class,
                        patch_kind="lesion",
                        sample_index=sample_index,
                        box=box,
                        source_shape=image.shape,
                        breast_box=breast_box,
                        overlap=overlap,
                        union_overlap_px=union_overlap,
                        tissue_fraction=tissue_fraction,
                        fallback_reason=fallback_reason,
                    )
                )

        n_background = len(valid_rows) * config.background_patches_per_roi
        backgrounds = sample_background_boxes(
            union_mask,
            tissue,
            n_background,
            config,
            _stable_rng(config.seed, split, image_id, "background"),
        )
        if len(backgrounds) < n_background:
            issues.append(
                {
                    "image_id": str(image_id),
                    "reason": "insufficient_unique_background_patches",
                    "found": len(backgrounds),
                    "required": n_background,
                }
            )
        for sample_index, (box, tissue_fraction) in enumerate(backgrounds):
            anchor = valid_rows.iloc[sample_index % len(valid_rows)]
            roi_id = str(anchor["roi_mask_id"])
            patch_id = _patch_id(config.seed, split, str(image_id), roi_id, "background", sample_index)
            relative = Path("patches") / split / "background" / f"{patch_id}.npy"
            destination = out_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, image[box.y0 : box.y1, box.x0 : box.x1].astype(np.float32))
            union_overlap = int(
                union_mask[box.y0 : box.y1, box.x0 : box.x1].sum()
            )
            records.append(
                _record(
                    patch_id=patch_id,
                    patch_path=relative,
                    source_row=anchor,
                    patch_class="background",
                    patch_kind="background",
                    sample_index=sample_index,
                    box=box,
                    source_shape=image.shape,
                    breast_box=breast_box,
                    overlap=Overlap(0.0, 0.0, 0.0, 0),
                    union_overlap_px=union_overlap,
                    tissue_fraction=tissue_fraction,
                    fallback_reason="",
                )
            )

    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    validate_patch_manifest(manifest)
    return manifest, _summary(manifest, issues)


def main(
    metadata_dir: Path,
    splits_dir: Path,
    raw_root: Path,
    out_dir: Path,
    config: PatchExtractionConfig,
) -> None:
    """Build, validate, freeze, and visually audit Stage 0 patch data."""
    setup_logging()
    frames = _read_locked_split_frames(splits_dir)
    exclusions = test_overlap_exclusion_ledger(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclusion_path = out_dir / "excluded-test-overlap-patients.csv"
    exclusions.to_csv(exclusion_path, index=False, lineterminator="\n")
    clean_split_paths = _write_quarantined_whole_image_splits(
        frames, exclusions, out_dir
    )
    source = build_lesion_source_manifest(metadata_dir, raw_root, splits_dir)
    combined, summary = generate_patch_manifests(source, raw_root, out_dir, config)
    validate_class_coverage(combined)
    summary["test_overlap_quarantine"] = {
        "n_patients": int(exclusions["patient_id"].nunique()),
        "n_train_images_removed": int(exclusions["n_train_images"].sum()),
        "n_val_images_removed": int(exclusions["n_val_images"].sum()),
        "patients": sorted(exclusions["patient_id"].astype(str).unique()),
    }

    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        frame = combined[combined["split"] == split].reset_index(drop=True)
        split_frames[split] = frame
        frame.to_csv(out_dir / f"{split}.csv", index=False, lineterminator="\n")
    source.to_csv(out_dir / "lesion-sources.csv", index=False, lineterminator="\n")
    (out_dir / "qa-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_qa_grids(combined, out_dir)

    inputs = {
        str(path): _sha256_file(path)
        for path in [
            *(splits_dir / f"{split}.csv" for split in ("train", "val", "test")),
            *(metadata_dir / f"{kind}_case_description_train_set.csv" for kind in ("mass", "calc")),
        ]
    }
    outputs = {
        name: _sha256_file(out_dir / name)
        for name in (
            "train.csv",
            "val.csv",
            "lesion-sources.csv",
            "qa-summary.json",
            "excluded-test-overlap-patients.csv",
        )
    }
    outputs.update(
        {
            path.relative_to(out_dir).as_posix(): _sha256_file(path)
            for path in clean_split_paths
        }
    )
    patch_digest = hashlib.sha256()
    for relative in sorted(combined["patch_path"].astype(str)):
        patch_digest.update(relative.encode("utf-8"))
        patch_digest.update(b"\0")
        patch_digest.update(_sha256_file(out_dir / relative).encode("ascii"))
        patch_digest.update(b"\n")
    lock = {
        "schema_version": 1,
        "config": asdict(config),
        "source_hashes": inputs,
        "output_hashes": outputs,
        "patch_tree_sha256": patch_digest.hexdigest(),
        "n_patch_files": len(combined),
        "n_quarantined_test_overlap_patients": int(
            exclusions["patient_id"].nunique()
        ),
        "quarantined_test_overlap_patients": sorted(
            exclusions["patient_id"].astype(str).unique()
        ),
        "test_images_masks_or_labels_used": False,
        "test_manifest_role": "patient/image exclusion and source hash only",
        "overlap_definition": "max(intersection/roi_area, intersection/patch_area)",
        "coordinate_system": "native-resolution breast crop; y/x end-exclusive",
    }
    (out_dir / "manifest-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    LOGGER.info(
        "Stage 0 complete: %d patches from %d patients; outputs in %s",
        len(combined),
        combined["patient_id"].nunique(),
        out_dir,
    )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("configs/patch_learning/stage0.toml"),
    show_default=True,
)
def cli(config_path: Path) -> None:
    metadata_dir, splits_dir, raw_root, out_dir, config = (
        load_patch_extraction_config(config_path)
    )
    main(metadata_dir, splits_dir, raw_root, out_dir, config)


if __name__ == "__main__":
    cli()
