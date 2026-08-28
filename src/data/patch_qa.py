"""Generate deterministic QA artefacts for frozen patch data."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.dicom_to_png import _find_dicom
from src.data.patch_manifest import (
    MANIFEST_COLUMNS,
    PATCH_CLASSES,
    PatchBox,
    _load_roi,
    _sha256_file,
    load_patch_extraction_config,
    overlap_with_roi,
    validate_patch_manifest,
)
from src.data.preprocessing import load_dicom, preprocess_aligned_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes

LOGGER = logging.getLogger(__name__)

LESION_CLASSES = tuple(name for name in PATCH_CLASSES if name != "background")
SELECTION_COLUMNS = ("audit_group", "selection_index", *MANIFEST_COLUMNS)


@dataclass(frozen=True)
class PatchQAConfig:
    """Patch QA sampling settings."""

    seed: int = 42
    n_per_group: int = 25
    context_margin_patches: float = 1.0

    def validate(self) -> None:
        if self.n_per_group <= 0:
            raise ValueError("n_per_group must be positive")
        if self.context_margin_patches < 0:
            raise ValueError("context_margin_patches must be non-negative")


@dataclass(frozen=True)
class ReviewSource:
    """A preprocessed image and its aligned ROI masks."""

    image: np.ndarray
    roi_masks: dict[str, np.ndarray]
    crop_box: tuple[int, int, int, int]

    @cached_property
    def union_mask(self) -> np.ndarray:
        return np.logical_or.reduce(list(self.roi_masks.values())).astype(np.uint8)


def _stable_rank(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_manifest_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fallback_reason"] = result["fallback_reason"].fillna("")
    for name in (
        "patch_id",
        "patch_path",
        "patient_id",
        "image_id",
        "roi_mask_id",
        "split",
        "patch_class",
        "patch_kind",
        "fallback_reason",
    ):
        result[name] = result[name].astype(str)
    return result


def load_frozen_manifest(data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and hash-check the frozen Stage 0 train/validation manifests."""
    data_root = Path(data_root)
    lock_path = data_root / "manifest-lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 0 lock not found: {lock_path}")
    lock = cast(dict[str, object], json.loads(lock_path.read_text()))
    output_hashes = cast(dict[str, str], lock.get("output_hashes", {}))
    frames: list[pd.DataFrame] = []
    for split in ("train", "val"):
        name = f"{split}.csv"
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen patch manifest not found: {path}")
        expected = output_hashes.get(name)
        if expected is None:
            raise ValueError(f"Stage 0 lock has no hash for {name}")
        if _sha256_file(path) != expected:
            raise ValueError(f"Frozen patch manifest hash mismatch: {path}")
        frames.append(pd.read_csv(path))
    manifest = _normalise_manifest_strings(pd.concat(frames, ignore_index=True))
    validate_patch_manifest(manifest)
    locked_count = lock.get("n_patch_files")
    if not isinstance(locked_count, int):
        raise TypeError("Stage 0 lock has no integer n_patch_files")
    expected_count = locked_count
    if len(manifest) != expected_count:
        raise ValueError(
            f"Manifest contains {len(manifest)} rows; lock expects {expected_count}"
        )
    return manifest, lock


def _one_per_unit(
    frame: pd.DataFrame,
    unit: str,
    *,
    seed: int,
    prefer_low: str | None = None,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_stable_rank"] = [
        _stable_rank(seed, row.patch_id, row.image_id, row.roi_mask_id)
        for row in ranked.itertuples()
    ]
    sort_columns = [unit]
    ascending = [True]
    if prefer_low is not None:
        sort_columns.append(prefer_low)
        ascending.append(True)
    sort_columns.append("_stable_rank")
    ascending.append(True)
    ranked = ranked.sort_values(sort_columns, ascending=ascending)
    return ranked.drop_duplicates(unit, keep="first").drop(columns="_stable_rank")


def _spread(frame: pd.DataFrame, n: int, metric: str) -> pd.DataFrame:
    ordered = frame.sort_values([metric, "patch_id"]).reset_index(drop=True)
    if len(ordered) < n:
        raise ValueError(f"Need {n} review cases but only {len(ordered)} are available")
    if n == 1:
        return ordered.iloc[[len(ordered) // 2]]
    indices = np.linspace(0, len(ordered) - 1, n, dtype=int)
    return ordered.iloc[indices]


def _take_ranked(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f"Need {n} review cases but only {len(frame)} are available")
    ranked = frame.copy()
    ranked["_stable_rank"] = [
        _stable_rank(seed, row.patch_id, row.image_id, row.roi_mask_id)
        for row in ranked.itertuples()
    ]
    return ranked.sort_values("_stable_rank").head(n).drop(columns="_stable_rank")


def _take_partitioned(
    frame: pd.DataFrame,
    n: int,
    *,
    strategy: str,
    seed: int,
    metric: str,
) -> pd.DataFrame:
    """Select deterministically while retaining train/validation coverage."""
    if len(frame) < n:
        raise ValueError(f"Need {n} review cases but only {len(frame)} are available")
    split_counts = frame["split"].value_counts()
    quotas: dict[str, int] = {}
    remaining = n
    available_splits = [split for split in ("train", "val") if split in split_counts]
    for index, split in enumerate(available_splits):
        if index == len(available_splits) - 1:
            quota = remaining
        else:
            quota = round(n * int(split_counts[split]) / len(frame))
            if n >= len(available_splits):
                quota = max(1, quota)
            reserved = len(available_splits) - index - 1
            quota = min(quota, int(split_counts[split]), remaining - reserved)
        quotas[split] = quota
        remaining -= quota
    while remaining > 0:
        progressed = False
        for split in available_splits:
            if quotas[split] < int(split_counts[split]):
                quotas[split] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("Unable to fill deterministic split quotas")

    selected: list[pd.DataFrame] = []
    for split in available_splits:
        subset = frame[frame["split"] == split]
        quota = quotas[split]
        if strategy == "spread":
            selected.append(_spread(subset, quota, metric))
        elif strategy == "lowest":
            selected.append(subset.sort_values([metric, "patch_id"]).head(quota))
        elif strategy == "ranked":
            selected.append(_take_ranked(subset, quota, seed=seed))
        else:
            raise ValueError(f"Unknown review selection strategy: {strategy}")
    return pd.concat(selected, ignore_index=True)


def select_review_cases(manifest: pd.DataFrame, config: PatchQAConfig) -> pd.DataFrame:
    """Select registered QA groups with one patch per ROI or source image."""
    config.validate()
    manifest = _normalise_manifest_strings(manifest)
    fallback = manifest["fallback_reason"].ne("")
    fallback_rois = set(manifest.loc[fallback, "roi_mask_id"])
    groups: list[pd.DataFrame] = []

    for patch_class in LESION_CLASSES:
        class_rows = manifest[manifest["patch_class"] == patch_class]
        representative = class_rows[
            class_rows["fallback_reason"].eq("")
            & ~class_rows["roi_mask_id"].isin(fallback_rois)
        ]
        representative = _one_per_unit(representative, "roi_mask_id", seed=config.seed)
        representative = _take_partitioned(
            representative,
            config.n_per_group,
            strategy="spread",
            seed=config.seed,
            metric="roi_overlap",
        )
        representative.insert(0, "audit_group", f"{patch_class}/representative")
        groups.append(representative)

        fallback_rows = _one_per_unit(
            class_rows[class_rows["fallback_reason"].ne("")],
            "roi_mask_id",
            seed=config.seed,
            prefer_low="roi_overlap",
        )
        fallback_rows = _take_partitioned(
            fallback_rows,
            config.n_per_group,
            strategy="spread",
            seed=config.seed,
            metric="roi_overlap",
        )
        fallback_rows.insert(0, "audit_group", f"{patch_class}/fallback")
        groups.append(fallback_rows)

    background = manifest[manifest["patch_class"] == "background"]
    background = _one_per_unit(
        background, "image_id", seed=config.seed, prefer_low="tissue_fraction"
    )
    difficult_background = _take_partitioned(
        background,
        config.n_per_group,
        strategy="lowest",
        seed=config.seed,
        metric="tissue_fraction",
    )
    representative_pool = background[
        ~background["image_id"].isin(difficult_background["image_id"])
    ]
    representative_background = _take_partitioned(
        representative_pool,
        config.n_per_group,
        strategy="ranked",
        seed=config.seed,
        metric="tissue_fraction",
    )
    representative_background.insert(0, "audit_group", "background/representative")
    groups.append(representative_background)
    difficult_background.insert(0, "audit_group", "background/difficult")
    groups.append(difficult_background)

    selected = pd.concat(groups, ignore_index=True)
    selected.insert(
        1,
        "selection_index",
        selected.groupby("audit_group", sort=False).cumcount(),
    )
    return cast(pd.DataFrame, selected.loc[:, SELECTION_COLUMNS])


def _box_from_row(row: pd.Series) -> PatchBox:
    return PatchBox(int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"]))


def _load_review_source(
    image_rows: pd.DataFrame,
    raw_root: Path,
    *,
    use_clahe: bool,
) -> ReviewSource:
    image_id = str(image_rows.iloc[0]["image_id"])
    image_path = _find_dicom(Path(raw_root), image_id)
    if image_path is None:
        raise FileNotFoundError(f"Source mammogram DICOM not found: {image_id}")
    raw = load_dicom(image_path)
    image, _, crop_box = preprocess_aligned_array(raw, use_clahe=use_clahe)
    expected_shapes = set(
        zip(
            image_rows["source_height"].astype(int),
            image_rows["source_width"].astype(int),
            strict=True,
        )
    )
    if expected_shapes != {image.shape}:
        raise ValueError(
            f"Reconstructed source shape {image.shape} does not match manifest "
            f"for {image_id}: {sorted(expected_shapes)}"
        )
    expected_origins = set(
        zip(
            image_rows["breast_y0"].astype(int),
            image_rows["breast_x0"].astype(int),
            strict=True,
        )
    )
    if expected_origins != {(crop_box[0], crop_box[2])}:
        raise ValueError(
            f"Reconstructed breast crop {crop_box} does not match manifest "
            f"origin for {image_id}: {sorted(expected_origins)}"
        )
    roi_masks: dict[str, np.ndarray] = {}
    for roi_id in sorted(image_rows["roi_mask_id"].astype(str).unique()):
        roi_path = _find_dicom(Path(raw_root), roi_id)
        if roi_path is None:
            raise FileNotFoundError(f"ROI-mask DICOM not found: {roi_id}")
        roi_masks[roi_id] = _load_roi(roi_path, raw.shape, crop_box)
    return ReviewSource(image=image, roi_masks=roi_masks, crop_box=crop_box)


def validate_reconstructed_case(
    row: pd.Series,
    source: ReviewSource,
    data_root: Path,
) -> dict[str, object]:
    """Check a frozen patch against its recorded source geometry."""
    box = _box_from_row(row)
    reconstructed = source.image[box.y0 : box.y1, box.x0 : box.x1]
    patch_path = Path(data_root) / str(row["patch_path"])
    if not patch_path.is_file():
        raise FileNotFoundError(f"Frozen patch not found: {patch_path}")
    frozen = np.load(patch_path)
    if not np.array_equal(reconstructed, frozen):
        raise ValueError(f"Frozen patch does not reconstruct exactly: {patch_path}")

    target_id = str(row["roi_mask_id"])
    if target_id not in source.roi_masks:
        raise ValueError(f"Target ROI {target_id} is absent from reconstructed masks")
    union = source.union_mask
    union_overlap = int(union[box.y0 : box.y1, box.x0 : box.x1].sum())
    if union_overlap != int(row["union_roi_overlap_px"]):
        raise ValueError(
            f"Union-ROI overlap changed for patch {row['patch_id']}: "
            f"{union_overlap} != {row['union_roi_overlap_px']}"
        )

    result: dict[str, object] = {
        "patch_id": str(row["patch_id"]),
        "patch_sha256": _sha256_file(patch_path),
        "exact_pixel_match": True,
        "union_roi_overlap_px": union_overlap,
    }
    if str(row["patch_kind"]) == "lesion":
        actual = overlap_with_roi(source.roi_masks[target_id], box)
        for field, observed in (
            ("roi_overlap", actual.score),
            ("roi_coverage", actual.roi_coverage),
            ("patch_lesion_fraction", actual.patch_fraction),
        ):
            if not np.isclose(observed, float(row[field]), atol=1e-12):
                raise ValueError(
                    f"Reconstructed {field} changed for patch {row['patch_id']}: "
                    f"{observed} != {row[field]}"
                )
        result["roi_overlap"] = actual.score
    elif union_overlap != 0:
        raise ValueError(f"Background review patch overlaps a known ROI: {patch_path}")
    return result


def _context_slices(
    box: PatchBox, shape: tuple[int, int], margin_patches: float
) -> tuple[slice, slice]:
    patch_size = max(box.shape)
    margin = round(patch_size * margin_patches)
    return (
        slice(max(0, box.y0 - margin), min(shape[0], box.y1 + margin)),
        slice(max(0, box.x0 - margin), min(shape[1], box.x1 + margin)),
    )


def _draw_mask(ax: Axes, mask: np.ndarray, colour: str, alpha: float) -> None:
    from matplotlib.colors import ListedColormap

    if not np.asarray(mask).any():
        return
    overlay = np.ma.masked_where(np.asarray(mask) <= 0, np.asarray(mask))
    ax.imshow(
        overlay,
        cmap=ListedColormap([colour]),
        alpha=alpha,
        vmin=0,
        vmax=1,
    )
    ax.contour(np.asarray(mask), levels=[0.5], colors=[colour], linewidths=0.8)


def _render_case(
    context_ax: Axes,
    patch_ax: Axes,
    row: pd.Series,
    source: ReviewSource,
    config: PatchQAConfig,
) -> None:
    from matplotlib.patches import Rectangle

    box = _box_from_row(row)
    ys, xs = _context_slices(box, source.image.shape, config.context_margin_patches)
    context = source.image[ys, xs]
    union_context = source.union_mask[ys, xs]
    target = source.roi_masks[str(row["roi_mask_id"])]
    target_context = target[ys, xs]

    context_ax.imshow(context, cmap="gray", vmin=0.0, vmax=1.0)
    _draw_mask(context_ax, union_context, "magenta", 0.12)
    if str(row["patch_kind"]) == "lesion":
        _draw_mask(context_ax, target_context, "lime", 0.18)
    context_ax.add_patch(
        Rectangle(
            (box.x0 - cast(int, xs.start), box.y0 - cast(int, ys.start)),
            box.x1 - box.x0,
            box.y1 - box.y0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.2,
        )
    )
    context_ax.set_title(f"{row['split']} context", fontsize=6)
    context_ax.axis("off")

    patch = source.image[box.y0 : box.y1, box.x0 : box.x1]
    patch_ax.imshow(patch, cmap="gray", vmin=0.0, vmax=1.0)
    union_patch = source.union_mask[box.y0 : box.y1, box.x0 : box.x1]
    _draw_mask(patch_ax, union_patch, "magenta", 0.12)
    if str(row["patch_kind"]) == "lesion":
        target_patch = target[box.y0 : box.y1, box.x0 : box.x1]
        _draw_mask(patch_ax, target_patch, "lime", 0.18)
        metric = f"overlap={float(row['roi_overlap']):.3f}"
    else:
        metric = f"tissue={float(row['tissue_fraction']):.3f}"
    patch_ax.set_title(f"{str(row['patch_id'])[:8]} {metric}", fontsize=6)
    patch_ax.axis("off")


def _grid_filename(audit_group: str) -> str:
    return audit_group.replace("/", "__") + ".png"


def _render_grid(
    group: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_root: Path,
    data_root: Path,
    destination: Path,
    extraction_use_clahe: bool,
    config: PatchQAConfig,
) -> list[dict[str, object]]:
    import matplotlib.pyplot as plt

    n_rows = int(np.ceil(len(group) / 5))
    fig, axes = plt.subplots(n_rows, 10, figsize=(20, 3.4 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    checks: list[dict[str, object]] = []
    cached_image_id = ""
    cached_source: ReviewSource | None = None
    for position, (_, row) in enumerate(group.iterrows()):
        image_id = str(row["image_id"])
        if image_id != cached_image_id:
            image_rows = manifest[manifest["image_id"] == image_id]
            cached_source = _load_review_source(
                image_rows, raw_root, use_clahe=extraction_use_clahe
            )
            cached_image_id = image_id
        assert cached_source is not None
        checks.append(validate_reconstructed_case(row, cached_source, data_root))
        grid_row, pair_column = divmod(position, 5)
        _render_case(
            axes[grid_row, pair_column * 2],
            axes[grid_row, pair_column * 2 + 1],
            row,
            cached_source,
            config,
        )
    audit_group = str(group.iloc[0]["audit_group"])
    fig.suptitle(
        "Stage 0 supplementary QA: "
        + audit_group
        + "\ncyan=patch, magenta=all known ROIs, green=target ROI",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return checks


def _summary(
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    checks: list[dict[str, object]],
    config: PatchQAConfig,
) -> dict[str, object]:
    fallback = manifest["fallback_reason"].ne("")
    split_counts: dict[str, int] = {}
    grouped = selection.groupby(["audit_group", "split"]).size()
    for key, count in grouped.items():
        group, split = cast(tuple[object, object], key)
        split_counts[f"{group}/{split}"] = int(count)
    return {
        "schema_version": 2,
        "status": "complete",
        "automated_checks_passed": True,
        "config": {
            "seed": config.seed,
            "n_per_group": config.n_per_group,
            "context_margin_patches": config.context_margin_patches,
        },
        "frozen_dataset": {
            "n_patches": len(manifest),
            "n_patients": int(manifest["patient_id"].nunique()),
            "n_fallback_lesion_patches": int(fallback.sum()),
            "n_background_patches": int(
                (manifest["patch_class"] == "background").sum()
            ),
        },
        "selection_counts": {
            str(name): int(count)
            for name, count in selection["audit_group"]
            .value_counts()
            .sort_index()
            .items()
        },
        "selection_split_counts": split_counts,
        "automated_reconstruction": {
            "n_cases": len(checks),
            "n_exact_pixel_matches": sum(
                bool(check["exact_pixel_match"]) for check in checks
            ),
            "n_background_known_roi_overlaps": sum(
                int(cast(int, check["union_roi_overlap_px"])) > 0
                for check in checks
                if "roi_overlap" not in check
            ),
        },
    }


def generate_patch_qa(
    raw_root: Path,
    data_root: Path,
    output_dir: Path,
    extraction_use_clahe: bool,
    config: PatchQAConfig,
) -> None:
    """Generate an atomic, locked QA package without altering frozen Stage 0."""
    setup_logging()
    config.validate()
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"QA output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest, stage0_lock = load_frozen_manifest(data_root)
    selection = select_review_cases(manifest, config)
    groups = list(selection["audit_group"].drop_duplicates())

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        grid_dir = temporary_dir / "grids"
        grid_dir.mkdir()
        selection.to_csv(
            temporary_dir / "selection.csv", index=False, lineterminator="\n"
        )
        checks: list[dict[str, object]] = []
        for audit_group in tqdm(groups, desc="Rendering supplementary patch QA"):
            group = selection[selection["audit_group"] == audit_group]
            checks.extend(
                _render_grid(
                    group,
                    manifest,
                    Path(raw_root),
                    data_root,
                    grid_dir / _grid_filename(audit_group),
                    extraction_use_clahe,
                    config,
                )
            )
        pd.DataFrame(checks).to_csv(
            temporary_dir / "reconstruction-checks.csv",
            index=False,
            lineterminator="\n",
        )
        summary_path = temporary_dir / "qa-review-summary.json"
        summary_path.write_text(
            json.dumps(_summary(manifest, selection, checks, config), indent=2) + "\n"
        )
        outputs = {
            path.relative_to(temporary_dir).as_posix(): _sha256_file(path)
            for path in sorted(temporary_dir.rglob("*"))
            if path.is_file()
        }
        lock = {
            "schema_version": 2,
            "implementation": {
                "patch_qa.py": _sha256_file(Path(__file__)),
                "patch_manifest.py": _sha256_file(
                    Path(__file__).with_name("patch_manifest.py")
                ),
                "preprocessing.py": _sha256_file(
                    Path(__file__).with_name("preprocessing.py")
                ),
            },
            "stage0_manifest_lock_sha256": _sha256_file(
                data_root / "manifest-lock.json"
            ),
            "stage0_patch_tree_sha256": stage0_lock["patch_tree_sha256"],
            "inputs": {
                "train.csv": _sha256_file(data_root / "train.csv"),
                "val.csv": _sha256_file(data_root / "val.csv"),
            },
            "outputs": outputs,
        }
        (temporary_dir / "qa-review-lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n"
        )
        temporary_dir.rename(output_dir)
    LOGGER.info("Stage 0 supplementary QA written to %s", output_dir)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("configs/patch_learning/stage0.toml"),
    show_default=True,
)
@click.option("--n-per-group", type=int, default=25, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Defaults to <Stage 0 out_dir>/qa_review.",
)
def cli(
    config_path: Path,
    n_per_group: int,
    seed: int | None,
    output_dir: Path | None,
) -> None:
    _, _, raw_root, data_root, extraction = load_patch_extraction_config(config_path)
    qa_config = PatchQAConfig(
        seed=extraction.seed if seed is None else seed,
        n_per_group=n_per_group,
    )
    destination = output_dir or data_root / "qa_review"
    generate_patch_qa(
        raw_root,
        data_root,
        destination,
        extraction.use_clahe,
        qa_config,
    )


if __name__ == "__main__":
    cli()
