"""Verify a Stage 0 patch tree against its frozen lock.

The 55,619-patch tree is the output of a 17-hour extraction that the patch
contract forbids repeating, and it is too large to keep in version control, so
it moves between machines by file copy. This command proves a copy is intact:
it recomputes every locked output hash and the whole-tree digest and compares
them with `manifest-lock.json`.

Run it after moving the patch data to a new host, before training on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from src.config import setup_logging
from src.data.patch_manifest import _sha256_file

LOGGER = logging.getLogger(__name__)

LOCKED_OUTPUTS = ("train.csv", "val.csv", "lesion-sources.csv", "qa-summary.json")


def patch_tree_digest(data_root: Path, patch_paths: list[str]) -> str:
    """Recompute the frozen whole-tree digest.

    Mirrors the construction in `patch_manifest.generate_patch_manifests`:
    sorted relative path, NUL, the file's hex digest, newline.
    """
    digest = hashlib.sha256()
    for relative in tqdm(sorted(patch_paths), desc="hashing patches", unit="patch"):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(data_root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_patch_data(data_root: Path) -> dict[str, object]:
    """Compare a patch tree with its lock. Raises on any mismatch."""
    data_root = Path(data_root)
    lock_path = data_root / "manifest-lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 0 lock not found: {lock_path}")
    lock = json.loads(lock_path.read_text())

    problems: list[str] = []
    for name in LOCKED_OUTPUTS:
        path = data_root / name
        expected = lock["output_hashes"][name]
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            problems.append(f"{name}: {actual[:12]} != locked {expected[:12]}")
    if problems:
        raise ValueError(
            "Stage 0 manifests do not match their lock: " + "; ".join(problems)
        )

    frames = [pd.read_csv(data_root / name) for name in ("train.csv", "val.csv")]
    patch_paths = [
        str(value) for frame in frames for value in frame["patch_path"].astype(str)
    ]
    expected_count = int(lock["n_patch_files"])
    if len(patch_paths) != expected_count:
        raise ValueError(
            f"Manifests list {len(patch_paths)} patches, lock records {expected_count}"
        )
    absent = [
        relative for relative in patch_paths if not (data_root / relative).is_file()
    ]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} patch file(s) missing under {data_root}; "
            f"first few: {absent[:5]}"
        )

    digest = patch_tree_digest(data_root, patch_paths)
    if digest != lock["patch_tree_sha256"]:
        raise ValueError(
            f"Patch tree digest {digest} does not match the frozen "
            f"{lock['patch_tree_sha256']}"
        )

    return {
        "data_root": str(data_root),
        "n_patch_files": len(patch_paths),
        "patch_tree_sha256": digest,
        "locked_outputs_verified": len(LOCKED_OUTPUTS),
    }


@click.command()
@click.option(
    "--data-root",
    type=click.Path(path_type=Path),
    default=Path("results/patch_learning/data"),
    show_default=True,
)
def cli(data_root: Path) -> None:
    """Verify the frozen Stage 0 patch tree at DATA_ROOT."""
    setup_logging()
    report = verify_patch_data(data_root)
    LOGGER.info(
        "Stage 0 patch data verified: %d patches under %s reproduce the frozen "
        "digest %s, and all %d locked manifests match.",
        report["n_patch_files"],
        report["data_root"],
        report["patch_tree_sha256"],
        report["locked_outputs_verified"],
    )


if __name__ == "__main__":
    cli()
