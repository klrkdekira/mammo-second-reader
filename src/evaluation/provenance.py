"""Record the files and code used for each result."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import string
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

PROVENANCE_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def describe_file(
    path: Path, root: Path, *, required: bool = True
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Evidence input not found: {path}")
        return {"path": _display_path(path, root), "exists": False}
    return {
        "path": _display_path(path, root),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _combined_fingerprint(descriptors: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(descriptors, key=lambda value: str(value["path"])):
        digest.update(str(item["path"]).encode())
        digest.update(str(item["sha256"]).encode())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "pydicom", "scikit-learn", "torch", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_run_provenance(
    *,
    config_path: Path,
    checkpoint_paths: Iterable[Path],
    manifest_paths: Iterable[Path],
    threshold_path: Path | None = None,
    extra: dict[str, object] | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Record what was used for one evaluation run."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    preprocessing_paths = [
        root / "src/data/preprocessing.py",
        root / "src/data/dataset.py",
        root / "src/data/augment.py",
        root / "src/data/dicom_to_png.py",
        root / "src/data/cache_roi_masks.py",
    ]
    evaluation_paths = [
        root / "src/evaluation/calibration.py",
        root / "src/evaluation/metrics.py",
        root / "src/evaluation/evaluate.py",
        root / "src/evaluation/provenance.py",
        root / "src/evaluation/results_io.py",
        root / "src/evaluation/freeze.py",
        root / "src/training/ensemble.py",
        root / "src/models/ensemble.py",
    ]
    preprocessing = [describe_file(path, root) for path in preprocessing_paths]
    evaluation = [describe_file(path, root) for path in evaluation_paths]
    commit_file = root / ".cuda-commit"
    synced_commit = commit_file.read_text().strip() if commit_file.is_file() else ""
    valid_synced_commit = len(synced_commit) == 40 and all(
        character in string.hexdigits for character in synced_commit
    )
    if valid_synced_commit:
        commit: str | None = synced_commit
        status: str | None = ""
        source = "cuda_sync"
    else:
        commit = _git_value(root, "rev-parse", "HEAD")
        status = _git_value(
            root,
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "src",
            "configs",
            "tests",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
        source = "git"
    record: dict[str, object] = {
        "version": PROVENANCE_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": commit,
            "dirty_evidence_files": bool(status),
            "evidence_status": status or "",
            "source": source,
        },
        "config": describe_file(Path(config_path), root),
        "checkpoints": [describe_file(Path(path), root) for path in checkpoint_paths],
        "manifests": [describe_file(Path(path), root) for path in manifest_paths],
        "threshold_sidecar": (
            describe_file(Path(threshold_path), root)
            if threshold_path is not None
            else None
        ),
        "code": {
            "preprocessing_fingerprint": _combined_fingerprint(preprocessing),
            "evaluation_fingerprint": _combined_fingerprint(evaluation),
            "preprocessing_files": preprocessing,
            "evaluation_files": evaluation,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    if extra:
        record["experiment"] = extra
    return record
