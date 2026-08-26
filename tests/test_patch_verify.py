"""Tests for the Stage 0 patch-tree verifier.

The patch tree is an 11 GB artefact of a 17-hour extraction that cannot be
regenerated cheaply and is not in version control, so it travels between hosts
by file copy. These tests cover the failure modes that a copy can introduce:
a missing file, a truncated file, a changed manifest, and a wrong file count.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.patch_verify import patch_tree_digest, verify_patch_data


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage0_tree(tmp_path, n_train=4, n_val=2):
    """Build a miniature frozen Stage 0 tree with a matching lock."""
    root = tmp_path / "data"
    (root / "patches" / "train").mkdir(parents=True)
    (root / "patches" / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)

    frames = {}
    for split, count in (("train", n_train), ("val", n_val)):
        rows = []
        for index in range(count):
            relative = f"patches/{split}/{split}_{index}.npy"
            np.save(root / relative, rng.random((4, 4)).astype(np.float32))
            rows.append({"patch_id": f"{split}_{index}", "patch_path": relative})
        frames[split] = pd.DataFrame(rows)
        frames[split].to_csv(root / f"{split}.csv", index=False)

    for name in ("lesion-sources.csv", "qa-summary.json"):
        (root / name).write_text("placeholder\n")

    paths = [
        str(value)
        for frame in frames.values()
        for value in frame["patch_path"].astype(str)
    ]
    lock = {
        "output_hashes": {
            name: _sha256(root / name)
            for name in (
                "train.csv",
                "val.csv",
                "lesion-sources.csv",
                "qa-summary.json",
            )
        },
        "n_patch_files": len(paths),
        "patch_tree_sha256": patch_tree_digest(root, paths),
    }
    (root / "manifest-lock.json").write_text(json.dumps(lock, indent=2))
    return root


def test_verify_accepts_an_intact_tree(tmp_path):
    root = _stage0_tree(tmp_path)
    report = verify_patch_data(root)
    assert report["n_patch_files"] == 6
    assert report["locked_outputs_verified"] == 4


def test_verify_rejects_a_missing_patch(tmp_path):
    root = _stage0_tree(tmp_path)
    (root / "patches" / "train" / "train_0.npy").unlink()
    with pytest.raises(FileNotFoundError, match="1 patch file"):
        verify_patch_data(root)


def test_verify_rejects_a_corrupted_patch(tmp_path):
    """A truncated transfer is the failure a file count would not catch."""
    root = _stage0_tree(tmp_path)
    target = root / "patches" / "val" / "val_0.npy"
    target.write_bytes(target.read_bytes()[:-8])
    with pytest.raises(ValueError, match="digest .* does not match the frozen"):
        verify_patch_data(root)


def test_verify_rejects_an_edited_manifest(tmp_path):
    root = _stage0_tree(tmp_path)
    with (root / "train.csv").open("a") as handle:
        handle.write("extra,row\n")
    with pytest.raises(ValueError, match="train.csv"):
        verify_patch_data(root)


def test_verify_rejects_a_wrong_patch_count(tmp_path):
    root = _stage0_tree(tmp_path)
    lock_path = root / "manifest-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["n_patch_files"] = 99
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="lock records 99"):
        verify_patch_data(root)


def test_verify_requires_a_lock(tmp_path):
    root = _stage0_tree(tmp_path)
    (root / "manifest-lock.json").unlink()
    with pytest.raises(FileNotFoundError, match="Stage 0 lock not found"):
        verify_patch_data(root)
