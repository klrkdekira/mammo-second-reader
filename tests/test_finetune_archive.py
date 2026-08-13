"""Tests for the Fine-tune upload fixture builder."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.make_finetune_archive import verify_archive, write_archive


def _splits(tmp_path, n_train=12, n_val=8, nested=True):
    """A miniature CBIS-DDSM-shaped splits directory with a matching cache."""
    splits_dir = tmp_path / "training"
    cache_dir = tmp_path / "cache"
    splits_dir.mkdir()
    cache_dir.mkdir()

    def make(split, n):
        rows = []
        for i in range(n):
            # Real CBIS-DDSM ids are nested paths; that nesting is what forces
            # the flattening this builder exists to do.
            image_id = (
                f"Mass-Training_P_{i:05d}_LEFT_CC/1.3.6.1.{i}/1-1"
                if nested
                else f"img_{split}_{i}"
            )
            target = cache_dir / f"{image_id}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, np.full((32, 32), 0.5, dtype=np.float32))
            rows.append(
                {
                    "image_id": image_id,
                    "label": i % 2,
                    "patient_id": f"P_{i:05d}",
                    "birads_density": (i % 4) + 1,
                    "lesion_type": "mass" if i % 2 else "calcification",
                }
            )
        pd.DataFrame(rows).to_csv(splits_dir / f"{split}.csv", index=False)

    make("train", n_train)
    make("val", n_val)
    return splits_dir, cache_dir


def _build(tmp_path, **overrides):
    splits_dir, cache_dir = _splits(tmp_path)
    output = tmp_path / "fixture.zip"
    kwargs = {
        "splits_dir": splits_dir,
        "cache_dir": cache_dir,
        "raw_root": cache_dir,
        "n_train": 8,
        "n_val": 4,
        "seed": 1,
        "source": "npy",
    }
    kwargs.update(overrides)
    return output, write_archive(output, **kwargs)


def test_archive_contains_both_manifests_and_flat_images(tmp_path):
    output, summary = _build(tmp_path)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()

    assert "train.csv" in names
    assert "val.csv" in names
    assert all("/" not in name for name in names), "archive must be flat"
    assert len(names) == summary["n_train"] + summary["n_val"] + 2
    assert len(set(names)) == len(names), "no duplicate basenames"


def test_nested_source_ids_are_rewritten_but_traceable(tmp_path):
    output, _ = _build(tmp_path)

    with zipfile.ZipFile(output) as archive:
        train = pd.read_csv(archive.open("train.csv"))

    assert train["image_id"].str.startswith("train_").all()
    assert not train["image_id"].str.contains("/").any()
    assert train["image_id"].is_unique
    # The original identifier survives so a result can be traced back.
    assert train["source_image_id"].str.contains("Mass-Training").all()


def test_both_classes_present_in_each_split(tmp_path):
    _, summary = _build(tmp_path)

    for key in ("train_labels", "val_labels"):
        assert set(summary[key]) == {0, 1}, f"{key} must cover both classes"


def test_carried_metadata_columns_survive(tmp_path):
    output, _ = _build(tmp_path)

    with zipfile.ZipFile(output) as archive:
        val = pd.read_csv(archive.open("val.csv"))

    for column in ("patient_id", "birads_density", "lesion_type"):
        assert column in val.columns


def test_archive_round_trips_through_the_web_unpacker(tmp_path):
    """The fixture must load through the same path the Fine-tune tab uses."""
    output, summary = _build(tmp_path)

    checked = verify_archive(output)

    assert checked["train_rows"] == summary["n_train"]
    assert checked["val_rows"] == summary["n_val"]


def test_single_class_source_is_rejected(tmp_path):
    splits_dir, cache_dir = _splits(tmp_path)
    frame = pd.read_csv(splits_dir / "train.csv")
    frame["label"] = 1
    frame.to_csv(splits_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="only one class"):
        write_archive(
            tmp_path / "bad.zip",
            splits_dir=splits_dir,
            cache_dir=cache_dir,
            raw_root=cache_dir,
            n_train=8,
            n_val=4,
            seed=1,
            source="npy",
        )


def test_missing_cache_is_reported_clearly(tmp_path):
    splits_dir, _ = _splits(tmp_path)

    with pytest.raises(ValueError, match="no source images were found"):
        write_archive(
            tmp_path / "bad.zip",
            splits_dir=splits_dir,
            cache_dir=tmp_path / "does-not-exist",
            raw_root=tmp_path / "does-not-exist",
            n_train=8,
            n_val=4,
            seed=1,
            source="npy",
        )


def test_requesting_fewer_than_two_images_is_rejected(tmp_path):
    splits_dir, cache_dir = _splits(tmp_path)

    with pytest.raises(ValueError, match="at least 2 images"):
        write_archive(
            tmp_path / "bad.zip",
            splits_dir=splits_dir,
            cache_dir=cache_dir,
            raw_root=cache_dir,
            n_train=1,
            n_val=4,
            seed=1,
            source="npy",
        )
