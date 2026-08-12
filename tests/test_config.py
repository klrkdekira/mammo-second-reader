"""Tests for config loading, schema validation, and the ensemble loader."""

import textwrap

import pytest

from src.config import Config, EnsembleConfig, load_config, load_ensemble_config

VALID_SINGLE = """
seed = 42
run_name = "baseline"
output_dir = "models"

[data]
train_csv = "train.csv"
val_csv = "val.csv"
test_csv = "test.csv"
image_root = "images"
image_size = 224

[model]
name = "baseline"

[train]
epochs = 10
batch_size = 32
"""

VALID_ENSEMBLE = """
seed = 42
run_name = "ensemble"
output_dir = "models"
members = ["vgg16_imagenet", "resnet50_imagenet"]

[data]
val_csv = "val.csv"
test_csv = "test.csv"
image_root = "images"
image_size = 224
"""


def _write(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


def test_load_config_parses_single_model(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_SINGLE))
    assert isinstance(cfg, Config)
    assert cfg.run_name == "baseline"
    assert cfg.model.name == "baseline"
    assert cfg.train.epochs == 10


def test_unknown_data_key_fails_loudly(tmp_path):
    bad = VALID_SINGLE.replace("image_size = 224", "augmnet = 'heavy'")
    with pytest.raises(ValueError, match="augmnet"):
        load_config(_write(tmp_path, bad))


def test_unknown_top_level_key_fails_loudly(tmp_path):
    bad = VALID_SINGLE.replace(
        'run_name = "baseline"', 'run_naem = "baseline"\nrun_name = "baseline"'
    )
    with pytest.raises(ValueError, match="run_naem"):
        load_config(_write(tmp_path, bad))


def test_load_config_rejects_ensemble_config(tmp_path):
    # An ensemble config carries a top-level `members` list and no [model]
    # section, so the single-model loader must fail loudly rather than
    # silently mis-parse it. Unknown-key validation catches `members` first.
    with pytest.raises(ValueError, match="members"):
        load_config(_write(tmp_path, VALID_ENSEMBLE))


def test_load_ensemble_config_parses(tmp_path):
    cfg = load_ensemble_config(_write(tmp_path, VALID_ENSEMBLE))
    assert isinstance(cfg, EnsembleConfig)
    assert cfg.members == ["vgg16_imagenet", "resnet50_imagenet"]
    assert str(cfg.test_csv) == "test.csv"
    assert str(cfg.val_csv) == "val.csv"
    assert cfg.batch_size == 32


def test_ensemble_missing_val_csv_is_none(tmp_path):
    no_val = VALID_ENSEMBLE.replace('val_csv = "val.csv"\n', "")
    cfg = load_ensemble_config(_write(tmp_path, no_val))
    assert cfg.val_csv is None


def test_ensemble_unknown_key_fails_loudly(tmp_path):
    bad = VALID_ENSEMBLE.replace("members =", "membrs =")
    with pytest.raises(ValueError, match="membrs"):
        load_ensemble_config(_write(tmp_path, bad))


def test_shipped_ensemble_config_loads():
    # The real repo config must parse under the loader that ensemble.py uses.
    cfg = load_ensemble_config("configs/ensemble.toml")
    assert cfg.run_name == "ensemble"
    assert len(cfg.members) == 4


def test_focused_highres_config_is_controlled():
    focused = load_config("configs/vgg16_highres_448.toml")
    reference = load_config("configs/vgg16_transfer.toml")

    assert focused.run_name == "vgg16_imagenet_448"
    assert focused.data.image_size == 448
    assert focused.train.batch_size == 8
    assert focused.model == reference.model
    assert focused.seed == reference.seed
    assert focused.train.epochs == reference.train.epochs
    assert focused.train.scheduler == reference.train.scheduler
    assert focused.train.early_stop_patience == reference.train.early_stop_patience
