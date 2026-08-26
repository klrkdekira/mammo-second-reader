"""Tests for config loading, schema validation, and the ensemble loader."""

import textwrap
from pathlib import Path

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
train_csv = "train.csv"
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
    assert str(cfg.train_csv) == "train.csv"
    assert str(cfg.test_csv) == "test.csv"
    assert str(cfg.val_csv) == "val.csv"
    assert cfg.batch_size == 32


def test_ensemble_requires_val_csv(tmp_path):
    no_val = VALID_ENSEMBLE.replace('val_csv = "val.csv"\n', "")
    with pytest.raises(KeyError, match="val_csv"):
        load_ensemble_config(_write(tmp_path, no_val))


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


@pytest.mark.parametrize(
    ("original_path", "extension_path", "extension_name"),
    [
        (
            "configs/regularised_base.toml",
            "configs/regularised_extensions/regularised_base_120.toml",
            "regularised_base_120",
        ),
        (
            "configs/regularised_label_smooth.toml",
            "configs/regularised_extensions/regularised_label_smooth_120.toml",
            "regularised_label_smooth_120",
        ),
        (
            "configs/regularised_mixup.toml",
            "configs/regularised_extensions/regularised_mixup_120.toml",
            "regularised_mixup_120",
        ),
    ],
)
def test_regularised_extension_changes_only_budget_and_name(
    original_path, extension_path, extension_name
):
    original = load_config(original_path)
    extension = load_config(extension_path)

    assert extension.run_name == extension_name
    assert extension.train.epochs == 120
    assert extension.seed == original.seed
    assert extension.output_dir == original.output_dir
    assert extension.data == original.data
    assert extension.model == original.model

    original_train = vars(original.train)
    extension_train = vars(extension.train)
    assert {
        key: value for key, value in extension_train.items() if key != "epochs"
    } == {key: value for key, value in original_train.items() if key != "epochs"}


def test_transfer_configs_are_matched_to_the_448_reference():
    """Patch pretraining must be the only factor that differs between the arms.

    A drifted batch size or schedule here would make the paired AUC difference
    uninterpretable, so the match is asserted rather than trusted to review.
    """
    reference = load_config("configs/vgg16_highres_448.toml")
    control = load_config("configs/patch_learning/vgg16_imagenet_448_quarantined.toml")
    candidate = load_config("configs/patch_learning/vgg16_patch_imagenet_448.toml")

    for arm in (control, candidate):
        assert arm.seed == reference.seed
        assert arm.data == reference.data
        assert vars(arm.train) == vars(reference.train)
        # Patch-learning runs must not overwrite the locked milestone evidence.
        assert arm.output_dir == Path("models/patch_learning")

    assert control.run_name == "vgg16_imagenet_448_quarantined"
    assert candidate.run_name == "vgg16_patch_imagenet_448"
    # The two arms differ in initialisation and nothing else.
    assert control.model.init_from_patch_checkpoint is None
    assert candidate.model.init_from_patch_checkpoint == Path(
        "models/patch_learning/vgg16_patch.pt"
    )
    assert vars(control.model) == {
        **vars(candidate.model),
        "init_from_patch_checkpoint": None,
    }


def test_shipped_patch_config_loads():
    from src.config import load_patch_config

    cfg = load_patch_config("configs/patch_learning/vgg16_patch.toml")
    assert cfg.run_name == "vgg16_patch"
    assert cfg.seed == 42
    assert cfg.output_dir == Path("models/patch_learning")
    assert cfg.data.exclusion_test_csv == Path("manifests/cbis-ddsm/test.csv")
    assert cfg.train.selection_metric == "macro_f1"


def test_patch_amendment_changes_only_augmentation():
    """Amendment 1 must isolate the augmentation level.

    A drifted learning rate or epoch budget here would make the validation
    comparison against vgg16_patch uninterpretable.
    """
    from src.config import load_patch_config

    original = load_patch_config("configs/patch_learning/vgg16_patch.toml")
    amended = load_patch_config("configs/patch_learning/vgg16_patch_aug.toml")

    assert original.data.augment == "light"
    assert amended.data.augment == "default"
    assert amended.run_name == "vgg16_patch_aug"
    assert amended.seed == original.seed
    assert amended.output_dir == original.output_dir
    assert vars(amended.train) == vars(original.train)
    assert vars(amended.model) == vars(original.model)

    original_data = {k: v for k, v in vars(original.data).items() if k != "augment"}
    amended_data = {k: v for k, v in vars(amended.data).items() if k != "augment"}
    assert amended_data == original_data
