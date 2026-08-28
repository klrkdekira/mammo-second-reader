"""Patch-classifier tests."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_patch_config
from src.data.patch_manifest import CLASS_TO_ID, PATCH_CLASSES
from src.evaluation.metrics import evaluate_patches
from src.models import build_model
from src.training.loss import make_patch_criterion


def _write_patches(root, frame):
    rng = np.random.default_rng(0)
    for path in frame["patch_path"]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, rng.random((224, 224)).astype(np.float32))


def _balanced_manifest(split, patients):
    rows = []
    for patient in patients:
        for patch_class in PATCH_CLASSES:
            patch_id = f"{patient}_{split}_{patch_class}"
            rows.append(
                {
                    "patch_id": patch_id,
                    "patch_path": f"patches/{split}/{patch_id}.npy",
                    "patient_id": patient,
                    "split": split,
                    "patch_class": patch_class,
                    "class_id": CLASS_TO_ID[patch_class],
                }
            )
    return pd.DataFrame(rows)


def _patch_project(tmp_path, train_patients, val_patients, test_patients=()):
    data = tmp_path / "data"
    data.mkdir()
    train = _balanced_manifest("train", train_patients)
    val = _balanced_manifest("val", val_patients)
    train.to_csv(data / "train.csv", index=False)
    val.to_csv(data / "val.csv", index=False)
    _write_patches(data, train)
    _write_patches(data, val)
    pd.DataFrame({"patient_id": list(test_patients) or ["P_TEST"]}).to_csv(
        data / "test.csv", index=False
    )
    config = tmp_path / "patch.toml"
    config.write_text(
        f"""
seed = 7
run_name = "unit_patch"
output_dir = "{tmp_path / "models"}"

[data]
train_csv = "{data / "train.csv"}"
val_csv = "{data / "val.csv"}"
patch_root = "{data}"
patch_size = 224
augment = "light"
num_workers = 0
exclusion_test_csv = "{data / "test.csv"}"

[model]
name = "vgg16"
pretrained = false

[train]
epochs = 2
batch_size = 4
stage1_epochs = 1
early_stop_patience = 2
"""
    )
    return config


def test_build_model_default_head_is_unchanged():
    model = build_model("vgg16", pretrained=False)
    assert model(torch.zeros(2, 1, 224, 224)).shape == (2, 1)


def test_build_model_five_class_head():
    model = build_model("vgg16", pretrained=False, num_classes=len(PATCH_CLASSES))
    assert model(torch.zeros(2, 1, 224, 224)).shape == (2, 5)


def test_build_model_rejects_zero_classes():
    with pytest.raises(ValueError, match="num_classes"):
        build_model("vgg16", pretrained=False, num_classes=0)


def test_patch_criterion_weights_are_inverse_frequency(tmp_path):
    csv = tmp_path / "train.csv"
    pd.DataFrame(
        {"class_id": [0] * 40 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 30}
    ).to_csv(csv, index=False)
    criterion = make_patch_criterion(csv)
    weights = criterion.weight
    assert weights is not None
    assert weights[1] > weights[4] > weights[0]
    assert float(weights.mean()) == pytest.approx(1.0)


def test_patch_criterion_rejects_missing_class(tmp_path):
    csv = tmp_path / "train.csv"
    pd.DataFrame({"class_id": [0, 1, 2, 3]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no examples of class id"):
        make_patch_criterion(csv)


def test_patch_config_rejects_test_csv(tmp_path):
    config = _patch_project(tmp_path, ["P_1"], ["P_2"])
    config.write_text(
        config.read_text().replace("patch_size = 224", 'test_csv = "nope.csv"')
    )
    with pytest.raises(ValueError, match="Unknown key"):
        load_patch_config(config)


def test_patch_config_rejects_mixup(tmp_path):
    config = _patch_project(tmp_path, ["P_1"], ["P_2"])
    config.write_text(config.read_text() + "mixup_alpha = 0.2\n")
    with pytest.raises(ValueError, match="mixup_alpha is not supported"):
        load_patch_config(config)


def test_patch_config_rejects_unknown_selection_metric(tmp_path):
    config = _patch_project(tmp_path, ["P_1"], ["P_2"])
    config.write_text(config.read_text() + 'selection_metric = "auc"\n')
    with pytest.raises(ValueError, match="selection_metric"):
        load_patch_config(config)


def test_preflight_rejects_shared_patient(tmp_path):
    from src.training.train_patch import _preflight

    config = _patch_project(tmp_path, ["P_1", "P_2"], ["P_2"])
    with pytest.raises(ValueError, match="Patient leakage"):
        _preflight(load_patch_config(config))


def test_preflight_rejects_locked_test_patient(tmp_path):
    from src.training.train_patch import _preflight

    config = _patch_project(tmp_path, ["P_1"], ["P_2"], test_patients=["P_1"])
    with pytest.raises(ValueError, match="locked test patient"):
        _preflight(load_patch_config(config))


def test_preflight_accepts_disjoint_folds(tmp_path):
    from src.training.train_patch import _preflight

    config = _patch_project(tmp_path, ["P_1"], ["P_2"], test_patients=["P_9"])
    _preflight(load_patch_config(config))


def test_patch_training_writes_checkpoint_and_report(tmp_path):
    from src.training import train_patch

    config = _patch_project(tmp_path, ["P_1", "P_2"], ["P_3"], test_patients=["P_9"])
    train_patch.main(config)

    models = tmp_path / "models"
    assert (models / "unit_patch.pt").is_file()

    history = json.loads((models / "unit_patch.history.json").read_text())
    assert len(history) == 2
    assert {"epoch", "train_loss", "val_macro_f1", "selected"} <= set(history[0])

    report = json.loads((models / "unit_patch.patch-metrics.json").read_text())
    assert report["class_names"] == list(PATCH_CLASSES)
    assert report["selection_metric"] == "macro_f1"
    assert np.array(report["validation"]["confusion"]).shape == (5, 5)
    assert set(report["validation"]["per_class_sensitivity"]) == set(PATCH_CLASSES)
    assert set(report["validation"]["lesion_pair_confusion"]) == {
        "calcification",
        "mass",
    }


def test_patch_training_is_reproducible(tmp_path):
    from src.training import train_patch

    histories = []
    for index in range(2):
        run = tmp_path / f"run{index}"
        run.mkdir()
        config = _patch_project(run, ["P_1", "P_2"], ["P_3"], test_patients=["P_9"])
        train_patch.main(config)
        histories.append(
            json.loads((run / "models" / "unit_patch.history.json").read_text())
        )
    first = [entry["train_loss"] for entry in histories[0]]
    second = [entry["train_loss"] for entry in histories[1]]
    assert first == pytest.approx(second)


def test_evaluate_patches_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="class ids"):
        evaluate_patches(np.array([0, 5]), np.array([0, 1]), PATCH_CLASSES)


def test_evaluate_patches_perfect_predictions():
    ids = np.arange(len(PATCH_CLASSES))
    panel = evaluate_patches(ids, ids, PATCH_CLASSES)
    assert panel.macro_f1 == pytest.approx(1.0)
    assert panel.balanced_accuracy == pytest.approx(1.0)
    assert panel.lesion_pair_confusion["mass"]["n_predicted_off_pair"] == 0.0


def test_patch_backbone_transfer_copies_features_only(tmp_path):
    from src.models.transfer import load_patch_backbone

    patch_model = build_model("vgg16", pretrained=False, num_classes=len(PATCH_CLASSES))
    checkpoint = tmp_path / "patch.pt"
    torch.save(patch_model.state_dict(), checkpoint)

    whole = build_model("vgg16", pretrained=False)
    head_before = whole.state_dict()["backbone.classifier.6.4.weight"].clone()
    fc_before = whole.state_dict()["backbone.classifier.0.weight"].clone()

    summary = load_patch_backbone(whole, checkpoint, "vgg16")

    after = whole.state_dict()
    source = patch_model.state_dict()
    assert summary["n_tensors_copied"] == 26
    assert summary["n_tensors_left_as_built"] == 8
    for key in source:
        if key.startswith("backbone.features."):
            assert torch.equal(after[key], source[key]), key
    assert after["backbone.classifier.6.4.weight"].shape == (1, 256)
    assert torch.equal(after["backbone.classifier.6.4.weight"], head_before)
    assert torch.equal(after["backbone.classifier.0.weight"], fc_before)


def test_patch_backbone_transfer_accepts_a_bare_backbone_checkpoint(tmp_path):
    from src.models.transfer import load_patch_backbone

    patch_model = build_model("vgg16", pretrained=False, num_classes=len(PATCH_CLASSES))
    stripped = {
        key.removeprefix("backbone."): value
        for key, value in patch_model.state_dict().items()
    }
    checkpoint = tmp_path / "bare.pt"
    torch.save(stripped, checkpoint)

    summary = load_patch_backbone(
        build_model("vgg16", pretrained=False), checkpoint, "vgg16"
    )
    assert summary["n_tensors_copied"] == 26


def test_patch_backbone_transfer_refuses_wrong_architecture(tmp_path):
    from src.models.transfer import load_patch_backbone

    vgg19_patch = build_model("vgg19", pretrained=False, num_classes=len(PATCH_CLASSES))
    checkpoint = tmp_path / "vgg19.pt"
    torch.save(vgg19_patch.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="Cannot transfer patch weights"):
        load_patch_backbone(build_model("vgg16", pretrained=False), checkpoint, "vgg16")


def test_patch_backbone_transfer_refuses_empty_checkpoint(tmp_path):
    from src.models.transfer import load_patch_backbone

    checkpoint = tmp_path / "empty.pt"
    torch.save({}, checkpoint)
    with pytest.raises(ValueError, match="missing from checkpoint.*features.0.weight"):
        load_patch_backbone(build_model("vgg16", pretrained=False), checkpoint, "vgg16")


def test_whole_image_training_consumes_a_patch_checkpoint(tmp_path):
    from src.models.transfer import load_patch_backbone

    patch_model = build_model("vgg16", pretrained=False, num_classes=len(PATCH_CLASSES))
    checkpoint = tmp_path / "vgg16_patch.pt"
    torch.save(patch_model.state_dict(), checkpoint)

    whole = build_model("vgg16", pretrained=False)
    load_patch_backbone(whole, checkpoint, "vgg16")
    assert whole(torch.zeros(1, 1, 448, 448)).shape == (1, 1)
