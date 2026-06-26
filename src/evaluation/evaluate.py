"""CLI evaluation entry-point. Reads a trained checkpoint and appends a
metrics record to results/metrics.json.
"""

import dataclasses
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.config import Config, get_device, load_config, setup_logging
from src.data.augment import val_augment
from src.data.dataset import MammogramDataset
from src.evaluation.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
)
from src.evaluation.decision_curve import decision_curve
from src.evaluation.density_strata import metrics_by_density
from src.evaluation.metrics import evaluate
from src.models import build_model

LOGGER = logging.getLogger(__name__)

METRICS_PATH = Path("results/metrics.json")


def _load_threshold(cfg: Config) -> float:
    sidecar = cfg.output_dir / f"{cfg.run_name}.threshold.json"
    return float(json.loads(sidecar.read_text())["youden_j"])


def _predict_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Collect labels and raw logits over a loader.

    The base panel only needs probabilities, but temperature scaling fits on
    logits, so evaluation collects logits and derives probabilities from them.
    """
    model.eval()
    ys, ls = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            ls.append(model(x).cpu().numpy().ravel())
            ys.append(y.cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(ls)


def _gradcam_roi_panel(
    model: torch.nn.Module,
    test_ds,
    model_name: str,
    y_prob: np.ndarray,
    threshold: float,
    device: torch.device,
) -> dict | None:
    """Quantitative Grad-CAM-vs-ROI stats over the malignant test cases.

    For every malignant case, score the best single model's Grad-CAM heatmap
    against the lesion ROI mask three ways (pointing game, IoU, centroid
    distance; see gradcam_roi), then report them for all malignant cases and
    split by predicted-correct (TP) vs predicted-incorrect (FN). The expected
    pattern is high agreement on TP and low on FN; its absence flags a model
    that is right for the wrong reasons. Returns None when no malignant case
    has a usable ROI mask (so evaluation degrades gracefully).
    """
    from src.evaluation.gradcam import TARGET_LAYERS, compute_gradcam
    from src.evaluation.gradcam_roi import grad_cam_subset_stats

    target_layer = TARGET_LAYERS.get(model_name.lower())
    if target_layer is None:
        return None
    mal_idx = np.where(test_ds.df["label"].values == 1)[0]
    cams: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    correct: list[bool] = []
    for i in mal_idx:
        image, _ = test_ds[int(i)]
        cam = compute_gradcam(model, image.unsqueeze(0).to(device), target_layer)
        roi = test_ds.load_roi(int(i), cam.shape)
        if roi is None:
            continue
        cams.append(cam)
        rois.append(roi)
        # malignant case predicted positive => true positive (correct)
        correct.append(bool(y_prob[i] >= threshold))
    if not cams:
        return None
    correct_arr = np.array(correct, dtype=bool)
    n = len(cams)
    return {
        "n_malignant_scored": n,
        "all": grad_cam_subset_stats(cams, rois, np.ones(n, dtype=bool)),
        "tp": grad_cam_subset_stats(cams, rois, correct_arr),
        "fn": grad_cam_subset_stats(cams, rois, ~correct_arr),
    }


def _append_record(record: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"runs": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    data.setdefault("runs", []).append(record)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(config_path: Path) -> None:
    setup_logging()
    cfg = load_config(config_path)
    device = get_device()

    weights_path = cfg.output_dir / f"{cfg.run_name}.pt"

    model = build_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        dropout_conv=cfg.model.dropout_conv,
        dropout_head=cfg.model.dropout_head,
        head_hidden=cfg.model.head_hidden,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    test_ds = MammogramDataset(
        cfg.data.test_csv,
        cfg.data.image_root,
        transform=val_augment(cfg.data.image_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2
    )
    y_true, test_logits = _predict_logits(model, test_loader, device)
    y_prob = 1.0 / (1.0 + np.exp(-test_logits))
    threshold = _load_threshold(cfg)
    panel = evaluate(y_true, y_prob, threshold=threshold)
    LOGGER.info("Test panel: %s", panel)

    record: dict = {
        "model": cfg.run_name,
        "val_threshold": threshold,
        "test": {**dataclasses.asdict(panel), "confusion": panel.confusion.tolist()},
    }

    # ROC points for the model-comparison figure (only AUC was stored before,
    # so plot_roc_comparison had to fake the curve).
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    record["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Density-stratified metrics. Strata with fewer than min_n cases are skipped.
    record["density_strata"] = metrics_by_density(
        test_ds.df, y_prob, threshold
    ).to_dict(orient="records")

    # Temperature scaling on validation logits. Calibrated probabilities feed the decision curve.
    cal_prob = y_prob
    if Path(cfg.data.val_csv).exists():
        val_ds = MammogramDataset(
            cfg.data.val_csv,
            cfg.data.image_root,
            transform=val_augment(cfg.data.image_size),
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2
        )
        val_true, val_logits = _predict_logits(model, val_loader, device)
        temperature = fit_temperature(
            torch.tensor(val_logits, dtype=torch.float32),
            torch.tensor(val_true, dtype=torch.float32),
        )
        cal_prob = 1.0 / (1.0 + np.exp(-test_logits / temperature))
        centres, pred_mean, obs_mean = reliability_bins(cal_prob, y_true)
        record["calibration"] = {
            "temperature": temperature,
            "ece_before": expected_calibration_error(y_prob, y_true),
            "ece_after": expected_calibration_error(cal_prob, y_true),
            "reliability": {
                "bin_centre": centres.tolist(),
                "pred_mean": pred_mean.tolist(),
                "obs_mean": obs_mean.tolist(),
            },
        }
    else:
        LOGGER.warning(
            "val_csv %s not found; skipping calibration and using "
            "uncalibrated probabilities for the decision curve",
            cfg.data.val_csv,
        )

    # Decision-curve analysis on calibrated probabilities.
    record["decision_curve"] = {
        k: np.asarray(v).tolist() for k, v in decision_curve(y_true, cal_prob).items()
    }

    # Quantitative Grad-CAM vs ROI. Computed on malignant test cases when
    # roi_mask_id is present and masks are on disk, skipped otherwise.
    gradcam_roi = _gradcam_roi_panel(
        model, test_ds, cfg.model.name, y_prob, threshold, device
    )
    if gradcam_roi is not None:
        record["gradcam_roi"] = gradcam_roi
    else:
        LOGGER.warning(
            "Grad-CAM-ROI (novelty A) skipped: no usable ROI masks "
            "for the malignant test cases (need a roi_mask_id column "
            "plus mask files; re-run make_splits and cache masks)."
        )

    _append_record(record)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="YAML experiment config.",
)
def cli(config_path: Path) -> None:
    main(config_path)


if __name__ == "__main__":
    cli()
