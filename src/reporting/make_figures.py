"""Regenerate every figure in results/figures/ from results/metrics.json."""

import json
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import setup_logging

LOGGER = logging.getLogger(__name__)


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text())


def plot_roc_comparison(metrics_path: Path, out_path: Path) -> None:
    """One ROC curve per model on a shared axis.

    Each run's metrics record carries the full (fpr, tpr) arrays under
    "roc", saved by src.evaluation.evaluate, so this draws the real curve.
    Runs without a "roc" block (e.g. from an older evaluation) are skipped
    with a warning rather than drawn as a fake straight line.
    """
    runs = _load_metrics(metrics_path)["runs"]
    if not runs:
        LOGGER.warning("No runs in %s", metrics_path)
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    drawn = 0
    for r in runs:
        roc = r.get("roc")
        if not roc:
            LOGGER.warning(
                "Run %s has no ROC arrays; re-run evaluation to include it",
                r.get("model"),
            )
            continue
        auc = r["test"]["auc"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{r['model']} AUC={auc:.3f}")
        drawn += 1
    if drawn == 0:
        LOGGER.warning("No runs with ROC arrays in %s", metrics_path)
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_subset(
    metrics_path: Path, out_path: Path, members: list[str], title: str
) -> None:
    """ROC overlay restricted to a named subset of runs, in `members` order.

    Used for the headline scratch-vs-transfer figure, which compares only the
    three prototype models rather than every trained backbone.
    """
    by_name = {r.get("model"): r for r in _load_metrics(metrics_path)["runs"]}
    fig, ax = plt.subplots(figsize=(6, 6))
    drawn = 0
    for name in members:
        r = by_name.get(name)
        roc = r.get("roc") if r else None
        if not roc:
            LOGGER.warning("Run %s missing or has no ROC arrays; skipping", name)
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=f"{name} AUC={r['test']['auc']:.3f}")
        drawn += 1
    if drawn == 0:
        LOGGER.warning("No requested runs with ROC arrays in %s", metrics_path)
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(history_path: Path, out_path: Path, title: str) -> None:
    """Training loss (left axis) and validation AUC (right axis) vs. epoch.

    Reads the per-epoch history written by src.training.train. Validation loss
    is not recorded by the trainer, so the loss axis shows training loss only.
    """
    if not history_path.exists():
        LOGGER.warning("History %s not found; skipping %s", history_path, out_path)
        return
    history = json.loads(history_path.read_text())
    if not history:
        LOGGER.warning("History %s is empty; skipping", history_path)
        return
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    val_auc = [e["val_auc"] for e in history]
    fig, ax_loss = plt.subplots(figsize=(7, 4.5))
    ax_loss.plot(epochs, train_loss, color="tab:red", label="train loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")
    ax_auc = ax_loss.twinx()
    ax_auc.plot(epochs, val_auc, color="tab:blue", label="val AUC")
    ax_auc.set_ylabel("Validation AUC", color="tab:blue")
    ax_auc.tick_params(axis="y", labelcolor="tab:blue")
    best = max(range(len(val_auc)), key=val_auc.__getitem__)
    ax_auc.axvline(epochs[best], linestyle=":", color="tab:blue", alpha=0.6)
    ax_loss.set_title(
        f"{title} (best val AUC {val_auc[best]:.3f} @ epoch {epochs[best]})"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(metrics_path: Path, out_dir: Path) -> None:
    """Annotated confusion matrix heatmap, one file per run.

    Layout is [[TN, FP], [FN, TP]] (rows = actual, cols = predicted).
    Title carries the headline AUC / sensitivity / specificity so the figure
    is self-contained without referring back to the metrics file.
    """
    runs = _load_metrics(metrics_path)["runs"]
    class_labels = ["Benign", "Malignant"]
    for r in runs:
        cm = r.get("test", {}).get("confusion")
        if cm is None:
            LOGGER.warning("Run %s has no confusion matrix; skipping", r.get("model"))
            continue
        cm = np.array(cm, dtype=int)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        t = r["test"]
        ax.set_title(
            f"{r['model']}\n"
            f"AUC {t['auc']:.3f}  sens {t['sensitivity']:.2f}  spec {t['specificity']:.2f}"
        )
        out_path = out_dir / f"confusion_{r['model']}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def plot_density_strata(metrics_path: Path, out_dir: Path) -> None:
    """AUC, sensitivity, and specificity broken down by BIRADS breast density.

    Three side-by-side bar charts share the same density axis (D1 to D4).
    Each model is a separate bar group so models can be compared at a glance
    within each density category.
    """
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("density_strata")]
    if not runs:
        LOGGER.warning("No density_strata data; skipping")
        return

    metrics_spec = [
        ("auc", "AUC"),
        ("sens", "Sensitivity"),
        ("spec", "Specificity"),
    ]
    densities = [1, 2, 3, 4]
    x = np.arange(len(densities))
    bar_width = 0.8 / max(len(runs), 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, label) in zip(axes, metrics_spec):
        for i, r in enumerate(runs):
            by_density = {s["density"]: s for s in r["density_strata"]}
            values = [by_density.get(d, {}).get(key, float("nan")) for d in densities]
            offset = (i - len(runs) / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=r["model"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{d}" for d in densities])
        ax.set_xlabel("BIRADS density")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by density")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8)
        ax.legend(fontsize=7)

    fig.suptitle("Density-stratified metrics", y=1.02)
    out_path = out_dir / "density_strata.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def plot_reliability(metrics_path: Path, out_dir: Path) -> None:
    """Reliability diagram for each run showing calibration after temperature scaling.

    Points on the diagonal are perfectly calibrated. The title shows the
    temperature T and ECE before/after scaling so the figure records whether
    scaling actually improved calibration.
    """
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("calibration")]
    if not runs:
        LOGGER.warning("No calibration data; skipping reliability diagrams")
        return

    for r in runs:
        cal = r["calibration"]
        rel = cal["reliability"]
        pred_mean = rel["pred_mean"]
        obs_mean = rel["obs_mean"]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration"
        )
        ax.plot(pred_mean, obs_mean, marker="o", label="model (post-scaling)")
        for pm, om in zip(pred_mean, obs_mean):
            ax.annotate(
                f"{pm:.2f}",
                (pm, om),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title(
            f"{r['model']} reliability\n"
            f"T={cal['temperature']:.3f}  "
            f"ECE before={cal['ece_before']:.3f}  after={cal['ece_after']:.3f}"
        )
        ax.legend()
        out_path = out_dir / f"reliability_{r['model']}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def plot_decision_curves(metrics_path: Path, out_dir: Path) -> None:
    """Decision curve analysis: net benefit vs probability threshold.

    The treat-all and treat-none reference lines are drawn from the first run
    (they are identical across runs for the same test set). Each model's curve
    is overlaid so relative benefit is easy to read.
    """
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("decision_curve")]
    if not runs:
        LOGGER.warning("No decision_curve data; skipping")
        return

    first_dc = runs[0]["decision_curve"]
    thresholds = first_dc["thresholds"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        thresholds,
        first_dc["treat_all"],
        linestyle="--",
        color="grey",
        label="treat all",
    )
    ax.plot(
        thresholds,
        first_dc["treat_none"],
        linestyle=":",
        color="black",
        label="treat none",
    )
    for r in runs:
        dc = r["decision_curve"]
        ax.plot(dc["thresholds"], dc["model"], label=r["model"])
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.legend()
    out_path = out_dir / "decision_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def plot_gradcam_roi(metrics_path: Path, out_dir: Path) -> None:
    """GradCAM vs lesion ROI spatial alignment metrics for all / TP / FN subsets.

    Three bar charts show pointing game rate, mean IoU, and normalised centroid
    distance for each subset. The TP/FN split reveals whether the model looks at
    the right region on cases it gets right versus wrong.
    """
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("gradcam_roi")]
    if not runs:
        LOGGER.warning("No gradcam_roi data; skipping")
        return

    metrics_spec = [
        ("pointing_game", "Pointing game"),
        ("iou_mean", "Mean IoU"),
        ("centroid_mean", "Centroid distance (norm.)"),
    ]
    subsets = ["all", "tp", "fn"]
    subset_labels = ["All malignant", "TP", "FN"]
    x = np.arange(len(subsets))
    bar_width = 0.8 / max(len(runs), 1)

    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(13, 4))
    for ax, (key, label) in zip(axes, metrics_spec):
        for i, r in enumerate(runs):
            gr = r["gradcam_roi"]
            values = [gr.get(s, {}).get(key, float("nan")) for s in subsets]
            offset = (i - len(runs) / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=r["model"])
        ax.set_xticks(x)
        ax.set_xticklabels(subset_labels)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("GradCAM vs lesion ROI alignment", y=1.02)
    out_path = out_dir / "gradcam_roi.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def main(
    metrics_path: Path, figures_dir: Path, models_dir: Path = Path("models")
) -> None:
    setup_logging()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_comparison(metrics_path, figures_dir / "roc_comparison.png")
    plot_roc_subset(
        metrics_path,
        figures_dir / "roc_baseline_vs_vgg.png",
        members=["baseline", "vgg16_scratch", "vgg16_imagenet"],
        title="ROC: baseline vs VGG-16 (scratch vs ImageNet)",
    )
    plot_learning_curve(
        models_dir / "baseline.history.json",
        figures_dir / "baseline_curves.png",
        "Baseline CNN",
    )
    plot_learning_curve(
        models_dir / "vgg16_imagenet.history.json",
        figures_dir / "transfer_curves.png",
        "VGG-16 + ImageNet (transfer)",
    )
    plot_confusion_matrices(metrics_path, figures_dir)
    plot_density_strata(metrics_path, figures_dir)
    plot_reliability(metrics_path, figures_dir)
    plot_decision_curves(metrics_path, figures_dir)
    plot_gradcam_roi(metrics_path, figures_dir)


@click.command()
@click.option(
    "--metrics",
    "metrics_path",
    type=click.Path(path_type=Path),
    default=Path("results/metrics.json"),
    show_default=True,
    help="Path to results/metrics.json produced by evaluate.py.",
)
@click.option(
    "--figures-dir",
    type=click.Path(path_type=Path),
    default=Path("results/figures"),
    show_default=True,
    help="Output directory for figures.",
)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=Path("models"),
    show_default=True,
    help="Directory holding the per-run *.history.json files.",
)
def cli(metrics_path: Path, figures_dir: Path, models_dir: Path) -> None:
    main(metrics_path, figures_dir, models_dir)


if __name__ == "__main__":
    cli()
