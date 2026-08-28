"""Regenerate every figure in results/figures/ from results/metrics.json."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from src.config import setup_logging

LOGGER = logging.getLogger(__name__)


def _legend_outside_right(ax: Axes, *, fontsize: float = 8) -> Legend:
    """Place a dense legend beside, rather than over, a plotting area."""
    return ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=fontsize,
        frameon=True,
    )


def _legend_below(
    ax: Axes, *, columns: int, fontsize: float = 8, y_anchor: float = -0.13
) -> Legend:
    """Place a compact legend below a single plotting area."""
    return ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        borderaxespad=0.0,
        ncol=columns,
        fontsize=fontsize,
        frameon=True,
    )


def _shared_legend_below(
    fig: Figure,
    axes: np.ndarray,
    *,
    columns: int = 3,
    fontsize: float = 7,
) -> Legend:
    """Give a multi-panel figure one legend in reserved space below its axes."""
    handles, labels = axes.flat[0].get_legend_handles_labels()
    return fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        borderaxespad=0.0,
        ncol=min(columns, max(len(labels), 1)),
        fontsize=fontsize,
        frameon=True,
    )


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text())


def _load_statistics(path: Path) -> dict:
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text())


def plot_roc_comparison(metrics_path: Path, out_path: Path) -> None:
    """Plot available model ROC curves on one axis."""
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
    _legend_outside_right(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(metrics_path: Path, out_path: Path) -> None:
    """Compare precision and recall for every evaluated model."""
    runs = [
        run
        for run in _load_metrics(metrics_path)["runs"]
        if run.get("precision_recall")
    ]
    if not runs:
        LOGGER.warning("No precision-recall data; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for run in runs:
        curve = run["precision_recall"]
        ax.plot(
            curve["recall"],
            curve["precision"],
            label=f"{run['model']} AP={curve['average_precision']:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _legend_outside_right(ax, fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_auc_intervals(statistics_path: Path, out_path: Path) -> None:
    """Show patient-bootstrap AUC estimates and 95% intervals."""
    models = _load_statistics(statistics_path).get("models", {})
    if not models:
        LOGGER.warning("No statistics data; skipping AUC intervals")
        return
    rows = []
    for name, model in models.items():
        auc = model["metrics"]["auc"]
        rows.append((name, auc["estimate"], auc["ci_lower"], auc["ci_upper"]))
    rows.sort(key=lambda row: row[1])
    names = [row[0] for row in rows]
    estimates = np.asarray([row[1] for row in rows])
    lower = np.asarray([row[2] for row in rows])
    upper = np.asarray([row[3] for row in rows])

    fig, ax = plt.subplots(figsize=(7, max(5, len(rows) * 0.42)))
    y = np.arange(len(rows))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.5, linestyle="--", color="grey", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Test AUC with patient-bootstrap 95% CI")
    ax.set_title("AUC uncertainty")
    ax.set_xlim(0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_specificity(metrics_path: Path, out_path: Path) -> None:
    """Compare test sensitivity at the shared validation specificity target."""
    runs = [
        run
        for run in _load_metrics(metrics_path)["runs"]
        if run.get("fixed_specificity")
    ]
    if not runs:
        LOGGER.warning("No fixed-specificity data; skipping")
        return
    runs.sort(key=lambda run: run["fixed_specificity"]["test"]["sensitivity"])
    names = [run["model"] for run in runs]
    sensitivity = [run["fixed_specificity"]["test"]["sensitivity"] for run in runs]
    specificity = [run["fixed_specificity"]["test"]["specificity"] for run in runs]
    target = float(runs[0]["fixed_specificity"]["target"])
    y = np.arange(len(runs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, max(5, len(runs) * 0.42)))
    ax.barh(y - width / 2, sensitivity, width, label="test sensitivity")
    ax.barh(y + width / 2, specificity, width, label="test specificity")
    ax.axvline(target, linestyle="--", color="grey", label=f"target {target:.0%}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Rate")
    ax.set_xlim(0, 1)
    ax.set_title("Performance at a validation-set specificity target")
    _legend_below(ax, columns=3, fontsize=8, y_anchor=-0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_subset(
    metrics_path: Path, out_path: Path, members: list[str], title: str
) -> None:
    """Plot ROC curves for selected runs in the given order."""
    by_name = {r.get("model"): r for r in _load_metrics(metrics_path)["runs"]}
    fig, ax = plt.subplots(figsize=(6, 6))
    drawn = 0
    for name in members:
        r = by_name.get(name)
        if r is None or not r.get("roc"):
            LOGGER.warning("Run %s missing or has no ROC arrays; skipping", name)
            continue
        roc = r["roc"]
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
    _legend_outside_right(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(history_path: Path, out_path: Path, title: str) -> None:
    """Plot training loss and validation AUC by epoch."""
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
    """Write one confusion-matrix heatmap per run."""
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
    """Plot AUC, sensitivity, and specificity by breast density."""
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

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
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

    fig.suptitle("Density-stratified metrics", y=0.97)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.34, wspace=0.20)
    _shared_legend_below(fig, axes)
    out_path = out_dir / "density_strata.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def plot_lesion_strata(metrics_path: Path, out_dir: Path) -> None:
    """Plot AUC, sensitivity, and specificity by lesion type."""
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("lesion_strata")]
    if not runs:
        LOGGER.warning("No lesion_strata data; skipping")
        return

    metrics_spec = [
        ("auc", "AUC"),
        ("sens", "Sensitivity"),
        ("spec", "Specificity"),
    ]
    lesions = ["mass", "calcification"]
    x = np.arange(len(lesions))
    bar_width = 0.8 / max(len(runs), 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    for ax, (key, label) in zip(axes, metrics_spec):
        for i, r in enumerate(runs):
            by_lesion = {s["lesion_type"]: s for s in r["lesion_strata"]}
            values = [by_lesion.get(le, {}).get(key, float("nan")) for le in lesions]
            offset = (i - len(runs) / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=r["model"])
        ax.set_xticks(x)
        ax.set_xticklabels([le.capitalize() for le in lesions])
        ax.set_xlabel("Lesion type")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by lesion type")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8)

    fig.suptitle("Lesion-type-stratified metrics", y=0.97)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.34, wspace=0.20)
    _shared_legend_below(fig, axes)
    out_path = out_dir / "lesion_strata.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def plot_reliability(metrics_path: Path, out_dir: Path) -> None:
    """Plot a post-calibration reliability diagram for each run."""
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
            near_right = pm >= 0.9
            near_top = om >= 0.95
            ax.annotate(
                f"{pm:.2f}",
                (pm, om),
                textcoords="offset points",
                xytext=(-5 if near_right else 5, -6 if near_top else 4),
                fontsize=7,
                ha="right" if near_right else "left",
                va="top" if near_top else "bottom",
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
        _legend_below(ax, columns=2, fontsize=8)
        out_path = out_dir / f"reliability_{r['model']}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def plot_decision_curves(metrics_path: Path, out_dir: Path) -> None:
    """Plot net benefit by probability threshold."""
    runs = _load_metrics(metrics_path)["runs"]
    runs = [r for r in runs if r.get("decision_curve")]
    if not runs:
        LOGGER.warning("No decision_curve data; skipping")
        return

    first_dc = runs[0]["decision_curve"]
    thresholds = first_dc["thresholds"]

    fig, ax = plt.subplots(figsize=(7, 6))
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
    _legend_outside_right(ax)
    out_path = out_dir / "decision_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def plot_gradcam_roi(metrics_path: Path, out_dir: Path) -> None:
    """Plot Grad-CAM localisation metrics for all, TP, and FN subsets."""
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

    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(13, 6))
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

    fig.suptitle("GradCAM vs lesion ROI alignment", y=0.97)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.34, wspace=0.22)
    _shared_legend_below(fig, axes)
    out_path = out_dir / "gradcam_roi.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out_path)


def main(
    metrics_path: Path,
    figures_dir: Path,
    models_dir: Path = Path("models"),
    statistics_path: Path = Path("results/statistics.json"),
) -> None:
    setup_logging()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_comparison(metrics_path, figures_dir / "roc_comparison.png")
    plot_precision_recall(metrics_path, figures_dir / "precision_recall.png")
    plot_auc_intervals(statistics_path, figures_dir / "auc_confidence_intervals.png")
    plot_fixed_specificity(metrics_path, figures_dir / "fixed_specificity.png")
    plot_roc_subset(
        metrics_path,
        figures_dir / "roc_baseline_vs_vgg.png",
        members=["baseline", "vgg16_scratch", "vgg16_imagenet"],
        title="ROC: baseline vs VGG-16 (scratch vs ImageNet)",
    )
    plot_roc_subset(
        metrics_path,
        figures_dir / "roc_highres_comparison.png",
        members=[
            "vgg16_imagenet_448",
            "vgg16_imagenet",
            "resnet50_imagenet",
        ],
        title="ROC: focused 448-pixel experiment",
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
    plot_lesion_strata(metrics_path, figures_dir)
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
    "--statistics",
    "statistics_path",
    type=click.Path(path_type=Path),
    default=Path("results/statistics.json"),
    show_default=True,
    help="Path to patient-bootstrap statistics.",
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
def cli(
    metrics_path: Path,
    statistics_path: Path,
    figures_dir: Path,
    models_dir: Path,
) -> None:
    main(metrics_path, figures_dir, models_dir, statistics_path)


if __name__ == "__main__":
    cli()
