import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.reporting import make_figures


def _capture_saved_figure(monkeypatch):
    captured = {}

    def capture(fig, *_args, **_kwargs):
        captured["figure"] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(make_figures.plt, "close", lambda _fig: None)
    return captured


def test_roc_legend_is_outside_the_plotting_area(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "model_a",
                        "test": {"auc": 0.75},
                        "roc": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                    }
                ]
            }
        )
    )
    captured = _capture_saved_figure(monkeypatch)

    make_figures.plot_roc_comparison(metrics_path, tmp_path / "roc.png")

    fig = captured["figure"]
    ax = fig.axes[0]
    legend = ax.get_legend()
    anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    assert anchor.x0 == pytest.approx(1.02)
    assert anchor.y0 == pytest.approx(0.5)
    plt.close("all")


def test_multi_panel_plot_uses_one_shared_legend(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.json"
    strata = [
        {"density": density, "auc": 0.7, "sens": 0.6, "spec": 0.8}
        for density in (1, 2, 3, 4)
    ]
    metrics_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"model": "model_a", "density_strata": strata},
                    {"model": "model_b", "density_strata": strata},
                ]
            }
        )
    )
    captured = _capture_saved_figure(monkeypatch)

    make_figures.plot_density_strata(metrics_path, tmp_path)

    fig = captured["figure"]
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "model_a",
        "model_b",
    ]
    anchor = fig.legends[0].get_bbox_to_anchor().transformed(
        fig.transFigure.inverted()
    )
    assert anchor.y0 == pytest.approx(0.015)
    assert np.isclose(fig.subplotpars.bottom, 0.34)
    plt.close("all")
