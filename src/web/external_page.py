"""Gradio page for viewing or running the INbreast external evaluation."""

from pathlib import Path

WARNING = (
    "> **Warning: this is a one-shot, pre-registered test.** The model, both thresholds "
    "and the calibration temperature are transferred unchanged from the frozen "
    "CBIS-DDSM run; nothing is fitted on INbreast. Re-running after reading a "
    "result and keeping the better one would turn a generalisation test into "
    "selection on the test set. Load the existing result unless you intend to "
    "use the one-shot test."
)

NOTES = (
    "INbreast labels are **radiological BI-RADS assessments**; training uses "
    "biopsy-confirmed pathology. The external target is therefore a "
    "related but different construct. The dense-breast (D4) and mass strata are "
    "small externally; interpret their point estimates with their intervals."
)


def _empty_table():
    import pandas as pd

    return pd.DataFrame([{"note": "load a result to populate this table"}])


def _roc_figure(fpr, tpr, subset_label: str):
    """ROC for the loaded subset, or None if plotting is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axes = plt.subplots(figsize=(4.2, 4.2))
    if fpr and tpr:
        axes.plot(fpr, tpr, linewidth=2, label=subset_label)
        axes.legend(loc="lower right", fontsize=8)
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title("Cold external ROC")
    figure.tight_layout()
    return figure


def render() -> None:
    import gradio as gr

    from src.web.external import (
        DEFAULT_CONFIG,
        DEFAULT_RESULT,
        SUBSET_LABELS,
        available_subsets,
        headline_table,
        load_result,
        locked_markdown,
        readiness,
        readiness_markdown,
        roc_points,
        run_cold_evaluation,
        strata_table,
        summary_markdown,
    )

    gr.Markdown(
        "Cold external evaluation on INbreast, using the promoted "
        "448-pixel VGG-16 exactly as frozen.\n\n" + WARNING
    )

    with gr.Row():
        config_box = gr.Textbox(
            value=str(DEFAULT_CONFIG), label="External config", scale=2
        )
        result_box = gr.Textbox(value=str(DEFAULT_RESULT), label="Result file", scale=2)

    with gr.Accordion("Readiness", open=True):
        readiness_md = gr.Markdown("Press *Check readiness* to inspect the inputs.")
        check_btn = gr.Button("Check readiness")

    with gr.Row():
        load_btn = gr.Button("Load existing result", variant="primary")
        subset_select = gr.Dropdown(
            choices=list(SUBSET_LABELS), value="full", label="Subset"
        )

    locked_md = gr.Markdown("No result loaded.")
    summary_md = gr.Markdown()
    with gr.Row():
        headline = gr.Dataframe(
            value=_empty_table(), label="Metrics with patient-level 95% CIs", scale=3
        )
        roc_plot = gr.Plot(label="ROC", scale=2)
    with gr.Row():
        density = gr.Dataframe(value=_empty_table(), label="Density strata")
        lesion = gr.Dataframe(value=_empty_table(), label="Lesion strata")

    gr.Markdown(NOTES)

    with gr.Accordion("Run the cold evaluation (uses the one-shot test)", open=False):
        gr.Markdown(WARNING)
        acknowledge = gr.Checkbox(
            value=False,
            label="I understand this consumes the one-shot pre-registered cold test.",
        )
        run_btn = gr.Button("Run cold evaluation", variant="stop")
        run_status = gr.Markdown()

    def check(config_path: str, result_path: str):
        state = readiness(Path(config_path), Path(result_path))
        return readiness_markdown(state)

    def load(result_path: str, subset: str):
        try:
            result = load_result(Path(result_path))
        except (FileNotFoundError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        subsets = available_subsets(result)
        if not subsets:
            raise gr.Error("The result file contains no scored subsets.")
        chosen = subset if subset in subsets else subsets[0]
        fpr, tpr = roc_points(result, chosen)
        return (
            gr.update(choices=subsets, value=chosen),
            locked_markdown(result),
            summary_markdown(result, chosen),
            headline_table(result, chosen),
            _roc_figure(fpr, tpr, SUBSET_LABELS.get(chosen, chosen)),
            strata_table(result, chosen, "density_strata"),
            strata_table(result, chosen, "lesion_strata"),
        )

    def switch(result_path: str, subset: str):
        """Re-render for another subset without re-reading anything else."""
        try:
            result = load_result(Path(result_path))
        except (FileNotFoundError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        fpr, tpr = roc_points(result, subset)
        return (
            summary_markdown(result, subset),
            headline_table(result, subset),
            _roc_figure(fpr, tpr, SUBSET_LABELS.get(subset, subset)),
            strata_table(result, subset, "density_strata"),
            strata_table(result, subset, "lesion_strata"),
        )

    def run(config_path: str, result_path: str, acknowledged: bool):
        try:
            run_cold_evaluation(
                Path(config_path),
                acknowledged=acknowledged,
                result_path=Path(result_path),
            )
        except (OSError, ValueError, KeyError) as exc:
            raise gr.Error(str(exc)) from exc
        return f"Cold evaluation complete. Wrote `{result_path}`. Load it above."

    check_btn.click(  # type: ignore[attr-defined]
        check, inputs=[config_box, result_box], outputs=[readiness_md]
    )
    load_btn.click(  # type: ignore[attr-defined]
        load,
        inputs=[result_box, subset_select],
        outputs=[
            subset_select,
            locked_md,
            summary_md,
            headline,
            roc_plot,
            density,
            lesion,
        ],
    )
    subset_select.change(  # type: ignore[attr-defined]
        switch,
        inputs=[result_box, subset_select],
        outputs=[summary_md, headline, roc_plot, density, lesion],
    )
    run_btn.click(  # type: ignore[attr-defined]
        run, inputs=[config_box, result_box, acknowledge], outputs=[run_status]
    )
