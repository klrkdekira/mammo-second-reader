def render() -> None:
    import gradio as gr

    from src.web.inference import available_models

    gr.Markdown(
        "Upload a ZIP containing one CSV manifest and DICOM or cached NPY images. "
        "DICOM metadata are stripped in the temporary work directory before scoring."
    )
    models = available_models()
    archive = gr.File(label="Evaluation ZIP", file_types=[".zip"], type="filepath")
    model = gr.Dropdown(
        choices=models, value=models[0] if models else None, label="Model"
    )
    run = gr.Button("Run Batch Evaluation", variant="primary")
    output = gr.JSON(label="Evaluation Metrics")

    def evaluate_archive(path: str, model_name: str):
        if not path or not model_name:
            raise gr.Error("Select an archive and an available model.")
        from src.web.evaluation import run_batch_evaluation

        try:
            return run_batch_evaluation(path, model_name)
        except (ValueError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc

    run.click(  # type: ignore[attr-defined]
        evaluate_archive, inputs=[archive, model], outputs=[output]
    )
