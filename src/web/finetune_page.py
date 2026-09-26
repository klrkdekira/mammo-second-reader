def render() -> None:
    import gradio as gr

    from src.web.finetune import default_output_name
    from src.web.inference import available_models

    gr.Markdown(
        "Fine-tune one of the trained checkpoints on a ZIP containing train.csv, "
        "val.csv and DICOM or cached NPY images. The result is saved as a new "
        "checkpoint and the original is left untouched. This research workflow "
        "is bounded to 50 epochs."
    )
    models = available_models()
    default_model = models[0] if models else None
    archive = gr.File(label="Fine-tuning ZIP", file_types=[".zip"], type="filepath")
    base_model = gr.Dropdown(
        choices=models, value=default_model, label="Base checkpoint"
    )
    output_name = gr.Textbox(
        value=default_output_name(default_model) if default_model else "",
        label="Save fine-tuned checkpoint as",
    )
    epochs = gr.Slider(1, 50, value=5, step=1, label="Epochs")
    learning_rate = gr.Number(value=1e-5, label="Learning rate")
    freeze = gr.Checkbox(value=True, label="Freeze backbone")
    run = gr.Button("Start Fine-tuning", variant="primary")
    output = gr.JSON(label="Latest Epoch")

    def refresh_models(selected: str | None):
        choices = available_models()
        value = selected if selected in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=value)

    def sync_output_name(selected: str | None):
        return default_output_name(selected) if selected else ""

    def fine_tune(
        archive_path: str,
        selected_model: str,
        new_name: str,
        n_epochs: int,
        lr: float,
        freeze_backbone: bool,
    ):
        if not archive_path:
            raise gr.Error("Upload a fine-tuning ZIP first.")
        if not selected_model:
            raise gr.Error("No trained checkpoints were found under models/.")
        if not new_name or not new_name.strip():
            raise gr.Error("Enter a name for the fine-tuned checkpoint.")
        import tempfile
        from pathlib import Path

        from src.web.finetune import materialise_workdir, stream_finetune_epochs

        try:
            with tempfile.TemporaryDirectory(prefix="mammo-finetune-") as tmp:
                workdir = materialise_workdir(archive_path, Path(tmp))
                yield from stream_finetune_epochs(
                    workdir,
                    selected_model,
                    new_name.strip(),
                    epochs=int(n_epochs),
                    lr=float(lr),
                    freeze_backbone=freeze_backbone,
                )
        except (ValueError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc

    base_model.focus(  # type: ignore[attr-defined]
        refresh_models, inputs=[base_model], outputs=[base_model]
    )
    base_model.change(  # type: ignore[attr-defined]
        sync_output_name, inputs=[base_model], outputs=[output_name]
    )
    run.click(  # type: ignore[attr-defined]
        fine_tune,
        inputs=[archive, base_model, output_name, epochs, learning_rate, freeze],
        outputs=[output],
    )
