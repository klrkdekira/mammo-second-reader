def render() -> None:
    import gradio as gr

    gr.Markdown(
        "Fine-tune a supplied checkpoint on a ZIP containing train.csv, val.csv "
        "and DICOM or cached NPY images. This research workflow is bounded to 50 epochs."
    )
    archive = gr.File(label="Fine-tuning ZIP", file_types=[".zip"], type="filepath")
    checkpoint = gr.File(label="Base checkpoint", file_types=[".pt"], type="filepath")
    model_name = gr.Textbox(value="baseline", label="Model name")
    epochs = gr.Slider(1, 50, value=5, step=1, label="Epochs")
    learning_rate = gr.Number(value=1e-5, label="Learning rate")
    freeze = gr.Checkbox(value=True, label="Freeze backbone")
    run = gr.Button("Start Fine-tuning", variant="primary")
    output = gr.JSON(label="Latest Epoch")

    def fine_tune(
        archive_path: str,
        checkpoint_path: str,
        selected_model: str,
        n_epochs: int,
        lr: float,
        freeze_backbone: bool,
    ):
        if not archive_path or not checkpoint_path or not selected_model:
            raise gr.Error("Archive, checkpoint and model name are required.")
        import tempfile
        from pathlib import Path

        from src.web.finetune import materialise_workdir, stream_finetune_epochs

        try:
            with tempfile.TemporaryDirectory(prefix="mammo-finetune-") as tmp:
                workdir = materialise_workdir(archive_path, Path(tmp))
                yield from stream_finetune_epochs(
                    workdir,
                    selected_model,
                    Path(checkpoint_path),
                    epochs=int(n_epochs),
                    lr=float(lr),
                    freeze_backbone=freeze_backbone,
                )
        except (ValueError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc

    run.click(  # type: ignore[attr-defined]
        fine_tune,
        inputs=[archive, checkpoint, model_name, epochs, learning_rate, freeze],
        outputs=[output],
    )
