def render() -> None:
    import gradio as gr

    file_uploader = gr.File(
        label="Upload mammogram",
        file_types=[".dcm", ".png", ".jpg", ".jpeg"],
        type="filepath",
    )
