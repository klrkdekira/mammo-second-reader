import base64
import io
from pathlib import Path


def render() -> None:
    import gradio as gr

    from src.web.inference import (
        available_models,
        model_threshold,
        run_single_inference,
    )

    models = available_models()
    default_model = (
        "vgg16_imagenet"
        if "vgg16_imagenet" in models
        else (models[0] if models else None)
    )

    gr.Markdown("Upload a mammogram image to classify it as benign or malignant.")
    with gr.Row():
        with gr.Column():
            model_select = gr.Dropdown(
                choices=models,
                value=default_model,
                label="Model",
            )
            file_input = gr.File(
                label="Upload mammogram",
                file_types=[".dcm", ".png", ".jpg", ".jpeg"],
                type="filepath",
            )
            threshold = gr.Slider(
                0.0,
                1.0,
                value=model_threshold(default_model) if default_model else 0.5,
                step=0.01,
                label="Threshold",
                precision=4,
            )
            run_btn = gr.Button("Run Inference", variant="primary")
        with gr.Column():
            prob_out = gr.Number(label="Probability of Malignancy", precision=4)
            label_out = gr.Label(label="Prediction")
            heatmap_out = gr.Image(label="Grad-CAM Overlay")

    def sync_threshold(model_name: str):
        return model_threshold(model_name) if model_name else 0.5

    def run_inference(file_path: str, model_name: str, threshold: float):
        if not file_path or not model_name:
            return None, None, None

        from PIL import Image

        contents = Path(file_path).read_bytes()
        result = run_single_inference(
            contents, Path(file_path).name, model_name, threshold
        )
        prob = result["probability"]
        confidences = {"Malignant": prob, "Benign": 1.0 - prob}
        overlay = None
        if result["gradcam_overlay"]:
            overlay_bytes = base64.b64decode(result["gradcam_overlay"])
            overlay = Image.open(io.BytesIO(overlay_bytes))
        return prob, confidences, overlay

    model_select.change(
        sync_threshold,
        inputs=[model_select],
        outputs=[threshold],
    )
    run_btn.click(
        run_inference,
        inputs=[file_input, model_select, threshold],
        outputs=[prob_out, label_out, heatmap_out],
    )
