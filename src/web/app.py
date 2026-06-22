def build_app():
    import gradio as gr

    import src.web.evaluation_page as evaluation_page
    import src.web.finetune_page as finetune_page
    import src.web.inference_page as inference_page
    from src.web.constants import DISCLAIMER, TITLE

    with gr.Blocks() as app:
        app.title = "Mammo Second Reader - Research Use Only"
        gr.Markdown(f"# {TITLE}\n\n{DISCLAIMER}")
        with gr.Tab("Inference"):
            inference_page.render()
        with gr.Tab("Evaluation"):
            evaluation_page.render()
        with gr.Tab("Fine-tune"):
            finetune_page.render()
    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
