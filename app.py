import gradio as gr
from main import app as fastapi_app

# Create a minimal Gradio UI to satisfy Hugging Face's SDK
demo = gr.Interface(
    fn=lambda: "Form Extraction API is running. The React UI connects to this endpoint.",
    inputs=None,
    outputs="text",
    title="API Server Active"
)

# Mount the dummy Gradio UI onto our real FastAPI application.
# Hugging Face will automatically detect 'app' and serve it using Uvicorn on port 7860.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
