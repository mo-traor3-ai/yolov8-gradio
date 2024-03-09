import PIL.Image as Image
import gradio as gr
from ultralytics import ASSETS, YOLO


model = YOLO("./faces.pt")

def predict_image(img, conf_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

interface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.40, label="Confidence Thresohld")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics YOLOv8 Gradio App",
    description="Upload images for Inference. The Ultralytics base YOLOv8n model is used by default.",
    examples=[
        ["./samples/face.jpg", 0.40],
        [ASSETS / "zidane.jpg", 0.40]
    ]
)

if __name__ == "__main__":
    interface.launch()
