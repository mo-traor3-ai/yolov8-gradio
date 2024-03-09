# Interactive Object Detection Demo: YOLOv8 🚀

## Introduction to Interactive Object Detection

This Gradio demo provides an easy and interactive way to perform object detection using a custom trained YOLOv8 Face Detection model [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) model. Users can upload images and adjust parameters like confidence threshold to get real-time detection results (e.g "detect faces in this image").

The Roboflow YOLOv8 Object Detection training notebook was used to train the model.

* [Roboflow Notebooks](https://github.com/roboflow/notebooks)

* [Face Detection dataset](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/18) from Mohamed Traore used in training.

[![Explore the Dataset](https://github.com/roboflow/notebooks/blob/main/assets/badges/roboflow-dataset.svg)](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i)

![Gradio Example Screenshot](/assets/gradio-interface.png)

![Example Terminal Output - CPU Inference with Gradio](/assets/cpu-infer-gradio-output.png)

## Installing Gradio

Create and activate a virtual environment running a Python version between 3.7 and 3.12

Python 3.9 example (Linux):

```bash
python3.9 -m venv gradio

source gradio/bin/activate
```

Install Gradio with PIP

```bash
pip install gradio
```

## Running the App

```bash
python3 app.py
```

Click on the link that opens in your terminal, or copy and paste it in a browser window. The default link for the app is `http://127.0.0.1:7860` (running locally).

## How to Use the Interface

1. **Upload Image:** Click on 'Upload Image' to choose an image file for object detection.
2. **Adjust Parameters:**
    * **Confidence Threshold:** Slider to set the minimum confidence level for detecting objects.
    * **IoU Threshold:** Slider to set the IoU threshold for distinguishing different objects.
3. **View Results:** The processed image with detected objects and their labels will be displayed.

## Example Test Images

* **Sample Image 1:** A picture of myself, with my face clearly visible.
* **Sample Image 2:** Detection on a sports image.

### Gradio Interface Components

| Component    | Description                              |
|--------------|------------------------------------------|
| Image Input  | To upload the image for detection.       |
| Sliders      | To adjust confidence threshold for displaying model predictions. |
| Image Output | To display the detection results.        |
