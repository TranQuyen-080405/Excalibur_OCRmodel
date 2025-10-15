# Excalibur_OCRmodel
# OCR Classification & Recognition System

This repository contains two main components:
1. **Classification model** – to identify the type of image and determine which OCR engine to use.
2. **OCR models** – including **PaddleOCR** and **NanoNet**, used for text extraction and post-processing.

---

## ⚙️ Environment Requirements

| Component | Python Version | Notes |
|------------|----------------|--------|
| `classification` (image classifier) | **Python 3.10** | Uses TensorFlow/Keras-based model (`keras_model.h5`) |
| `ocr modules` (PaddleOCR, NanoNet) | **Python 3.13** | Uses PaddleOCR, OpenCV, PIL, NumPy, etc. |

> ⚠️ It’s recommended to create two separate environments to avoid dependency conflicts.



---

## 📁 Folder Structure

project_root/
│
├──Checkpoint
│ └── keras_model.h5 # Pretrained classifier model
├── classification.ipynb # Script to classify images
├── Paddle.ipynb # PaddleOCR class definition
├── VLM.ipynb # NanoNet OCR class definition
├── utils.py # Image preprocessing functions
├── images/ # Input images for classification
│
└── README.md

---

## Image Path Configuration

Make sure to define your **image paths** properly in your scripts.

```python
# Example (for PaddleOCR or NanoNet)
img_path = r"D:\Classification\images\sample.jpg"
txt_dir  = r"D:\Classification\results"

---

## How to Run

1. **Open Jupyter Notebook or VSCode Notebook interface.**  
2. **Run notebooks in order** as listed below:

   1️⃣ `classification.ipynb`  
   → Classifies the document and decides which OCR model should be used.  
   Output: saves classification results inside `results/classification/`.

   2️⃣ `paddle_ocr.ipynb`  
   → Performs OCR using PaddleOCR model.  
   Output: saves text results and processed images to `results/paddle/`.

   3️⃣ `nanonet_ocr.ipynb`  
   → Performs OCR using Nanonet model.  
   Output: saves text results and processed images to `results/nanonet/`.

   4️⃣ `evaluation.ipynb`  
   → Compares OCR outputs from both models and evaluates accuracy.  
   Output: combined final result inside `results/merged/`.

---
