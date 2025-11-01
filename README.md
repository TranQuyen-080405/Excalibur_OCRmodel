# Excalibur_OCR
# OCR Classification & Recognition System

This repository contains the complete Optical Character Recognition (OCR) pipeline, designed to analyze the document type and apply the optimal OCR engine.

## 🌟 Key Components

1.  **Classification Model** – A TensorFlow/Keras-based deep learning model used to identify the image type and determine which OCR engine to invoke.
2.  **OCR Models** – Including **PaddleOCR** and **NanoNet** (referred to as VLM), used for text extraction and post-processing.

---

## ⚙️ Environment Requirements

| Component | Python Version | Notes |
|------------|----------------|--------|
| `classification` (image classifier) | **Python 3.10** | Uses the Keras-based model (`keras_model.h5`). |
| `ocr modules` (PaddleOCR, NanoNet) | **Python 3.13** | Uses PaddleOCR, OpenCV, PIL, NumPy, etc. |

> ⚠️ **Recommendation:** It is highly recommended to create **two separate virtual environments** to avoid dependency conflicts (e.g., `excalibur-class` for 3.10 and `excalibur-ocr` for 3.13).

---

## 📁 Folder Structure

project_root/
│
├── Checkpoint
│ └── keras_model.h5 # Pretrained classifier model
├── PaddleOCR.ipynb # PaddleOCR class definition
├── README.md
├── VLM.ipynb # NanoNet OCR class definition
├── classification.ipynb # Script to classify images (for running classification)
└── utils.py # Image preprocessing functions

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

   2️⃣ `Paddle.ipynb`  
   → Performs OCR using PaddleOCR model.  
   Output: saves text results and processed images to `results/paddle/`.

   3️⃣ `VLM.ipynb`  
   → Performs OCR using Nanonet model.  
   Output: saves text results and processed images to `results/nanonet/`..

---
