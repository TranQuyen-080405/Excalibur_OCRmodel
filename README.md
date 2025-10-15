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
