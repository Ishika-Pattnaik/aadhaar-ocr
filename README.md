---
title: Aadhaar OCR API
emoji: 🆔
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Aadhaar OCR API

A high-performance, Privacy-First OCR API for extracting and validating Aadhaar card details from images. Built with FastAPI, PaddleOCR, and OpenCV.

## Features

- **Robust OCR**: Uses PaddleOCR with angle classification for high accuracy.
- **Smart Preprocessing**: Auto-cropping, denoising, and adaptive thresholding to handle scanned/camera images.
- **Validation**:
  - Verhoeff algorithm check for Aadhaar numbers.
  - Fuzzy matching for Name verification.
- **Privacy**: PII (Personally Identifiable Information) masking in logs.
- **Deployment Ready**: Dockerized and optimized for Hugging Face Spaces.

## Quick Start (Local)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**
   ```bash
   python server.py
   ```
   Server will start at `http://0.0.0.0:7860`.

3. **Test API**
   Open `http://localhost:7860/docs` to see the Swagger UI.

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Upload all files in this repository.
4. The Space will automatically build and start the server on port 7860.

## API Endpoints

### `POST /extract`
Extracts Name and Aadhaar number from an uploaded image.
- **Headers**: `X-API-Key: <your-secret-key>`
- **Form Data**:
  - `file`: (required) Image file.
  - `name`: (optional) User's name for matching.
  - `aadhaar`: (optional) User's Aadhaar number for validation.

### `POST /validate-aadhaar`
Validates a 12-digit Aadhaar number using the Verhoeff algorithm.
- **JSON Body**: `{"aadhaar_number": "1234 5678 9012"}`

## Environment Variables
- `API_KEY`: Secret key for authentication (Default: `your-secret-api-key-change-in-production`)
- `DEBUG_MODE`: Set to `true` for verbose logging.
# aadhaar-ocr
