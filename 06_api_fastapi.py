# -*- coding: utf-8 -*-
"""
=============================================================================
 06_api_fastapi.py — Deployment: REST API (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 REST API menggunakan FastAPI untuk integrasi dengan Flutter.
 Endpoints:
   POST /predict  — Upload audio + target huruf → skor keakuratan
   GET  /health   — Health check
   GET  /huruf    — Daftar 84 kelas huruf hijaiyah
=============================================================================
 Untuk menjalankan di Google Colab, gunakan ngrok sebagai tunnel.
=============================================================================
"""

# ===========================================================================
# CELL 1: Install Dependencies
# ===========================================================================
# !pip install -q fastapi uvicorn python-multipart pyngrok nest_asyncio
# !pip install -q tensorflow librosa opencv-python-headless soundfile

# ===========================================================================
# CELL 2: Import Dependencies
# ===========================================================================
import os
import io
import json
import numpy as np
import cv2
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import nest_asyncio
import uvicorn

nest_asyncio.apply()

print(f"TensorFlow: {tf.__version__}")

# ===========================================================================
# CELL 3: Mount Google Drive & Load Model
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')

# Load model
print("⏳ Memuat model...")
model = keras.models.load_model(os.path.join(MODEL_DIR, 'best_model_phase2.keras'))
print("  ✅ Model dimuat")

# Load metadata
with open(os.path.join(PREPROCESSED_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
num_classes = metadata['num_classes']
preprocess_params = metadata['preprocessing_params']

TARGET_SR = preprocess_params['target_sr']
TARGET_DURATION = preprocess_params['target_duration']
N_MFCC = preprocess_params['n_mfcc']
N_FFT = preprocess_params['n_fft']
HOP_LENGTH = preprocess_params['hop_length']
TOP_DB = preprocess_params['top_db']
IMG_SIZE = metadata['image_size']

print(f"  ✅ Config: {num_classes} kelas, SR={TARGET_SR}, MFCC={N_MFCC}")

# ===========================================================================
# CELL 4: Fungsi Preprocessing (Identik dengan 05_inference.py)
# ===========================================================================
def normalize_to_uint8(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)


def audio_to_mfcc_image(audio_array, sr_original):
    """Konversi audio → MFCC image 224×224×3."""
    y = np.array(audio_array, dtype=np.float32)

    if sr_original != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr_original, target_sr=TARGET_SR)
    sr = TARGET_SR

    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trimmed) < sr * 0.1:
        y_trimmed = y

    max_amp = np.max(np.abs(y_trimmed))
    if max_amp > 0:
        y_trimmed = y_trimmed / max_amp

    target_length = int(sr * TARGET_DURATION)
    if len(y_trimmed) < target_length:
        y_padded = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
    else:
        start = (len(y_trimmed) - target_length) // 2
        y_padded = y_trimmed[start:start + target_length]

    mfcc = librosa.feature.mfcc(y=y_padded, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    ch1 = normalize_to_uint8(cv2.resize(mfcc, (IMG_SIZE, IMG_SIZE)))
    ch2 = normalize_to_uint8(cv2.resize(delta, (IMG_SIZE, IMG_SIZE)))
    ch3 = normalize_to_uint8(cv2.resize(delta2, (IMG_SIZE, IMG_SIZE)))

    return np.stack([ch1, ch2, ch3], axis=-1)


def run_prediction(audio_array, sr, target_huruf, top_k=5):
    """Jalankan prediksi dan hitung skor."""
    # Preprocessing
    mfcc_img = audio_to_mfcc_image(audio_array, sr)

    # Prediksi
    img_input = mfcc_img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    probabilities = model.predict(img_input, verbose=0)[0]

    # Hasil
    predicted_idx = int(np.argmax(probabilities))
    predicted_huruf = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    # Cari target
    target_idx = None
    if target_huruf in class_names:
        target_idx = class_names.index(target_huruf)
    else:
        matches = [i for i, name in enumerate(class_names)
                    if target_huruf.lower() in name.lower()]
        if matches:
            target_idx = matches[0]
            target_huruf = class_names[target_idx]

    # Skor
    score = float(probabilities[target_idx]) * 100 if target_idx is not None else 0.0

    # Top-K
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_results = [
        {
            "huruf": class_names[int(idx)],
            "probability": round(float(probabilities[idx]), 6)
        }
        for idx in top_k_indices
    ]

    return {
        "target_huruf": target_huruf,
        "predicted_huruf": predicted_huruf,
        "score": round(score, 2),
        "confidence": round(confidence, 6),
        "is_correct": predicted_idx == target_idx if target_idx is not None else False,
        "top5": top_k_results
    }

# ===========================================================================
# CELL 5: Inisialisasi FastAPI
# ===========================================================================
app = FastAPI(
    title="Hijaiyah Pronunciation API",
    description=(
        "REST API untuk penilaian pelafalan huruf hijaiyah "
        "menggunakan MobileNetV2 dan ekstraksi MFCC.\n\n"
        "Tugas Akhir — Mohammad Putra Fauzan Fatah\n"
        "UIN Sunan Gunung Djati Bandung"
    ),
    version="1.0.0"
)

# CORS middleware (agar bisa diakses dari Flutter / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# CELL 6: API Endpoints
# ===========================================================================

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": num_classes,
        "tensorflow_version": tf.__version__
    }


# --- Daftar Huruf ---
@app.get("/huruf")
async def get_huruf_list():
    """Mendapatkan daftar 84 kelas huruf hijaiyah."""
    huruf_list = [
        {"index": i, "name": name}
        for i, name in enumerate(class_names)
    ]
    return {
        "total_classes": num_classes,
        "huruf": huruf_list
    }


# --- Prediksi Pelafalan ---
@app.post("/predict")
async def predict_pronunciation(
    audio: UploadFile = File(..., description="File audio (WAV/MP3/OGG)"),
    target_huruf: str = Form(..., description="Huruf target yang diminta dilafalkan")
):
    """
    Upload audio + target huruf → skor keakuratan pelafalan.

    **Request:**
    - `audio`: File audio (WAV, MP3, OGG, FLAC)
    - `target_huruf`: Nama kelas huruf target (contoh: "alif_fathah", "ba_kasrah")

    **Response:**
    - `target_huruf`: Huruf yang diminta
    - `predicted_huruf`: Huruf yang diprediksi model
    - `score`: Skor keakuratan (0-100)
    - `confidence`: Confidence prediksi (0-1)
    - `is_correct`: Apakah prediksi sesuai target
    - `top5`: Top 5 prediksi dengan probabilitas
    """
    try:
        # Validasi target huruf
        target_matched = target_huruf
        if target_huruf not in class_names:
            matches = [name for name in class_names
                       if target_huruf.lower() in name.lower()]
            if not matches:
                raise HTTPException(
                    status_code=400,
                    detail=f"Huruf target '{target_huruf}' tidak valid. "
                           f"Gunakan GET /huruf untuk melihat daftar kelas."
                )
            target_matched = matches[0]

        # Baca file audio
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio dengan soundfile/librosa
        try:
            y, sr = sf.read(audio_buffer)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # Stereo → mono
            y = y.astype(np.float32)
        except Exception:
            # Fallback: gunakan librosa
            audio_buffer.seek(0)
            y, sr = librosa.load(audio_buffer, sr=None, mono=True)

        if len(y) == 0:
            raise HTTPException(status_code=400, detail="File audio kosong.")

        # Jalankan prediksi
        result = run_prediction(y, sr, target_matched, top_k=5)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")

# ===========================================================================
# CELL 7: Jalankan API dengan ngrok
# ===========================================================================
print("\n🚀 Menjalankan FastAPI Server...")
print("=" * 60)

# --- Setup ngrok untuk public URL ---
# Anda perlu mendaftar di https://ngrok.com dan mendapatkan authtoken
# Uncomment dan ganti dengan authtoken Anda:

# from pyngrok import ngrok
# NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)
# public_url = ngrok.connect(8000)
# print(f"\n🌐 Public URL: {public_url}")
# print(f"📖 API Docs  : {public_url}/docs")

# --- Jalankan server ---
print(f"\n📖 API Docs (local): http://localhost:8000/docs")
print(f"\n  Endpoints:")
print(f"    GET  /health  — Health check")
print(f"    GET  /huruf   — Daftar huruf hijaiyah")
print(f"    POST /predict — Prediksi pelafalan")
print(f"\n  Contoh curl:")
print(f'    curl -X POST http://localhost:8000/predict \\')
print(f'      -F "audio=@audio.wav" \\')
print(f'      -F "target_huruf=alif_fathah"')
print(f"\n{'='*60}")

uvicorn.run(app, host="0.0.0.0", port=8000)
