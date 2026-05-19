# -*- coding: utf-8 -*-
"""
=============================================================================
 05_inference.py — Deployment: Inference (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 Fungsi:
 1. Load model terbaik
 2. Prediksi audio baru → huruf terprediksi + skor keakuratan 0–100
 3. Skor = probabilitas softmax kelas target × 100
 4. Demo inferensi dengan audio dari test set dan upload baru
=============================================================================
"""

# ===========================================================================
# CELL 1: Install & Import Dependencies
# ===========================================================================
# !pip install -q tensorflow librosa opencv-python-headless soundfile

import os
import json
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import soundfile as sf

print(f"TensorFlow: {tf.__version__}")

# ===========================================================================
# CELL 2: Mount Google Drive & Setup
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '05_inference')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
# CELL 3: Load Model & Metadata
# ===========================================================================
print("⏳ Memuat model dan metadata...")

# Load model terbaik
model = keras.models.load_model(os.path.join(MODEL_DIR, 'best_model_phase2.keras'))
print(f"  ✅ Model dimuat")

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

print(f"  ✅ Metadata dimuat: {num_classes} kelas")
print(f"  ✅ Params: SR={TARGET_SR}, MFCC={N_MFCC}, Duration={TARGET_DURATION}s")

# ===========================================================================
# CELL 4: Fungsi Preprocessing (Sama dengan 02_preprocessing)
# ===========================================================================
def normalize_to_uint8(arr):
    """Normalisasi array ke range [0, 255] uint8."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)


def audio_to_mfcc_image(audio_array, sr_original):
    """
    Konversi sinyal audio → citra MFCC 224×224×3.
    Sama persis dengan pipeline di 02_preprocessing_mfcc.py.
    """
    y = np.array(audio_array, dtype=np.float32)

    # Resample jika perlu
    if sr_original != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr_original, target_sr=TARGET_SR)
    sr = TARGET_SR

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trimmed) < sr * 0.1:
        y_trimmed = y

    # Peak normalization
    max_amp = np.max(np.abs(y_trimmed))
    if max_amp > 0:
        y_trimmed = y_trimmed / max_amp

    # Pad/truncate
    target_length = int(sr * TARGET_DURATION)
    if len(y_trimmed) < target_length:
        y_padded = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
    else:
        start = (len(y_trimmed) - target_length) // 2
        y_padded = y_trimmed[start:start + target_length]

    # MFCC + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(y=y_padded, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Resize ke 224×224
    ch1 = normalize_to_uint8(cv2.resize(mfcc, (IMG_SIZE, IMG_SIZE)))
    ch2 = normalize_to_uint8(cv2.resize(delta, (IMG_SIZE, IMG_SIZE)))
    ch3 = normalize_to_uint8(cv2.resize(delta2, (IMG_SIZE, IMG_SIZE)))

    return np.stack([ch1, ch2, ch3], axis=-1)

# ===========================================================================
# CELL 5: Fungsi Prediksi + Skor Keakuratan
# ===========================================================================
def predict_pronunciation(audio_path_or_array, target_huruf,
                          sr_original=None, top_k=5, show_plot=True):
    """
    Prediksi pelafalan huruf hijaiyah dan hitung skor keakuratan.

    Parameters:
        audio_path_or_array : str atau np.ndarray
            Path ke file audio ATAU numpy array audio
        target_huruf : str
            Nama kelas huruf target yang diminta dilafalkan
            (harus sesuai dengan nama kelas di dataset)
        sr_original : int (optional)
            Sampling rate jika input berupa array
        top_k : int
            Jumlah prediksi teratas yang ditampilkan
        show_plot : bool
            Tampilkan visualisasi atau tidak

    Returns:
        dict: {
            'target_huruf': str,
            'predicted_huruf': str,
            'score': float (0-100),
            'confidence': float (0-1),
            'top_k': list of dict,
            'is_correct': bool
        }
    """
    # === 1. Load audio ===
    if isinstance(audio_path_or_array, str):
        y, sr = librosa.load(audio_path_or_array, sr=None)
    else:
        y = audio_path_or_array
        sr = sr_original if sr_original else TARGET_SR

    # === 2. Preprocessing: audio → MFCC image ===
    mfcc_img = audio_to_mfcc_image(y, sr)

    # === 3. Prediksi ===
    img_input = mfcc_img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1, 224, 224, 3)

    probabilities = model.predict(img_input, verbose=0)[0]

    # === 4. Hasil ===
    predicted_idx = np.argmax(probabilities)
    predicted_huruf = class_names[predicted_idx]
    predicted_confidence = float(probabilities[predicted_idx])

    # Cari index target
    if target_huruf in class_names:
        target_idx = class_names.index(target_huruf)
    else:
        # Cari partial match
        matches = [i for i, name in enumerate(class_names) if target_huruf.lower() in name.lower()]
        if matches:
            target_idx = matches[0]
            target_huruf = class_names[target_idx]
            print(f"  ℹ️  Target huruf dicocokkan ke: {target_huruf}")
        else:
            print(f"  ⚠️ Target huruf '{target_huruf}' tidak ditemukan!")
            print(f"     Kelas yang tersedia: {class_names[:10]}...")
            return None

    # Skor keakuratan = probabilitas kelas target × 100
    target_probability = float(probabilities[target_idx])
    score = target_probability * 100

    # Top-K prediksi
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_results = [
        {
            'huruf': class_names[idx],
            'probability': float(probabilities[idx]),
            'percentage': float(probabilities[idx]) * 100
        }
        for idx in top_k_indices
    ]

    is_correct = predicted_idx == target_idx

    result = {
        'target_huruf': target_huruf,
        'predicted_huruf': predicted_huruf,
        'score': round(score, 2),
        'confidence': round(predicted_confidence, 4),
        'is_correct': is_correct,
        'top_k': top_k_results
    }

    # === 5. Tampilkan hasil ===
    print(f"\n{'='*50}")
    print(f"🎯 TARGET   : {target_huruf}")
    print(f"🔮 PREDIKSI : {predicted_huruf} (confidence: {predicted_confidence:.2%})")
    print(f"📊 SKOR     : {score:.1f}/100")
    print(f"{'✅ BENAR' if is_correct else '❌ SALAH'}")
    print(f"\n  Top-{top_k} Prediksi:")
    for i, item in enumerate(top_k_results):
        marker = "→" if item['huruf'] == target_huruf else " "
        print(f"  {marker} {i+1}. {item['huruf']:20s} — {item['percentage']:6.2f}%")
    print(f"{'='*50}")

    # === 6. Visualisasi ===
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color='steelblue')
        axes[0].set_title('Waveform Audio Input', fontsize=12, fontweight='bold')

        # MFCC Image (RGB)
        axes[1].imshow(mfcc_img)
        axes[1].set_title('MFCC Image (R=MFCC, G=Δ, B=ΔΔ)', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Top-K Bar Chart
        huruf_names = [item['huruf'] for item in top_k_results]
        probs = [item['percentage'] for item in top_k_results]
        colors_bar = ['green' if h == target_huruf else 'steelblue' for h in huruf_names]
        axes[2].barh(range(len(huruf_names)), probs, color=colors_bar)
        axes[2].set_yticks(range(len(huruf_names)))
        axes[2].set_yticklabels(huruf_names, fontsize=10)
        axes[2].set_xlabel('Probabilitas (%)', fontsize=11)
        axes[2].set_title(f'Top-{top_k} Prediksi', fontsize=12, fontweight='bold')
        axes[2].invert_yaxis()

        status = "✅ BENAR" if is_correct else "❌ SALAH"
        plt.suptitle(
            f'Inferensi: Target={target_huruf} | Prediksi={predicted_huruf} | '
            f'Skor={score:.1f}/100 | {status}',
            fontsize=14, fontweight='bold', y=1.03
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'inference_{target_huruf}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

    return result

# ===========================================================================
# CELL 6: Demo Inferensi dengan Data Test
# ===========================================================================
print("\n🧪 DEMO INFERENSI DENGAN DATA TEST")
print("=" * 60)

# Load beberapa sampel test
X_test = np.load(os.path.join(PREPROCESSED_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PREPROCESSED_DIR, 'y_test.npy'))

# Load dataset asli untuk mendapatkan audio mentah
from datasets import load_dataset
print("⏳ Memuat dataset asli untuk demo audio...")
dataset = load_dataset("Reinjin/Pelafalan_Huruf_Hijaiyah", split="train")

# Demo: ambil 5 sampel dari kelas berbeda
demo_classes = list(range(0, min(num_classes, 25), 5))  # 5 kelas tersebar
demo_results = []

for cls_idx in demo_classes:
    # Cari sampel dari kelas ini di dataset asli
    sample_idx = next(i for i, s in enumerate(dataset) if s['label'] == cls_idx)
    sample = dataset[sample_idx]

    audio_array = np.array(sample['audio']['array'], dtype=np.float32)
    sr = sample['audio']['sampling_rate']
    target = class_names[cls_idx]

    result = predict_pronunciation(
        audio_path_or_array=audio_array,
        target_huruf=target,
        sr_original=sr,
        top_k=5,
        show_plot=True
    )
    if result:
        demo_results.append(result)

# ===========================================================================
# CELL 7: Demo Upload Audio Baru (Opsional)
# ===========================================================================
print("\n📤 DEMO UPLOAD AUDIO BARU")
print("=" * 60)
print("Untuk menguji dengan audio baru, gunakan kode berikut:\n")

print("""
# --- Upload file audio dari komputer ---
from google.colab import files

uploaded = files.upload()  # Upload file .wav

for filename in uploaded.keys():
    print(f"\\nMemproses: {filename}")
    result = predict_pronunciation(
        audio_path_or_array=filename,
        target_huruf='alif_fathah',  # Ganti dengan huruf target
        top_k=5,
        show_plot=True
    )

# --- Atau rekam langsung dari browser ---
# (Membutuhkan izin mikrofon)
#
# from google.colab import output
# from IPython.display import HTML, Audio
#
# Rekam menggunakan widget HTML5:
# display(HTML('''
# <button onclick="startRecording()">🎙️ Mulai Rekam</button>
# <button onclick="stopRecording()">⏹️ Stop</button>
# <script>
# // ... JavaScript recording code ...
# </script>
# '''))
""")

# ===========================================================================
# CELL 8: Ringkasan Inferensi
# ===========================================================================
if demo_results:
    print(f"\n{'='*60}")
    print(f"📊 RINGKASAN DEMO INFERENSI")
    print(f"{'='*60}")
    print(f"  Total sampel diuji: {len(demo_results)}")
    correct = sum(1 for r in demo_results if r['is_correct'])
    print(f"  Prediksi benar    : {correct}/{len(demo_results)}")
    avg_score = np.mean([r['score'] for r in demo_results])
    print(f"  Rata-rata skor    : {avg_score:.1f}/100")
    print(f"\n  Detail:")
    for r in demo_results:
        status = "✅" if r['is_correct'] else "❌"
        print(f"    {status} Target: {r['target_huruf']:20s} → "
              f"Pred: {r['predicted_huruf']:20s} | Skor: {r['score']:.1f}")
    print(f"{'='*60}")

    # Simpan hasil
    inference_path = os.path.join(RESULTS_DIR, 'demo_inference_results.json')
    with open(inference_path, 'w') as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Hasil disimpan: {inference_path}")
