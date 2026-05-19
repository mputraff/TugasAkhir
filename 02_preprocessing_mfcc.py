# -*- coding: utf-8 -*-
"""
=============================================================================
 02_preprocessing_mfcc.py — Data Preparation (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 Pipeline:
 1. Load daftar file valid dari valid_files.json (output 01_data_exploration)
 2. Trim silence → normalisasi → pad/truncate ke durasi tetap
 3. Ekstraksi MFCC (n_mfcc=40)
 4. Hitung Delta MFCC & Delta-Delta MFCC → 3 channel RGB
 5. Resize ke 224×224×3
 6. Split data: 80% train, 10% val, 10% test (stratified)
 7. Simpan sebagai numpy arrays ke Google Drive
=============================================================================
"""

# ===========================================================================
# CELL 1: Install & Import Dependencies
# ===========================================================================
# !pip install -q librosa scikit-learn tqdm opencv-python-headless

import os
import json
import numpy as np
import cv2
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===========================================================================
# CELL 2: Mount Google Drive & Setup Direktori
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '02_preprocessing')

os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📁 Preprocessed dir : {PREPROCESSED_DIR}")
print(f"📁 Results dir      : {RESULTS_DIR}")

# ===========================================================================
# CELL 3: Konfigurasi Parameter
# ===========================================================================
# Parameter audio
TARGET_SR = 22050         # Sampling rate target
TARGET_DURATION = 1.0     # Durasi target dalam detik (sesuaikan setelah eksplorasi)
TOP_DB = 25               # Threshold trim silence (dB)

# Parameter MFCC
N_MFCC = 40               # Jumlah koefisien MFCC
N_FFT = 2048              # Ukuran FFT window
HOP_LENGTH = 512          # Hop length

# Parameter output
IMG_SIZE = 224             # Ukuran output image (224×224)
NUM_CHANNELS = 3           # 3 channel: MFCC + Delta + Delta-Delta

# Parameter split
TEST_SIZE = 0.10           # 10% test
VAL_SIZE = 0.10            # 10% validation (dari sisa setelah test)
RANDOM_STATE = 42

print(f"\n{'='*60}")
print(f"⚙️  KONFIGURASI PARAMETER")
print(f"{'='*60}")
print(f"  Sampling Rate     : {TARGET_SR} Hz")
print(f"  Durasi Target     : {TARGET_DURATION}s")
print(f"  n_mfcc            : {N_MFCC}")
print(f"  n_fft             : {N_FFT}")
print(f"  hop_length        : {HOP_LENGTH}")
print(f"  Image Size        : {IMG_SIZE}×{IMG_SIZE}×{NUM_CHANNELS}")
print(f"  3 Channel         : MFCC + Delta + Delta-Delta")
print(f"  Split             : Train 80% / Val 10% / Test 10%")
print(f"{'='*60}")

# ===========================================================================
# CELL 4: Load Daftar File Valid dari 01_data_exploration
# ===========================================================================
print("\n⏳ Memuat daftar file valid dari valid_files.json...")

valid_files_path = os.path.join(BASE_DIR, 'valid_files.json')
with open(valid_files_path, 'r') as f:
    valid_files = json.load(f)

# Bangun class_names dari data
class_names = sorted(list(set(item['class_name'] for item in valid_files)))
num_classes = len(class_names)
num_samples = len(valid_files)

# Buat mapping class_name → index yang konsisten
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print(f"  ✅ File valid dimuat: {num_samples} sampel, {num_classes} kelas")
print(f"  📄 Sumber: {valid_files_path}")

# ===========================================================================
# CELL 5: Fungsi Preprocessing
# ===========================================================================
def normalize_to_uint8(arr):
    """Normalisasi array ke range [0, 255] uint8."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return normalized.astype(np.uint8)


def preprocess_audio_to_mfcc_image(audio_array, sr_original):
    """
    Mengkonversi sinyal audio menjadi citra MFCC 224×224×3.

    Pipeline:
    1. Resample ke TARGET_SR jika perlu
    2. Trim silence
    3. Normalisasi amplitudo (peak normalization)
    4. Pad atau truncate ke TARGET_DURATION
    5. Ekstraksi MFCC (n_mfcc=40)
    6. Hitung Delta MFCC dan Delta-Delta MFCC
    7. Resize masing-masing ke 224×224
    8. Normalisasi ke [0, 255] uint8
    9. Stack 3 channel: [MFCC, Delta, Delta-Delta]

    Parameters:
        audio_array : np.ndarray — sinyal audio mentah
        sr_original : int — sampling rate asli

    Returns:
        np.ndarray — citra MFCC (224, 224, 3) uint8
        atau None jika gagal
    """
    try:
        y = np.array(audio_array, dtype=np.float32)

        # 1. Resample jika sampling rate berbeda
        if sr_original != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr_original, target_sr=TARGET_SR)
        sr = TARGET_SR

        # 2. Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

        # Jika setelah trim terlalu pendek, gunakan audio asli
        if len(y_trimmed) < sr * 0.1:  # minimal 0.1 detik
            y_trimmed = y

        # 3. Normalisasi amplitudo (peak normalization)
        max_amp = np.max(np.abs(y_trimmed))
        if max_amp > 0:
            y_trimmed = y_trimmed / max_amp

        # 4. Pad atau truncate ke durasi target
        target_length = int(sr * TARGET_DURATION)
        if len(y_trimmed) < target_length:
            # Pad dengan zero di akhir
            y_padded = np.pad(y_trimmed, (0, target_length - len(y_trimmed)),
                              mode='constant', constant_values=0)
        else:
            # Truncate dari tengah
            start = (len(y_trimmed) - target_length) // 2
            y_padded = y_trimmed[start:start + target_length]

        # 5. Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(
            y=y_padded, sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # 6. Hitung Delta dan Delta-Delta
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # 7. Resize masing-masing ke IMG_SIZE × IMG_SIZE
        mfcc_resized = cv2.resize(mfcc, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
        delta_resized = cv2.resize(delta_mfcc, (IMG_SIZE, IMG_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
        delta2_resized = cv2.resize(delta2_mfcc, (IMG_SIZE, IMG_SIZE),
                                    interpolation=cv2.INTER_LINEAR)

        # 8. Normalisasi ke [0, 255] uint8
        ch_mfcc = normalize_to_uint8(mfcc_resized)
        ch_delta = normalize_to_uint8(delta_resized)
        ch_delta2 = normalize_to_uint8(delta2_resized)

        # 9. Stack 3 channel → (224, 224, 3)
        img = np.stack([ch_mfcc, ch_delta, ch_delta2], axis=-1)

        return img

    except Exception as e:
        print(f"    ⚠️ Error: {e}")
        return None

# ===========================================================================
# CELL 6: Proses Seluruh Dataset
# ===========================================================================
print(f"\n🔄 Memproses {num_samples} file audio valid → MFCC images...")
print(f"   Target output: ({IMG_SIZE}, {IMG_SIZE}, {NUM_CHANNELS}) uint8\n")

images = []
labels_processed = []
failed_indices = []

for i, item in enumerate(tqdm(valid_files, desc="Preprocessing")):
    file_path = item['path']
    class_name = item['class_name']
    label = class_to_idx[class_name]

    try:
        # Load audio dari file di Google Drive
        y, sr = librosa.load(file_path, sr=None, mono=True)
        img = preprocess_audio_to_mfcc_image(y, sr)
    except Exception as e:
        print(f"    ⚠️ Gagal load {os.path.basename(file_path)}: {e}")
        img = None

    if img is not None:
        images.append(img)
        labels_processed.append(label)
    else:
        failed_indices.append(i)

# Konversi ke numpy arrays
X = np.array(images, dtype=np.uint8)
y = np.array(labels_processed, dtype=np.int32)

print(f"\n{'='*60}")
print(f"✅ PREPROCESSING SELESAI")
print(f"{'='*60}")
print(f"  Shape X      : {X.shape}")
print(f"  Dtype X      : {X.dtype}")
print(f"  Shape y      : {y.shape}")
print(f"  Sampel gagal : {len(failed_indices)}")
print(f"  Memory X     : {X.nbytes / 1e6:.1f} MB")
print(f"{'='*60}")

# ===========================================================================
# CELL 7: Visualisasi Sampel Hasil Preprocessing
# ===========================================================================
print("\n🖼️  Visualisasi Sampel Hasil Preprocessing...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx in range(8):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]

    sample_idx = idx * (len(X) // 8)
    img_sample = X[sample_idx]
    label_idx = y[sample_idx]

    # Tampilkan sebagai RGB image
    ax.imshow(img_sample)
    ax.set_title(f"Kelas: {class_names[label_idx]}\n"
                 f"R=MFCC, G=Δ, B=ΔΔ", fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sampel Hasil Preprocessing: MFCC → Image 224×224×3\n'
             '(Red=MFCC, Green=Delta, Blue=Delta-Delta)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sampel_preprocessing.png'), dpi=150, bbox_inches='tight')
plt.show()

# Visualisasi per-channel
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sample_img = X[0]
channel_names = ['MFCC', 'Delta MFCC', 'Delta-Delta MFCC']
cmaps = ['Reds', 'Greens', 'Blues']

for ch in range(3):
    axes[ch].imshow(sample_img[:, :, ch], cmap=cmaps[ch], aspect='auto')
    axes[ch].set_title(f'Channel {ch+1}: {channel_names[ch]}', fontsize=13, fontweight='bold')
    axes[ch].set_xlabel('Time axis')
    axes[ch].set_ylabel('MFCC coefficient axis')

plt.suptitle(f'Decomposisi 3 Channel — Kelas: {class_names[y[0]]}',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sampel_3channel.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualisasi disimpan.")

# ===========================================================================
# CELL 8: Split Data (Stratified)
# ===========================================================================
print(f"\n📊 Membagi data: Train 80% / Val 10% / Test 10% (Stratified)...")

# Split 1: train+val (90%) vs test (10%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Split 2: train (≈89% dari trainval) vs val (≈11% dari trainval) → overall 80/10
val_ratio = VAL_SIZE / (1 - TEST_SIZE)  # = 0.10 / 0.90 ≈ 0.111
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_trainval
)

print(f"\n  📦 Train : {X_train.shape[0]:,} sampel ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  📦 Val   : {X_val.shape[0]:,} sampel ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  📦 Test  : {X_test.shape[0]:,} sampel ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verifikasi stratified split
from collections import Counter
print(f"\n  Verifikasi distribusi kelas (5 kelas pertama):")
for cls_idx in range(5):
    n_train = sum(1 for l in y_train if l == cls_idx)
    n_val = sum(1 for l in y_val if l == cls_idx)
    n_test = sum(1 for l in y_test if l == cls_idx)
    print(f"    Kelas {cls_idx} ({class_names[cls_idx]}): "
          f"train={n_train}, val={n_val}, test={n_test}")

# ===========================================================================
# CELL 9: Simpan Data ke Google Drive
# ===========================================================================
print(f"\n💾 Menyimpan data ke Google Drive...")
print(f"   Direktori: {PREPROCESSED_DIR}\n")

# Simpan arrays
np.save(os.path.join(PREPROCESSED_DIR, 'X_train.npy'), X_train)
print(f"  ✅ X_train.npy  : {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")

np.save(os.path.join(PREPROCESSED_DIR, 'X_val.npy'), X_val)
print(f"  ✅ X_val.npy    : {X_val.shape} ({X_val.nbytes/1e6:.1f} MB)")

np.save(os.path.join(PREPROCESSED_DIR, 'X_test.npy'), X_test)
print(f"  ✅ X_test.npy   : {X_test.shape} ({X_test.nbytes/1e6:.1f} MB)")

np.save(os.path.join(PREPROCESSED_DIR, 'y_train.npy'), y_train)
print(f"  ✅ y_train.npy  : {y_train.shape}")

np.save(os.path.join(PREPROCESSED_DIR, 'y_val.npy'), y_val)
print(f"  ✅ y_val.npy    : {y_val.shape}")

np.save(os.path.join(PREPROCESSED_DIR, 'y_test.npy'), y_test)
print(f"  ✅ y_test.npy   : {y_test.shape}")

# Simpan metadata
metadata = {
    'class_names': class_names,
    'num_classes': num_classes,
    'image_size': IMG_SIZE,
    'num_channels': NUM_CHANNELS,
    'channel_description': ['MFCC', 'Delta MFCC', 'Delta-Delta MFCC'],
    'preprocessing_params': {
        'target_sr': TARGET_SR,
        'target_duration': TARGET_DURATION,
        'n_mfcc': N_MFCC,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'top_db': TOP_DB,
    },
    'split_info': {
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'total_samples': int(len(X)),
        'random_state': RANDOM_STATE,
    },
    'data_dtype': 'uint8',
    'value_range': [0, 255],
    'failed_indices': failed_indices,
}

metadata_path = os.path.join(PREPROCESSED_DIR, 'metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ metadata.json saved")

print(f"\n{'='*60}")
print(f"✅ DATA PREPARATION SELESAI")
print(f"{'='*60}")
print(f"  Total file tersimpan di: {PREPROCESSED_DIR}")
print(f"  Total ukuran: ~{(X_train.nbytes + X_val.nbytes + X_test.nbytes)/1e9:.2f} GB")
print(f"\n  File yang dihasilkan:")
print(f"    - X_train.npy, X_val.npy, X_test.npy (citra MFCC uint8)")
print(f"    - y_train.npy, y_val.npy, y_test.npy (label integer)")
print(f"    - metadata.json (konfigurasi & class names)")
print(f"{'='*60}")
