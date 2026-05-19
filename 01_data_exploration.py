# -*- coding: utf-8 -*-
"""
=============================================================================
 01_data_exploration.py — Data Understanding (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 Tahap ini bertujuan untuk:
 1. Memuat dataset dari Google Drive (sudah diunduh)
 2. Memahami struktur dataset (84 kelas = 28 huruf × 3 harakat)
 3. Validasi & pembersihan data (file corrupt, silent, terlalu pendek)
 4. Menganalisis distribusi kelas, durasi, dan sampling rate
 5. Memvisualisasikan waveform, spectrogram, dan MFCC
=============================================================================
"""

# ===========================================================================
# CELL 1: Install & Import Dependencies
# ===========================================================================
# !pip install -q librosa matplotlib seaborn pandas soundfile tqdm

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from collections import Counter
from tqdm import tqdm

# Konfigurasi plot
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# ===========================================================================
# CELL 2: Mount Google Drive & Setup Direktori
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

# Direktori dataset yang sudah diunduh di Google Drive
DATASET_DIR = '/content/drive/MyDrive/dataset-huruf-hijaiyah'

# Direktori utama proyek untuk menyimpan hasil
BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '01_exploration')

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📁 Dataset directory : {DATASET_DIR}")
print(f"📁 Base directory    : {BASE_DIR}")
print(f"📁 Results directory : {RESULTS_DIR}")

# ===========================================================================
# CELL 3: Scan Struktur Dataset dari Google Drive
# ===========================================================================
print("\n⏳ Scanning struktur dataset di Google Drive...")

# Scan semua subfolder (setiap subfolder = 1 kelas)
class_folders = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

# Ekstensi audio yang didukung
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}

# Bangun daftar semua file audio + labelnya
all_files = []       # list of (file_path, class_name, class_index)
class_names = []     # list nama kelas (urut)

for cls_idx, folder_name in enumerate(class_folders):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    audio_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]
    class_names.append(folder_name)
    for fname in sorted(audio_files):
        all_files.append((os.path.join(folder_path, fname), folder_name, cls_idx))

num_classes = len(class_names)
num_samples = len(all_files)

print(f"\n{'='*60}")
print(f"📊 RINGKASAN DATASET")
print(f"{'='*60}")
print(f"  Lokasi dataset      : {DATASET_DIR}")
print(f"  Jumlah total sampel : {num_samples:,}")
print(f"  Jumlah kelas        : {num_classes}")
print(f"{'='*60}")

# Tampilkan daftar kelas
print(f"\n📋 Daftar {num_classes} Kelas:")
for i, name in enumerate(class_names):
    count = sum(1 for f in all_files if f[2] == i)
    print(f"  [{i:2d}] {name:25s} ({count} file)", end="\t" if (i + 1) % 3 != 0 else "\n")
print()

# ===========================================================================
# CELL 4: Analisis Distribusi Kelas
# ===========================================================================
print("\n📊 Analisis Distribusi Kelas...")

labels = [f[2] for f in all_files]
label_counts = Counter(labels)
class_distribution = pd.DataFrame({
    'Kelas': [class_names[i] for i in range(num_classes)],
    'Jumlah': [label_counts.get(i, 0) for i in range(num_classes)]
}).sort_values('Jumlah', ascending=False).reset_index(drop=True)

print(f"\n  Min sampel/kelas  : {class_distribution['Jumlah'].min()}")
print(f"  Max sampel/kelas  : {class_distribution['Jumlah'].max()}")
print(f"  Mean sampel/kelas : {class_distribution['Jumlah'].mean():.1f}")
print(f"  Std sampel/kelas  : {class_distribution['Jumlah'].std():.1f}")

# --- Plot Distribusi Kelas ---
fig, ax = plt.subplots(figsize=(20, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_classes))
bars = ax.bar(range(num_classes), class_distribution['Jumlah'], color=colors)
ax.set_xlabel('Kelas Huruf Hijaiyah', fontsize=14)
ax.set_ylabel('Jumlah Sampel', fontsize=14)
ax.set_title('Distribusi Jumlah Sampel per Kelas (84 Kelas)', fontsize=16, fontweight='bold')
ax.set_xticks(range(num_classes))
ax.set_xticklabels(class_distribution['Kelas'], rotation=90, fontsize=7)

mean_count = class_distribution['Jumlah'].mean()
ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, label=f'Rata-rata: {mean_count:.0f}')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distribusi_kelas.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot distribusi kelas disimpan.")

# ===========================================================================
# CELL 5: Validasi & Pembersihan Data
# ===========================================================================
print("\n🧹 VALIDASI & PEMBERSIHAN DATA...")
print("   Memeriksa setiap file audio untuk masalah kualitas...\n")

# Threshold pembersihan
MIN_DURATION = 0.15       # Minimal 0.15 detik (file terlalu pendek tidak informatif)
MAX_DURATION = 10.0       # Maksimal 10 detik (kemungkinan file salah)
MIN_RMS = 0.001           # Minimal RMS amplitude (deteksi file silent/kosong)
MIN_MAX_AMP = 0.01        # Minimal amplitudo puncak

# Simpan info setiap file
file_info = []            # list of dict: path, class, sr, duration, rms, max_amp, status
corrupted_files = []      # file yang error saat dibaca
silent_files = []         # file yang terlalu pelan / silent
too_short_files = []      # file terlalu pendek
too_long_files = []       # file terlalu panjang
valid_files = []          # file yang lolos validasi

for file_path, class_name, cls_idx in tqdm(all_files, desc="Validasi audio"):
    info = {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'class_name': class_name,
        'class_idx': cls_idx,
        'status': 'valid',
        'issue': ''
    }

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=True)

        duration = len(y) / sr
        max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0
        rms = float(np.sqrt(np.mean(y**2))) if len(y) > 0 else 0

        info['sr'] = int(sr)
        info['duration'] = round(duration, 4)
        info['max_amp'] = round(max_amp, 6)
        info['rms'] = round(rms, 6)
        info['samples'] = len(y)

        # --- Cek masalah ---
        issues = []

        # File terlalu pendek
        if duration < MIN_DURATION:
            issues.append(f'terlalu pendek ({duration:.3f}s < {MIN_DURATION}s)')
            too_short_files.append(file_path)

        # File terlalu panjang
        if duration > MAX_DURATION:
            issues.append(f'terlalu panjang ({duration:.1f}s > {MAX_DURATION}s)')
            too_long_files.append(file_path)

        # File silent / hampir silent
        if rms < MIN_RMS or max_amp < MIN_MAX_AMP:
            issues.append(f'silent/kosong (RMS={rms:.6f}, MaxAmp={max_amp:.6f})')
            silent_files.append(file_path)

        # File kosong (0 sampel)
        if len(y) == 0:
            issues.append('file kosong (0 sampel)')
            corrupted_files.append(file_path)

        if issues:
            info['status'] = 'bermasalah'
            info['issue'] = '; '.join(issues)
        else:
            valid_files.append((file_path, class_name, cls_idx))

    except Exception as e:
        info['status'] = 'corrupt'
        info['issue'] = str(e)
        info['sr'] = 0
        info['duration'] = 0
        info['max_amp'] = 0
        info['rms'] = 0
        info['samples'] = 0
        corrupted_files.append(file_path)

    file_info.append(info)

# Konversi ke DataFrame
df_info = pd.DataFrame(file_info)

# --- Laporan Pembersihan ---
total_bermasalah = len(corrupted_files) + len(silent_files) + len(too_short_files) + len(too_long_files)

print(f"\n{'='*60}")
print(f"🧹 HASIL VALIDASI DATA")
print(f"{'='*60}")
print(f"  Total file diperiksa : {num_samples}")
print(f"  ✅ File valid         : {len(valid_files)}")
print(f"  ❌ File corrupt/error : {len(corrupted_files)}")
print(f"  🔇 File silent/kosong : {len(silent_files)}")
print(f"  ⏱️  File terlalu pendek: {len(too_short_files)} (< {MIN_DURATION}s)")
print(f"  ⏱️  File terlalu panjang: {len(too_long_files)} (> {MAX_DURATION}s)")
print(f"{'='*60}")

if corrupted_files:
    print(f"\n  ❌ File corrupt (max 10 ditampilkan):")
    for f in corrupted_files[:10]:
        print(f"     - {os.path.basename(f)}")

if silent_files:
    print(f"\n  🔇 File silent (max 10 ditampilkan):")
    for f in silent_files[:10]:
        print(f"     - {os.path.basename(f)}")

if too_short_files:
    print(f"\n  ⏱️  File terlalu pendek (max 10 ditampilkan):")
    for f in too_short_files[:10]:
        print(f"     - {os.path.basename(f)}")

# Simpan laporan validasi
df_issues = df_info[df_info['status'] != 'valid']
if len(df_issues) > 0:
    issues_path = os.path.join(RESULTS_DIR, 'file_bermasalah.csv')
    df_issues.to_csv(issues_path, index=False)
    print(f"\n  📄 Daftar file bermasalah disimpan: {issues_path}")

print(f"\n💡 Dataset yang akan digunakan: {len(valid_files)} file "
      f"({len(valid_files)/num_samples*100:.1f}% dari total)")

# ===========================================================================
# CELL 6: Analisis Karakteristik Audio (Hanya File Valid)
# ===========================================================================
print("\n🔊 Analisis Karakteristik Audio (file valid saja)...")

df_valid = df_info[df_info['status'] == 'valid'].copy()

durations = df_valid['duration'].values
sampling_rates = df_valid['sr'].values
amplitudes_max = df_valid['max_amp'].values
amplitudes_rms = df_valid['rms'].values

print(f"\n{'='*60}")
print(f"🔊 STATISTIK AUDIO (FILE VALID)")
print(f"{'='*60}")
print(f"  Jumlah file valid   : {len(df_valid)}")
print(f"  Sampling rate unik  : {np.unique(sampling_rates)}")
print(f"  Durasi (detik):")
print(f"    Min   : {durations.min():.3f}")
print(f"    Max   : {durations.max():.3f}")
print(f"    Mean  : {durations.mean():.3f}")
print(f"    Median: {np.median(durations):.3f}")
print(f"    Std   : {durations.std():.3f}")
print(f"  Amplitudo Max:")
print(f"    Min   : {amplitudes_max.min():.4f}")
print(f"    Max   : {amplitudes_max.max():.4f}")
print(f"    Mean  : {amplitudes_max.mean():.4f}")
print(f"  Amplitudo RMS:")
print(f"    Mean  : {amplitudes_rms.mean():.4f}")
print(f"{'='*60}")

# ===========================================================================
# CELL 7: Visualisasi Distribusi Durasi
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram durasi
axes[0].hist(durations, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=durations.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {durations.mean():.2f}s')
axes[0].axvline(x=np.median(durations), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(durations):.2f}s')
axes[0].set_xlabel('Durasi (detik)', fontsize=13)
axes[0].set_ylabel('Frekuensi', fontsize=13)
axes[0].set_title('Distribusi Durasi Audio', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# Box plot durasi per huruf (10 kelas sampel)
sample_class_indices = list(range(0, num_classes, max(1, num_classes // 10)))[:10]
duration_per_class = []
class_labels_for_plot = []
for cls_idx in sample_class_indices:
    cls_data = df_valid[df_valid['class_idx'] == cls_idx]
    for _, row in cls_data.iterrows():
        duration_per_class.append(row['duration'])
        class_labels_for_plot.append(class_names[cls_idx])

df_dur = pd.DataFrame({'Durasi': duration_per_class, 'Kelas': class_labels_for_plot})
sns.boxplot(data=df_dur, x='Kelas', y='Durasi', ax=axes[1], palette='Set2')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_title('Distribusi Durasi (10 Kelas Sampel)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Durasi (detik)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distribusi_durasi.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot distribusi durasi disimpan.")

# ===========================================================================
# CELL 8: Visualisasi Waveform Sampel
# ===========================================================================
print("\n🎵 Visualisasi Waveform Beberapa Sampel...")

sample_class_vis = list(range(0, num_classes, max(1, num_classes // 6)))[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, cls_idx in enumerate(sample_class_vis):
    cls_valid = [f for f in valid_files if f[2] == cls_idx]
    if not cls_valid:
        continue
    fpath = cls_valid[0][0]
    y, sr = librosa.load(fpath, sr=None, mono=True)

    librosa.display.waveshow(y, sr=sr, ax=axes[idx], color='steelblue')
    axes[idx].set_title(f"Kelas: {class_names[cls_idx]}", fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Waktu (detik)', fontsize=11)
    axes[idx].set_ylabel('Amplitudo', fontsize=11)

plt.suptitle('Waveform Sampel Audio (6 Kelas Representatif)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'waveform_sampel.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot waveform disimpan.")

# ===========================================================================
# CELL 9: Visualisasi MFCC Spectrogram Sampel
# ===========================================================================
print("\n🎨 Visualisasi MFCC Spectrogram...")

N_MFCC = 40  # Sesuai parameter yang telah disepakati

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, cls_idx in enumerate(sample_class_vis):
    cls_valid = [f for f in valid_files if f[2] == cls_idx]
    if not cls_valid:
        continue
    fpath = cls_valid[0][0]
    y, sr = librosa.load(fpath, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)

    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=512,
                                   ax=axes[idx], cmap='viridis')
    axes[idx].set_title(f"MFCC: {class_names[cls_idx]}", fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Waktu', fontsize=11)
    axes[idx].set_ylabel('Koefisien MFCC', fontsize=11)
    plt.colorbar(img, ax=axes[idx], format='%+2.0f')

plt.suptitle(f'MFCC Spectrogram (n_mfcc={N_MFCC})', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mfcc_spectrogram.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot MFCC spectrogram disimpan.")

# ===========================================================================
# CELL 10: Visualisasi Delta & Delta-Delta MFCC
# ===========================================================================
print("\n📐 Visualisasi MFCC + Delta + Delta-Delta (3 Channel)...")

# Ambil satu sampel valid dari kelas pertama
demo_file = valid_files[0][0]
y_demo, sr_demo = librosa.load(demo_file, sr=None, mono=True)

mfcc_demo = librosa.feature.mfcc(y=y_demo, sr=sr_demo, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
delta_demo = librosa.feature.delta(mfcc_demo)
delta2_demo = librosa.feature.delta(mfcc_demo, order=2)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
titles = ['Channel 1: MFCC', 'Channel 2: Delta MFCC', 'Channel 3: Delta-Delta MFCC']
data_list = [mfcc_demo, delta_demo, delta2_demo]

for ax, title, data in zip(axes, titles, data_list):
    img = librosa.display.specshow(data, x_axis='time', sr=sr_demo, hop_length=512,
                                   ax=ax, cmap='viridis')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Koefisien', fontsize=11)
    plt.colorbar(img, ax=ax, format='%+2.0f')

plt.suptitle(f'3-Channel Input: MFCC + Delta + Delta-Delta\n(Kelas: {valid_files[0][1]})',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mfcc_3channel.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 3-channel MFCC disimpan.")

# ===========================================================================
# CELL 11: Simpan Daftar File Valid & Ringkasan Eksplorasi
# ===========================================================================

# Simpan daftar file valid (akan dipakai di 02_preprocessing)
valid_files_data = [
    {'path': f[0], 'class_name': f[1], 'class_idx': f[2]}
    for f in valid_files
]
valid_files_path = os.path.join(BASE_DIR, 'valid_files.json')
with open(valid_files_path, 'w') as f:
    json.dump(valid_files_data, f, indent=2, ensure_ascii=False)
print(f"\n💾 Daftar file valid disimpan: {valid_files_path}")

# Simpan ringkasan
summary = {
    'dataset_source': DATASET_DIR,
    'total_files_scanned': int(num_samples),
    'total_valid_files': len(valid_files),
    'total_removed': num_samples - len(valid_files),
    'num_classes': int(num_classes),
    'class_names': class_names,
    'cleaning_stats': {
        'corrupted': len(corrupted_files),
        'silent': len(silent_files),
        'too_short': len(too_short_files),
        'too_long': len(too_long_files),
    },
    'cleaning_thresholds': {
        'min_duration': MIN_DURATION,
        'max_duration': MAX_DURATION,
        'min_rms': MIN_RMS,
        'min_max_amp': MIN_MAX_AMP,
    },
    'sampling_rates_unique': [int(x) for x in np.unique(sampling_rates)],
    'duration_stats': {
        'min': float(durations.min()),
        'max': float(durations.max()),
        'mean': float(durations.mean()),
        'median': float(np.median(durations)),
        'std': float(durations.std())
    },
    'amplitude_stats': {
        'max_mean': float(amplitudes_max.mean()),
        'rms_mean': float(amplitudes_rms.mean())
    },
    'class_distribution': {
        'min_samples': int(class_distribution['Jumlah'].min()),
        'max_samples': int(class_distribution['Jumlah'].max()),
        'mean_samples': float(class_distribution['Jumlah'].mean()),
    },
    'mfcc_params': {
        'n_mfcc': N_MFCC,
        'n_fft': 2048,
        'hop_length': 512,
        'channels': 'MFCC + Delta + Delta-Delta'
    }
}

summary_path = os.path.join(RESULTS_DIR, 'exploration_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

class_distribution.to_csv(os.path.join(RESULTS_DIR, 'class_distribution.csv'), index=False)

print(f"\n{'='*60}")
print(f"✅ EKSPLORASI & VALIDASI DATA SELESAI")
print(f"{'='*60}")
print(f"  📄 Summary         : {summary_path}")
print(f"  📄 File valid      : {valid_files_path} ({len(valid_files)} file)")
print(f"  📊 CSV distribusi  : {os.path.join(RESULTS_DIR, 'class_distribution.csv')}")
print(f"  🖼️  Gambar          : {RESULTS_DIR}/")
print(f"{'='*60}")
print(f"\n💡 Rekomendasi untuk tahap selanjutnya (02_preprocessing):")
print(f"   - Gunakan {len(valid_files)} file valid dari valid_files.json")
print(f"   - Target durasi audio: {np.median(durations):.2f}s (median)")
print(f"   - Sampling rate: {np.unique(sampling_rates)} Hz")
if class_distribution['Jumlah'].min() < class_distribution['Jumlah'].mean() * 0.5:
    print(f"   ⚠️ Ada ketidakseimbangan kelas! Min={class_distribution['Jumlah'].min()}, "
          f"Mean={class_distribution['Jumlah'].mean():.0f}")