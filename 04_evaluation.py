# -*- coding: utf-8 -*-
"""
=============================================================================
 04_evaluation.py — Evaluation (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 Evaluasi model menggunakan:
 1. Overall Accuracy pada test set
 2. Classification Report (Precision, Recall, F1-Score per kelas)
 3. Confusion Matrix (84×84) + heatmap visualization
 4. Analisis huruf dengan performa tertinggi & terendah
 5. Visualisasi prediksi benar & salah
=============================================================================
"""

# ===========================================================================
# CELL 1: Install & Import Dependencies
# ===========================================================================
# !pip install -q tensorflow scikit-learn matplotlib seaborn

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10

# ===========================================================================
# CELL 2: Mount Google Drive & Setup
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '04_evaluation')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
# CELL 3: Load Model & Test Data
# ===========================================================================
print("⏳ Memuat model dan data test...")

# Load model terbaik (Fase 2)
model_path = os.path.join(MODEL_DIR, 'best_model_phase2.keras')
model = keras.models.load_model(model_path)
print(f"  ✅ Model dimuat: {model_path}")

# Load test data
X_test = np.load(os.path.join(PREPROCESSED_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PREPROCESSED_DIR, 'y_test.npy'))

# Load metadata
with open(os.path.join(PREPROCESSED_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
num_classes = metadata['num_classes']

print(f"  ✅ X_test: {X_test.shape}")
print(f"  ✅ y_test: {y_test.shape}, kelas unik: {len(np.unique(y_test))}")

# ===========================================================================
# CELL 4: Prediksi pada Test Set
# ===========================================================================
print("\n⏳ Melakukan prediksi pada test set...")

# Normalisasi ke [0, 1]
X_test_normalized = X_test.astype(np.float32) / 255.0

# Prediksi probabilitas
y_pred_proba = model.predict(X_test_normalized, batch_size=32, verbose=1)

# Prediksi kelas (argmax)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confidence untuk prediksi terpilih
y_pred_confidence = np.max(y_pred_proba, axis=1)

print(f"\n  ✅ Prediksi selesai: {len(y_pred)} sampel")
print(f"  Mean confidence: {y_pred_confidence.mean():.4f}")

# ===========================================================================
# CELL 5: Overall Metrics
# ===========================================================================
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n{'='*60}")
print(f"📊 HASIL EVALUASI MODEL")
print(f"{'='*60}")
print(f"\n  Overall Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n  Macro Average:")
print(f"    Precision          : {precision_macro:.4f}")
print(f"    Recall             : {recall_macro:.4f}")
print(f"    F1-Score           : {f1_macro:.4f}")
print(f"\n  Weighted Average:")
print(f"    Precision          : {precision_weighted:.4f}")
print(f"    Recall             : {recall_weighted:.4f}")
print(f"    F1-Score           : {f1_weighted:.4f}")
print(f"{'='*60}")

# ===========================================================================
# CELL 6: Classification Report (Per-Kelas)
# ===========================================================================
print("\n📋 Classification Report (Per-Kelas):\n")

report_str = classification_report(
    y_test, y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)
print(report_str)

# Simpan report sebagai dict untuk CSV
report_dict = classification_report(
    y_test, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

# Konversi ke DataFrame dan simpan
import pandas as pd

report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = os.path.join(RESULTS_DIR, 'classification_report.csv')
report_df.to_csv(report_csv_path)
print(f"✅ Classification report disimpan: {report_csv_path}")

# ===========================================================================
# CELL 7: Confusion Matrix
# ===========================================================================
print("\n🔲 Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

# --- Plot Confusion Matrix (Full 84×84) ---
fig, ax = plt.subplots(figsize=(28, 24))

sns.heatmap(
    cm, annot=False, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    ax=ax, linewidths=0.5
)
ax.set_xlabel('Prediksi', fontsize=14)
ax.set_ylabel('Aktual', fontsize=14)
ax.set_title('Confusion Matrix — MobileNetV2 + MFCC\n'
             f'(Overall Accuracy: {accuracy*100:.2f}%)',
             fontsize=16, fontweight='bold')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_full.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix (full) disimpan.")

# --- Normalized Confusion Matrix ---
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

fig, ax = plt.subplots(figsize=(28, 24))
sns.heatmap(
    cm_normalized, annot=False, fmt='.2f', cmap='YlOrRd',
    xticklabels=class_names, yticklabels=class_names,
    ax=ax, vmin=0, vmax=1, linewidths=0.5
)
ax.set_xlabel('Prediksi', fontsize=14)
ax.set_ylabel('Aktual', fontsize=14)
ax.set_title('Normalized Confusion Matrix — MobileNetV2 + MFCC',
             fontsize=16, fontweight='bold')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_normalized.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Normalized confusion matrix disimpan.")

# ===========================================================================
# CELL 8: Analisis Per-Kelas (Top & Bottom Performers)
# ===========================================================================
print("\n📊 Analisis Per-Kelas...")

# Hitung akurasi per kelas
per_class_accuracy = cm_normalized.diagonal()
per_class_df = pd.DataFrame({
    'Kelas': class_names,
    'Accuracy': per_class_accuracy,
    'Precision': [report_dict.get(name, {}).get('precision', 0) for name in class_names],
    'Recall': [report_dict.get(name, {}).get('recall', 0) for name in class_names],
    'F1-Score': [report_dict.get(name, {}).get('f1-score', 0) for name in class_names],
    'Support': [report_dict.get(name, {}).get('support', 0) for name in class_names],
}).sort_values('Accuracy', ascending=False)

# Top 10 kelas terbaik
print("\n🏆 Top 10 Kelas dengan Akurasi Tertinggi:")
print(per_class_df.head(10).to_string(index=False))

# Bottom 10 kelas terburuk
print("\n⚠️  Bottom 10 Kelas dengan Akurasi Terendah:")
print(per_class_df.tail(10).to_string(index=False))

# --- Plot Per-Class Accuracy ---
fig, ax = plt.subplots(figsize=(20, 8))
sorted_df = per_class_df.sort_values('Accuracy', ascending=True)
colors = plt.cm.RdYlGn(sorted_df['Accuracy'])
ax.barh(range(len(sorted_df)), sorted_df['Accuracy'], color=colors, edgecolor='white')
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df['Kelas'], fontsize=6)
ax.set_xlabel('Accuracy', fontsize=13)
ax.set_title('Akurasi Per-Kelas (Sorted)', fontsize=15, fontweight='bold')
ax.axvline(x=accuracy, color='red', linestyle='--', linewidth=2,
           label=f'Overall Acc: {accuracy:.4f}')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()

# Simpan per-class metrics
per_class_df.to_csv(os.path.join(RESULTS_DIR, 'per_class_metrics.csv'), index=False)
print("✅ Per-class metrics disimpan.")

# ===========================================================================
# CELL 9: Visualisasi Prediksi Benar & Salah
# ===========================================================================
print("\n🔍 Visualisasi Sampel Prediksi...")

# Cari prediksi benar (high confidence)
correct_mask = y_pred == y_test
correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(~correct_mask)[0]

# --- Prediksi Benar ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Sort by confidence (highest first)
correct_conf_sorted = np.argsort(-y_pred_confidence[correct_indices])

for i in range(min(8, len(correct_conf_sorted))):
    idx = correct_indices[correct_conf_sorted[i]]
    ax = axes[i]
    ax.imshow(X_test[idx])
    ax.set_title(
        f"✅ {class_names[y_test[idx]]}\n"
        f"Pred: {class_names[y_pred[idx]]}\n"
        f"Conf: {y_pred_confidence[idx]:.2%}",
        fontsize=9, fontweight='bold', color='green'
    )
    ax.axis('off')

plt.suptitle('Prediksi BENAR (Confidence Tertinggi)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'prediksi_benar.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Prediksi Salah ---
if len(incorrect_indices) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Sort by confidence (highest first - most confident wrong predictions)
    incorrect_conf_sorted = np.argsort(-y_pred_confidence[incorrect_indices])

    for i in range(min(8, len(incorrect_conf_sorted))):
        idx = incorrect_indices[incorrect_conf_sorted[i]]
        ax = axes[i]
        ax.imshow(X_test[idx])
        ax.set_title(
            f"❌ Aktual: {class_names[y_test[idx]]}\n"
            f"Pred: {class_names[y_pred[idx]]}\n"
            f"Conf: {y_pred_confidence[idx]:.2%}",
            fontsize=9, fontweight='bold', color='red'
        )
        ax.axis('off')

    # Hide unused axes
    for i in range(min(8, len(incorrect_conf_sorted)), 8):
        axes[i].axis('off')

    plt.suptitle('Prediksi SALAH (Confidence Tertinggi — Most Confident Errors)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediksi_salah.png'), dpi=150, bbox_inches='tight')
    plt.show()

print("✅ Visualisasi prediksi disimpan.")

# ===========================================================================
# CELL 10: Simpan Ringkasan Evaluasi
# ===========================================================================
evaluation_summary = {
    'model_path': model_path,
    'test_samples': int(len(y_test)),
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
    },
    'per_class_accuracy': {
        'best_class': per_class_df.iloc[0]['Kelas'],
        'best_accuracy': float(per_class_df.iloc[0]['Accuracy']),
        'worst_class': per_class_df.iloc[-1]['Kelas'],
        'worst_accuracy': float(per_class_df.iloc[-1]['Accuracy']),
    },
    'prediction_stats': {
        'total_correct': int(correct_mask.sum()),
        'total_incorrect': int((~correct_mask).sum()),
        'mean_confidence': float(y_pred_confidence.mean()),
        'mean_confidence_correct': float(y_pred_confidence[correct_mask].mean()) if correct_mask.sum() > 0 else 0,
        'mean_confidence_incorrect': float(y_pred_confidence[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0,
    }
}

summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.json')
with open(summary_path, 'w') as f:
    json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"✅ EVALUASI SELESAI")
print(f"{'='*60}")
print(f"  📊 Overall Accuracy  : {accuracy*100:.2f}%")
print(f"  📊 F1-Score (Macro)  : {f1_macro:.4f}")
print(f"  📊 F1-Score (Weight) : {f1_weighted:.4f}")
print(f"  🏆 Kelas terbaik     : {per_class_df.iloc[0]['Kelas']} ({per_class_df.iloc[0]['Accuracy']:.2%})")
print(f"  ⚠️  Kelas terburuk    : {per_class_df.iloc[-1]['Kelas']} ({per_class_df.iloc[-1]['Accuracy']:.2%})")
print(f"\n  File tersimpan di: {RESULTS_DIR}/")
print(f"{'='*60}")
