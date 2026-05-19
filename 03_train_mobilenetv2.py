# -*- coding: utf-8 -*-
"""
=============================================================================
 03_train_mobilenetv2.py — Modeling (CRISP-DM)
=============================================================================
 Tugas Akhir: Implementasi MobileNetV2 dan Ekstraksi MFCC dalam
              Penilaian Pelafalan Huruf Hijaiyah
 Mahasiswa  : Mohammad Putra Fauzan Fatah (1227050075)
 Universitas: UIN Sunan Gunung Djati Bandung
=============================================================================
 Training 2 Fase:
   Fase 1: Transfer Learning — backbone MobileNetV2 dibekukan (frozen),
           hanya classification head yang dilatih
   Fase 2: Fine-tuning — 30 layer terakhir MobileNetV2 di-unfreeze
           dan dilatih bersama head dengan learning rate lebih kecil
=============================================================================
"""

# ===========================================================================
# CELL 1: Install & Import Dependencies
# ===========================================================================
# !pip install -q tensorflow matplotlib

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ===========================================================================
# CELL 2: Mount Google Drive & Setup Direktori
# ===========================================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/TA_Hijaiyah'
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '03_training')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===========================================================================
# CELL 3: Load Preprocessed Data
# ===========================================================================
print("⏳ Memuat data preprocessed...")

X_train = np.load(os.path.join(PREPROCESSED_DIR, 'X_train.npy'))
X_val = np.load(os.path.join(PREPROCESSED_DIR, 'X_val.npy'))
y_train = np.load(os.path.join(PREPROCESSED_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(PREPROCESSED_DIR, 'y_val.npy'))

with open(os.path.join(PREPROCESSED_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
num_classes = metadata['num_classes']
IMG_SIZE = metadata['image_size']

print(f"\n{'='*60}")
print(f"📊 DATA LOADED")
print(f"{'='*60}")
print(f"  X_train : {X_train.shape} ({X_train.dtype})")
print(f"  X_val   : {X_val.shape} ({X_val.dtype})")
print(f"  y_train : {y_train.shape}, kelas: {len(np.unique(y_train))}")
print(f"  y_val   : {y_val.shape}, kelas: {len(np.unique(y_val))}")
print(f"  Jumlah kelas: {num_classes}")
print(f"{'='*60}")

# ===========================================================================
# CELL 4: Buat tf.data.Dataset Pipeline
# ===========================================================================
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(X, y, batch_size, shuffle=True, augment=False):
    """Buat tf.data.Dataset dari numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000))

    def preprocess(image, label):
        # Konversi uint8 → float32 [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # One-hot encode label
        label = tf.one_hot(label, num_classes)
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        # Data augmentation untuk training
        data_augmentation = keras.Sequential([
            layers.RandomRotation(0.05),        # ±18°
            layers.RandomZoom(0.1),              # ±10%
            layers.RandomTranslation(0.1, 0.1),  # ±10% shift
            layers.RandomBrightness(0.1),        # ±10% brightness
        ])

        def apply_augmentation(image, label):
            image = data_augmentation(image, training=True)
            return image, label

        dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# Buat datasets
train_ds = create_dataset(X_train, y_train, BATCH_SIZE, shuffle=True, augment=True)
val_ds = create_dataset(X_val, y_val, BATCH_SIZE, shuffle=False, augment=False)

# Verifikasi
for images, labels in train_ds.take(1):
    print(f"\n  Batch shape  : images={images.shape}, labels={labels.shape}")
    print(f"  Value range  : [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    print(f"  Label example: {labels[0].numpy()[:5]}... (one-hot)")

# ===========================================================================
# CELL 5: Bangun Arsitektur Model
# ===========================================================================
def build_model(num_classes, img_size=224):
    """
    Arsitektur:
        Input (224, 224, 3)
            ↓
        MobileNetV2 (pretrained ImageNet, include_top=False)
            ↓
        GlobalAveragePooling2D
            ↓
        Dense(256, ReLU) + BatchNormalization + Dropout(0.5)
            ↓
        Dense(128, ReLU) + BatchNormalization + Dropout(0.3)
            ↓
        Dense(84, Softmax) → 84 kelas huruf hijaiyah
    """
    # Base model: MobileNetV2 pretrained on ImageNet
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model (akan di-unfreeze di Fase 2)
    base_model.trainable = False

    # Custom classification head
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Dropout(0.5, name='dropout_05')(x)
    x = layers.Dense(128, activation='relu', name='dense_128')(x)
    x = layers.BatchNormalization(name='bn_128')(x)
    x = layers.Dropout(0.3, name='dropout_03')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='MobileNetV2_Hijaiyah')

    return model, base_model

model, base_model = build_model(num_classes, IMG_SIZE)

# Tampilkan summary
print(f"\n{'='*60}")
print(f"🏗️  ARSITEKTUR MODEL")
print(f"{'='*60}")
model.summary()

total_params = model.count_params()
trainable_params = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
non_trainable_params = total_params - trainable_params
print(f"\n  Total params       : {total_params:,}")
print(f"  Trainable params   : {trainable_params:,}")
print(f"  Non-trainable params: {non_trainable_params:,}")

# ===========================================================================
# CELL 6: FASE 1 — Transfer Learning (Backbone Frozen)
# ===========================================================================
print(f"\n{'='*60}")
print(f"🚀 FASE 1: TRANSFER LEARNING (Backbone Frozen)")
print(f"{'='*60}")
print(f"  Optimizer   : Adam (LR=1e-3)")
print(f"  Epochs      : 20")
print(f"  Trainable   : Hanya classification head")
print(f"{'='*60}")

# Compile model untuk Fase 1
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks Fase 1
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model_phase1.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# Training Fase 1
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_phase1
)

print(f"\n✅ Fase 1 selesai!")
print(f"   Best val_accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

# ===========================================================================
# CELL 7: FASE 2 — Fine-tuning (30 Layer Terakhir)
# ===========================================================================
print(f"\n{'='*60}")
print(f"🔧 FASE 2: FINE-TUNING (30 Layer Terakhir)")
print(f"{'='*60}")
print(f"  Optimizer   : Adam (LR=1e-5)")
print(f"  Epochs      : 30")
print(f"  Unfreeze    : 30 layer terakhir MobileNetV2")
print(f"{'='*60}")

# Unfreeze 30 layer terakhir dari backbone
base_model.trainable = True

# Freeze semua kecuali 30 layer terakhir
num_layers = len(base_model.layers)
fine_tune_at = num_layers - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"\n  Total layers MobileNetV2  : {num_layers}")
print(f"  Frozen layers             : {fine_tune_at}")
print(f"  Trainable layers          : {num_layers - fine_tune_at}")

# Re-compile dengan learning rate lebih kecil
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Hitung ulang trainable params
trainable_params_p2 = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
print(f"  Trainable params (Fase 2) : {trainable_params_p2:,}")

# Callbacks Fase 2
callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model_phase2.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
]

# Training Fase 2
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_phase2
)

print(f"\n✅ Fase 2 selesai!")
print(f"   Best val_accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

# ===========================================================================
# CELL 8: Plot Learning Curves
# ===========================================================================
def plot_training_history(history_p1, history_p2, save_path):
    """Plot learning curves untuk kedua fase training."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Gabungkan history
    acc_p1 = history_p1.history['accuracy']
    val_acc_p1 = history_p1.history['val_accuracy']
    loss_p1 = history_p1.history['loss']
    val_loss_p1 = history_p1.history['val_loss']

    acc_p2 = history_p2.history['accuracy']
    val_acc_p2 = history_p2.history['val_accuracy']
    loss_p2 = history_p2.history['loss']
    val_loss_p2 = history_p2.history['val_loss']

    total_epochs_p1 = len(acc_p1)
    epochs_p1 = range(1, total_epochs_p1 + 1)
    epochs_p2 = range(total_epochs_p1 + 1, total_epochs_p1 + len(acc_p2) + 1)

    # --- Plot Accuracy ---
    axes[0].plot(epochs_p1, acc_p1, 'b-', linewidth=2, label='Train Acc (Fase 1)')
    axes[0].plot(epochs_p1, val_acc_p1, 'b--', linewidth=2, label='Val Acc (Fase 1)')
    axes[0].plot(epochs_p2, acc_p2, 'r-', linewidth=2, label='Train Acc (Fase 2)')
    axes[0].plot(epochs_p2, val_acc_p2, 'r--', linewidth=2, label='Val Acc (Fase 2)')
    axes[0].axvline(x=total_epochs_p1 + 0.5, color='gray', linestyle=':', linewidth=2,
                     label='Fase 1 → Fase 2')
    axes[0].set_xlabel('Epoch', fontsize=13)
    axes[0].set_ylabel('Accuracy', fontsize=13)
    axes[0].set_title('Model Accuracy', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # --- Plot Loss ---
    axes[1].plot(epochs_p1, loss_p1, 'b-', linewidth=2, label='Train Loss (Fase 1)')
    axes[1].plot(epochs_p1, val_loss_p1, 'b--', linewidth=2, label='Val Loss (Fase 1)')
    axes[1].plot(epochs_p2, loss_p2, 'r-', linewidth=2, label='Train Loss (Fase 2)')
    axes[1].plot(epochs_p2, val_loss_p2, 'r--', linewidth=2, label='Val Loss (Fase 2)')
    axes[1].axvline(x=total_epochs_p1 + 0.5, color='gray', linestyle=':', linewidth=2,
                     label='Fase 1 → Fase 2')
    axes[1].set_xlabel('Epoch', fontsize=13)
    axes[1].set_ylabel('Loss', fontsize=13)
    axes[1].set_title('Model Loss', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Learning Curves: MobileNetV2 + MFCC (2-Phase Training)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(
    history_phase1, history_phase2,
    os.path.join(RESULTS_DIR, 'learning_curves.png')
)
print("✅ Learning curves disimpan.")

# ===========================================================================
# CELL 9: Simpan Training History
# ===========================================================================
training_history = {
    'phase1': {
        'accuracy': [float(x) for x in history_phase1.history['accuracy']],
        'val_accuracy': [float(x) for x in history_phase1.history['val_accuracy']],
        'loss': [float(x) for x in history_phase1.history['loss']],
        'val_loss': [float(x) for x in history_phase1.history['val_loss']],
        'epochs': len(history_phase1.history['accuracy']),
        'config': {
            'optimizer': 'Adam',
            'learning_rate': 1e-3,
            'max_epochs': 20,
            'backbone_frozen': True,
        }
    },
    'phase2': {
        'accuracy': [float(x) for x in history_phase2.history['accuracy']],
        'val_accuracy': [float(x) for x in history_phase2.history['val_accuracy']],
        'loss': [float(x) for x in history_phase2.history['loss']],
        'val_loss': [float(x) for x in history_phase2.history['val_loss']],
        'epochs': len(history_phase2.history['accuracy']),
        'config': {
            'optimizer': 'Adam',
            'learning_rate': 1e-5,
            'max_epochs': 30,
            'fine_tune_layers': 30,
        }
    },
    'best_val_accuracy_phase1': float(max(history_phase1.history['val_accuracy'])),
    'best_val_accuracy_phase2': float(max(history_phase2.history['val_accuracy'])),
}

history_path = os.path.join(RESULTS_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ TRAINING SELESAI")
print(f"{'='*60}")
print(f"  Fase 1 — Best Val Accuracy: {training_history['best_val_accuracy_phase1']:.4f}")
print(f"  Fase 2 — Best Val Accuracy: {training_history['best_val_accuracy_phase2']:.4f}")
print(f"\n  📦 Model tersimpan:")
print(f"    - {os.path.join(MODEL_DIR, 'best_model_phase1.keras')}")
print(f"    - {os.path.join(MODEL_DIR, 'best_model_phase2.keras')}")
print(f"  📊 History: {history_path}")
print(f"  🖼️  Curves : {os.path.join(RESULTS_DIR, 'learning_curves.png')}")
print(f"{'='*60}")
