"""
PTB-XL ECG Arrhythmia Classification - Kaggle Training Script
==============================================================

This script trains a ResNet1D model on the PTB-XL dataset for 14-class arrhythmia detection.
It's designed to run on Kaggle with the PTB-XL dataset mounted at /kaggle/input/ptbxl-ekg/

DATASET SETUP ON KAGGLE:
1. Go to https://www.kaggle.com/
2. Create a new notebook
3. Add dataset: Search for "PTB-XL" and add "khyeh0719/ptb-xl-dataset" or upload from PhysioNet
4. Paste this script and run!

MODEL OUTPUT:
- model_weights.h5: Best model weights
- classes.npy: Label encoder classes (maps indices to disease names)
- output.zip: Combined package ready for download

Authors: Elissa Matlock, Eugene Ho, Jonah Zembower
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect environment
IS_KAGGLE = Path('/kaggle/input').exists()

# Paths (adjust for local testing if needed)
if IS_KAGGLE:
    DATA_DIR = Path('/kaggle/input/ptbxl-ekg')
    OUTPUT_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('./data/ptb-xl')  # For local testing
    OUTPUT_DIR = Path('./output')
    OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🔍 Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# Model configuration
TARGET_CLASSES = 14
SIGNAL_LENGTH = 1000  # 10 seconds @ 100Hz
N_LEADS = 12  # 12-lead ECG
SAMPLING_RATE = 100  # Hz (using 100Hz for efficiency)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("📊 STEP 1: Loading PTB-XL Dataset")
print("="*70)

# Load metadata CSV
metadata_file = DATA_DIR / 'ptbxl_database.csv'
if not metadata_file.exists():
    print(f"❌ ERROR: Cannot find metadata file at {metadata_file}")
    print("Please ensure PTB-XL dataset is properly mounted in Kaggle.")
    sys.exit(1)

df = pd.read_csv(metadata_file)
print(f"✅ Loaded {len(df)} ECG records from metadata")

# Parse SCP codes (diagnostic labels)
df['scp_codes'] = df['scp_codes'].apply(lambda x: eval(x) if isinstance(x, str) else {})

# Map SCP codes to 14 diagnostic classes
# Based on PTB-XL superclass categories
CLASS_MAPPING = {
    'NORM': 'Normal',
    'MI': 'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy',
    'AFIB': 'Atrial Fibrillation',
    'LAFB': 'Left Anterior Fascicular Block',
    'LPFB': 'Left Posterior Fascicular Block',
    'RBBB': 'Right Bundle Branch Block',
    'LBBB': 'Left Bundle Branch Block',
    'WPW': 'Wolff-Parkinson-White',
    'PVC': 'Premature Ventricular Contraction',
    'PACE': 'Pacemaker',
    'IVCD': 'Intraventricular Conduction Delay'
}

def extract_primary_label(scp_dict):
    """Extract primary diagnostic label from SCP codes"""
    if not scp_dict:
        return 'NORM'
    
    # Get all codes with confidence >= 50%
    valid_codes = [code for code, conf in scp_dict.items() if conf >= 50]
    
    # Priority-based selection for multi-label cases
    for code in CLASS_MAPPING.keys():
        if code in valid_codes:
            return code
    
    return 'NORM'  # Default to normal if no match

df['primary_label'] = df['scp_codes'].apply(extract_primary_label)

# Filter to only include records with valid labels
df = df[df['primary_label'].isin(CLASS_MAPPING.keys())]
print(f"✅ Filtered to {len(df)} records with valid diagnostic labels")

# Label encoding
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['primary_label'])

print(f"\n📋 Class distribution:")
for label, count in df['primary_label'].value_counts().items():
    print(f"   {label:30s}: {count:5d} samples ({count/len(df)*100:.1f}%)")

# ============================================================================
# 2. SIGNAL LOADING & PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("🔊 STEP 2: Loading and Preprocessing ECG Signals")
print("="*70)

def load_ecg_signal(filename_hr, data_dir):
    """
    Load ECG signal from WFDB format (.hea + .dat files)
    Returns: numpy array of shape (signal_length, n_leads)
    """
    try:
        import wfdb
        
        # Remove extension if present
        record_name = str(filename_hr).replace('.hea', '').replace('.dat', '')
        record_path = data_dir / record_name
        
        # Read WFDB record
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal  # Shape: (samples, 12 leads)
        
        return signal
    except Exception as e:
        print(f"⚠️  Error loading {filename_hr}: {e}")
        return None

def preprocess_signal(signal, target_length=SIGNAL_LENGTH):
    """
    Preprocess ECG signal:
    1. Z-score normalization (per lead)
    2. Pad or truncate to target length
    """
    if signal is None or len(signal) == 0:
        return None
    
    # Z-score normalization per lead
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
    
    # Pad or truncate
    current_length = signal.shape[0]
    if current_length < target_length:
        # Pad with zeros
        pad_width = ((0, target_length - current_length), (0, 0))
        signal = np.pad(signal, pad_width, mode='constant', constant_values=0)
    elif current_length > target_length:
        # Truncate
        signal = signal[:target_length, :]
    
    return signal

# Load all signals
print("Loading ECG signals... (this may take a few minutes)")
signals = []
labels = []
valid_indices = []

for idx, row in df.iterrows():
    signal = load_ecg_signal(row['filename_hr'], DATA_DIR)
    if signal is not None:
        processed = preprocess_signal(signal)
        if processed is not None and processed.shape == (SIGNAL_LENGTH, N_LEADS):
            signals.append(processed)
            labels.append(row['label_encoded'])
            valid_indices.append(idx)
    
    if (idx + 1) % 1000 == 0:
        print(f"   Processed {idx + 1}/{len(df)} records...")

X = np.array(signals, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\n✅ Successfully loaded {len(X)} ECG signals")
print(f"   Signal shape: {X.shape}")
print(f"   Labels shape: {y.shape}")

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("✂️  STEP 3: Creating Train/Validation/Test Splits")
print("="*70)

# Use PTB-XL official splits if available, otherwise create our own
df_valid = df.loc[valid_indices]
if 'strat_fold' in df_valid.columns:
    print("Using PTB-XL official stratified folds...")
    # Folds 1-8: train, 9: val, 10: test
    train_mask = df_valid['strat_fold'].isin(range(1, 9))
    val_mask = df_valid['strat_fold'] == 9
    test_mask = df_valid['strat_fold'] == 10
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
else:
    print("Creating custom train/val/test splits (70/15/15)...")
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

print(f"✅ Train set: {len(X_train)} samples")
print(f"✅ Val set:   {len(X_val)} samples")
print(f"✅ Test set:  {len(X_test)} samples")

# Convert labels to categorical (one-hot encoding)
num_classes = len(np.unique(y))
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# ============================================================================
# 4. MODEL ARCHITECTURE: ResNet1D
# ============================================================================

print("\n" + "="*70)
print("🏗️  STEP 4: Building ResNet1D Architecture")
print("="*70)

def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.2):
    """
    Residual block with skip connection
    """
    # Main path
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (adjust dimensions if needed)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def build_resnet1d(input_shape=(SIGNAL_LENGTH, N_LEADS), num_classes=TARGET_CLASSES):
    """
    Build ResNet1D model for ECG classification
    
    Architecture:
    - Initial Conv1D layer
    - 4 residual blocks with increasing filters
    - Global Average Pooling
    - Dense classification head
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=512, stride=2)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Build model
model = build_resnet1d(input_shape=(SIGNAL_LENGTH, N_LEADS), num_classes=num_classes)

print("\n📐 Model Architecture:")
model.summary()

# Calculate class weights for imbalanced data
class_weights = {}
unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
for cls, count in zip(unique, counts):
    class_weights[cls] = total / (len(unique) * count)

print(f"\n⚖️  Class weights (for handling imbalance):")
for cls, weight in class_weights.items():
    print(f"   Class {cls}: {weight:.2f}")

# ============================================================================
# 5. MODEL COMPILATION & TRAINING
# ============================================================================

print("\n" + "="*70)
print("🎯 STEP 5: Compiling and Training Model")
print("="*70)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        str(OUTPUT_DIR / 'model_weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Initial learning rate: {LEARNING_RATE}")

# Train model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("📊 STEP 6: Evaluating Model Performance")
print("="*70)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    X_test, y_test_cat, 
    batch_size=BATCH_SIZE, 
    verbose=0
)

print(f"\n🎯 Test Set Results:")
print(f"   Loss:      {test_loss:.4f}")
print(f"   Accuracy:  {test_acc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8):.4f}")

# Per-class accuracy
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

print(f"\n📋 Per-Class Accuracy:")
for cls in range(num_classes):
    mask = y_test_classes == cls
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == cls).mean()
        class_name = label_encoder.inverse_transform([cls])[0]
        print(f"   {class_name:30s}: {acc:.4f} ({mask.sum()} samples)")

# ============================================================================
# 7. SAVE ARTIFACTS
# ============================================================================

print("\n" + "="*70)
print("💾 STEP 7: Saving Model Artifacts")
print("="*70)

# Save label encoder classes
classes_file = OUTPUT_DIR / 'classes.npy'
np.save(classes_file, label_encoder.classes_)
print(f"✅ Saved label classes to: {classes_file}")

# Model weights already saved by ModelCheckpoint callback
weights_file = OUTPUT_DIR / 'model_weights.h5'
print(f"✅ Model weights saved to: {weights_file}")

# Create output.zip for easy download
zip_file = OUTPUT_DIR / 'output.zip'
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(weights_file, 'model_weights.h5')
    zipf.write(classes_file, 'classes.npy')

print(f"✅ Created output package: {zip_file}")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

if IS_KAGGLE:
    print("\n📥 To download your trained model:")
    print("   1. Look for 'output.zip' in the Output section (right panel)")
    print("   2. Click the download icon")
    print("   3. Extract the zip file to use model_weights.h5 and classes.npy")
else:
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")

print("\n📚 Next Steps:")
print("   1. Download output.zip from Kaggle")
print("   2. Integrate model_weights.h5 into your backend API")
print("   3. Use classes.npy for label decoding")
print("   4. Test with your mobile app!")

print("\n✨ Happy ECG analyzing! ❤️")
