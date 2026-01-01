"""
PTB-XL ECG Arrhythmia Classification - Kaggle Training Script (Memory-Efficient)
================================================================================
500Hz, Focal Loss, 1D Augmentations, Generator-based loading

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import wfdb

# ============================================================================
# CONFIGURATION
# ============================================================================

IS_KAGGLE = Path('/kaggle/input').exists()

if IS_KAGGLE:
    DATA_DIR = Path('/kaggle/input/ptbxl-ekg')
    OUTPUT_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('./data/ptb-xl')
    OUTPUT_DIR = Path('./output')
    OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🔍 Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Data directory: {DATA_DIR}")

# Model configuration - 500Hz
TARGET_CLASSES = 14
SAMPLING_RATE = 500  # 500Hz for higher resolution
SIGNAL_LENGTH = 5000  # 10 seconds @ 500Hz
NUM_LEADS = 12
BATCH_SIZE = 16  # Reduced for memory
EPOCHS = 50
LEARNING_RATE = 0.001

# Focal Loss parameters
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# ============================================================================
# FOCAL LOSS FUNCTION
# ============================================================================

def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Focal Loss for handling class imbalance.
    Focuses learning on hard, misclassified examples.
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fn

# ============================================================================
# 1D AUGMENTATION FUNCTIONS
# ============================================================================

def augment_signal(signal):
    """
    Apply 1D augmentations to ECG signal.
    - Gaussian noise
    - Amplitude scaling
    - Baseline wander
    """
    # Gaussian noise (10% chance, std=0.05)
    if tf.random.uniform([]) < 0.1:
        noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=0.05)
        signal = signal + noise
    
    # Amplitude scaling (20% chance, scale 0.8-1.2)
    if tf.random.uniform([]) < 0.2:
        scale = tf.random.uniform([], 0.8, 1.2)
        signal = signal * scale
    
    # Baseline wander (15% chance)
    if tf.random.uniform([]) < 0.15:
        t = tf.linspace(0.0, 1.0, SIGNAL_LENGTH)
        freq = tf.random.uniform([], 0.1, 0.5)
        amplitude = tf.random.uniform([], 0.05, 0.15)
        wander = amplitude * tf.sin(2.0 * 3.14159 * freq * t)
        wander = tf.expand_dims(wander, axis=-1)
        wander = tf.tile(wander, [1, NUM_LEADS])
        signal = signal + wander
    
    return signal

# ============================================================================
# MEMORY-EFFICIENT DATA LOADING
# ============================================================================

def load_single_signal(path_str):
    """Load and preprocess a single ECG signal."""
    try:
        record = wfdb.rdrecord(path_str)
        signal = record.p_signal.astype(np.float32)
        
        # Pad/truncate to SIGNAL_LENGTH
        if signal.shape[0] < SIGNAL_LENGTH:
            pad_width = ((0, SIGNAL_LENGTH - signal.shape[0]), (0, 0))
            signal = np.pad(signal, pad_width, mode='constant')
        else:
            signal = signal[:SIGNAL_LENGTH]
        
        # Ensure 12 leads
        if signal.shape[1] < NUM_LEADS:
            pad_width = ((0, 0), (0, NUM_LEADS - signal.shape[1]))
            signal = np.pad(signal, pad_width, mode='constant')
        elif signal.shape[1] > NUM_LEADS:
            signal = signal[:, :NUM_LEADS]
        
        # Z-score normalize per lead
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True) + 1e-8
        signal = (signal - mean) / std
        
        return signal
    except Exception as e:
        print(f"Error loading {path_str}: {e}")
        return np.zeros((SIGNAL_LENGTH, NUM_LEADS), dtype=np.float32)

def create_tf_dataset(file_paths, labels, batch_size=BATCH_SIZE, augment=False, shuffle=False):
    """Create memory-efficient tf.data.Dataset that loads signals on-the-fly."""
    
    def tf_load_signal(path, label):
        signal = tf.py_function(
            lambda p: load_single_signal(p.numpy().decode('utf-8')),
            [path],
            tf.float32
        )
        signal.set_shape([SIGNAL_LENGTH, NUM_LEADS])
        return signal, label
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(tf_load_signal, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(
            lambda x, y: (augment_signal(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("📊 STEP 1: Loading PTB-XL Metadata")
print("="*70)

metadata_file = DATA_DIR / 'ptbxl_database.csv'
if not metadata_file.exists():
    print(f"❌ ERROR: Cannot find metadata file at {metadata_file}")
    sys.exit(1)

df = pd.read_csv(metadata_file)
print(f"✅ Loaded {len(df)} ECG records from metadata")

df['scp_codes'] = df['scp_codes'].apply(lambda x: eval(x) if isinstance(x, str) else {})

CLASS_MAPPING = {
    'NORM': 'Normal', 'MI': 'Myocardial Infarction', 'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance', 'HYP': 'Hypertrophy', 'AFIB': 'Atrial Fibrillation',
    'LAFB': 'Left Anterior Fascicular Block', 'LPFB': 'Left Posterior Fascicular Block',
    'RBBB': 'Right Bundle Branch Block', 'LBBB': 'Left Bundle Branch Block',
    'WPW': 'Wolff-Parkinson-White', 'PVC': 'Premature Ventricular Contraction',
    'PACE': 'Pacemaker', 'IVCD': 'Intraventricular Conduction Delay'
}

def extract_primary_label(scp_dict):
    if not scp_dict:
        return 'NORM'
    valid_codes = [code for code, conf in scp_dict.items() if conf >= 50]
    for code in CLASS_MAPPING.keys():
        if code in valid_codes:
            return code
    return 'NORM'

df['primary_label'] = df['scp_codes'].apply(extract_primary_label)
df = df[df['primary_label'].isin(CLASS_MAPPING.keys())]
print(f"✅ Filtered to {len(df)} records with valid labels")

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['primary_label'])
num_classes = len(label_encoder.classes_)

print(f"\n📋 Class distribution:")
for label, count in df['primary_label'].value_counts().items():
    print(f"   {label:30s}: {count:5d} ({count/len(df)*100:.1f}%)")

# ============================================================================
# 2. BUILD FILE PATHS (500Hz records)
# ============================================================================

print("\n" + "="*70)
print("📁 STEP 2: Building File Paths for 500Hz Records")
print("="*70)

# Use filename_hr for 500Hz data
df['file_path'] = df['filename_hr'].apply(
    lambda x: str(DATA_DIR / x.replace('.hea', '').replace('.dat', ''))
)

# Verify files exist
valid_mask = df['file_path'].apply(lambda p: Path(p + '.hea').exists())
df = df[valid_mask]
print(f"✅ Found {len(df)} valid 500Hz records")

file_paths = df['file_path'].values
labels = df['label_encoded'].values

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("✂️  STEP 3: Creating Train/Validation/Test Splits")
print("="*70)

if 'strat_fold' in df.columns:
    print("Using PTB-XL official stratified folds...")
    train_mask = df['strat_fold'].isin(range(1, 9)).values
    val_mask = (df['strat_fold'] == 9).values
    test_mask = (df['strat_fold'] == 10).values
    
    train_paths, train_labels = file_paths[train_mask], labels[train_mask]
    val_paths, val_labels = file_paths[val_mask], labels[val_mask]
    test_paths, test_labels = file_paths[test_mask], labels[test_mask]
else:
    print("Creating custom 70/15/15 splits...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

print(f"✅ Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# Convert labels to one-hot
train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)
val_labels_cat = keras.utils.to_categorical(val_labels, num_classes)
test_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

# Create tf.data datasets
print("\n📦 Creating tf.data pipelines...")
train_dataset = create_tf_dataset(train_paths, train_labels_cat, BATCH_SIZE, augment=True, shuffle=True)
val_dataset = create_tf_dataset(val_paths, val_labels_cat, BATCH_SIZE, augment=False, shuffle=False)
test_dataset = create_tf_dataset(test_paths, test_labels_cat, BATCH_SIZE, augment=False, shuffle=False)

# ============================================================================
# 4. MODEL ARCHITECTURE: ResNet1D (500Hz)
# ============================================================================

print("\n" + "="*70)
print("🏗️  STEP 4: Building ResNet1D Architecture (500Hz)")
print("="*70)

def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.2):
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet1d(input_shape=(SIGNAL_LENGTH, NUM_LEADS), num_classes=TARGET_CLASSES):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(64, kernel_size=15, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=512, stride=2)
    
    # Global pooling + classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

model = build_resnet1d(input_shape=(SIGNAL_LENGTH, NUM_LEADS), num_classes=num_classes)
print("\n📐 Model Architecture:")
model.summary()

# Class weights
unique, counts = np.unique(train_labels, return_counts=True)
total = len(train_labels)
class_weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}

print(f"\n⚖️  Class weights:")
for cls, weight in class_weights.items():
    print(f"   Class {cls}: {weight:.2f}")

# ============================================================================
# 5. COMPILE & TRAIN WITH FOCAL LOSS
# ============================================================================

print("\n" + "="*70)
print("🎯 STEP 5: Training with Focal Loss")
print("="*70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

callbacks = [
    ModelCheckpoint(
        str(OUTPUT_DIR / 'model.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

print(f"\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"   Focal Loss: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")

steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(val_paths) // BATCH_SIZE

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# ============================================================================
# 6. EVALUATION
# ============================================================================

print("\n" + "="*70)
print("📊 STEP 6: Evaluating Model")
print("="*70)

test_steps = len(test_paths) // BATCH_SIZE
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    test_dataset, steps=test_steps, verbose=0
)

f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8)

print(f"\n🎯 Test Results:")
print(f"   Loss:      {test_loss:.4f}")
print(f"   Accuracy:  {test_acc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# ============================================================================
# 7. SAVE ARTIFACTS
# ============================================================================

print("\n" + "="*70)
print("💾 STEP 7: Saving Artifacts")
print("="*70)

classes_file = OUTPUT_DIR / 'classes.npy'
np.save(classes_file, label_encoder.classes_)
print(f"✅ Saved classes to: {classes_file}")

weights_file = OUTPUT_DIR / 'model.weights.h5'
print(f"✅ Model weights saved to: {weights_file}")

zip_file = OUTPUT_DIR / 'output.zip'
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(weights_file, 'model.weights.h5')
    zipf.write(classes_file, 'classes.npy')

print(f"✅ Created: {zip_file}")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)