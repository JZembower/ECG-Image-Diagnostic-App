"""
PTB-XL ECG Arrhythmia Classification - Kaggle Training Script (Phase 1.5: 500Hz + Focal Loss + Augmentation)
====

This script trains a ResNet1D model on the PTB-XL dataset for 14-class arrhythmia detection
using 500Hz sampling (10 seconds -> 5000 samples).

NEW in Phase 1.5:
- Focal Loss (handles class imbalance better than cross-entropy)
- 1D signal augmentation (noise, scaling, baseline wander)
- tf.data pipeline for efficient training

Outputs (same as before for app compatibility):
- model.weights.h5: Best model weights
- classes.npy: Label encoder classes (maps indices to disease names)
- output.zip: Combined package ready for download

Designed to plug into the existing backend API with minimal changes.

Authors: Elissa Matlock, Eugene Ho, Jonah Zembower
Phase 1.5 adaptation: Focal Loss + Augmentation
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
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import tensorflow.keras.backend as K

# ====
# CONFIGURATION
# ====

# Auto-detect environment
IS_KAGGLE = Path("/kaggle/input").exists()

# Paths (adjust for local testing if needed)
if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/ptbxl-ekg")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path("./data/ptb-xl")  # For local testing
    OUTPUT_DIR = Path("./output")
    OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🔍 Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# Model configuration (Phase 1.5: 500Hz + Focal Loss + Augmentation)
TARGET_CLASSES = 14
SAMPLING_RATE = 500         # Hz
SIGNAL_LENGTH = 5000        # 10 seconds @ 500Hz
N_LEADS = 12                # 12-lead ECG
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# NEW: Augmentation flags
USE_AUGMENTATION = True
AUGMENTATION_NOISE_STD = 0.02
AUGMENTATION_SCALE_RANGE = (0.9, 1.1)


# ====
# NEW: FOCAL LOSS IMPLEMENTATION
# ====

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification.
    
    Addresses class imbalance by down-weighting easy examples and focusing
    on hard, misclassified examples.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
               Typical range: [0, 5]. Default: 2.0
        alpha: Balancing factor for class importance. Default: 0.25
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    
    Why this is best practice:
        - Standard in medical imaging with severe class imbalance
        - Improves recall on rare arrhythmias (LPFB, WPW, PACE, etc.)
        - Used in SOTA ECG classification papers (e.g., Ribeiro et al. 2020)
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross-entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate focal term: (1 - p_t)^gamma
        # This down-weights easy examples (high confidence correct predictions)
        focal_weight = K.pow(1.0 - y_pred, gamma)
        
        # Apply focal weight and alpha balancing
        focal_loss_value = alpha * focal_weight * cross_entropy
        
        # Sum across classes, mean across batch
        return K.sum(focal_loss_value, axis=-1)
    
    return loss_fn


# ====
# NEW: 1D SIGNAL AUGMENTATION
# ====

def augment_signal(signal):
    """
    Apply realistic augmentations to ECG signals.
    
    Augmentations simulate real-world conditions:
    1. Gaussian noise (electrode noise, EMG interference)
    2. Amplitude scaling (patient-to-patient variability, gain differences)
    3. Baseline wander (respiration, patient movement)
    
    Args:
        signal: Tensor of shape (batch_size, 5000, 12)
    
    Returns:
        Augmented signal of same shape
    
    Why this is best practice:
        - Improves model robustness to digitizer artifacts
        - Simulates real-world ECG acquisition variability
        - Common in SOTA ECG papers (e.g., Hannun et al. 2019, Ribeiro et al. 2020)
    
    References:
        - "Data Augmentation for ECG Classification" (Iwana & Uchida, 2021)
        - PTB-XL benchmark paper discusses augmentation strategies
    """
    # 1. Add Gaussian noise (simulates electrode noise)
    noise = tf.random.normal(
        shape=tf.shape(signal),
        mean=0.0,
        stddev=AUGMENTATION_NOISE_STD,
        dtype=tf.float32
    )
    signal = signal + noise
    
    # 2. Random amplitude scaling (simulates gain variability)
    # Scale per sample (not per lead) to maintain lead relationships
    scale = tf.random.uniform(
        shape=[tf.shape(signal)[0], 1, 1],
        minval=AUGMENTATION_SCALE_RANGE[0],
        maxval=AUGMENTATION_SCALE_RANGE[1],
        dtype=tf.float32
    )
    signal = signal * scale
    
    # 3. Baseline wander (simulates respiration, movement)
    # Low-frequency sinusoidal drift
    if tf.random.uniform([]) > 0.5:  # Apply 50% of the time
        # Random frequency between 0.1-0.5 Hz (typical baseline wander range)
        freq = tf.random.uniform([], 0.1, 0.5)
        time_steps = tf.cast(tf.shape(signal)[1], tf.float32)
        t = tf.linspace(0.0, time_steps / SAMPLING_RATE, tf.shape(signal)[1])
        
        # Create sinusoidal wander
        wander = 0.1 * tf.sin(2.0 * np.pi * freq * t)
        wander = tf.reshape(wander, [1, -1, 1])
        wander = tf.tile(wander, [tf.shape(signal)[0], 1, N_LEADS])
        
        signal = signal + wander
    
    return signal


def create_tf_dataset(X, y, batch_size, shuffle=True, augment=False):
    """
    Create optimized tf.data pipeline.
    
    Args:
        X: Signals array (N, 5000, 12)
        y: One-hot labels (N, num_classes)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
    
    Returns:
        tf.data.Dataset ready for model.fit()
    
    Why this is best practice:
        - tf.data is faster than feeding numpy arrays directly
        - Enables parallel data loading and preprocessing
        - Prefetching overlaps data loading with training
        - Standard in TensorFlow 2.x workflows
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        # Shuffle with buffer larger than dataset for true randomness
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    
    # Batch before augmentation (more efficient)
    dataset = dataset.batch(batch_size)
    
    if augment:
        # Apply augmentation to batches
        dataset = dataset.map(
            lambda x, y: (augment_signal(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch to overlap data loading with training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ====
# 1. DATA LOADING & PREPROCESSING
# ====

print("\n" + "=" * 70)
print("📊 STEP 1: Loading PTB-XL Dataset")
print("=" * 70)

# Load metadata CSV
metadata_file = DATA_DIR / "ptbxl_database.csv"
if not metadata_file.exists():
    print(f"❌ ERROR: Cannot find metadata file at {metadata_file}")
    print("Please ensure PTB-XL dataset is properly mounted in Kaggle.")
    sys.exit(1)

df = pd.read_csv(metadata_file)
print(f"✅ Loaded {len(df)} ECG records from metadata")

# Parse SCP codes (diagnostic labels)
df["scp_codes"] = df["scp_codes"].apply(lambda x: eval(x) if isinstance(x, str) else {})

# Map SCP codes to 14 diagnostic classes
CLASS_MAPPING = {
    "NORM": "Normal",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
    "AFIB": "Atrial Fibrillation",
    "LAFB": "Left Anterior Fascicular Block",
    "LPFB": "Left Posterior Fascicular Block",
    "RBBB": "Right Bundle Branch Block",
    "LBBB": "Left Bundle Branch Block",
    "WPW": "Wolff-Parkinson-White",
    "PVC": "Premature Ventricular Contraction",
    "PACE": "Pacemaker",
    "IVCD": "Intraventricular Conduction Delay",
}


def extract_primary_label(scp_dict):
    """Extract primary diagnostic label from SCP codes"""
    if not scp_dict:
        return "NORM"

    valid_codes = [code for code, conf in scp_dict.items() if conf >= 50]

    for code in CLASS_MAPPING.keys():
        if code in valid_codes:
            return code

    return "NORM"


df["primary_label"] = df["scp_codes"].apply(extract_primary_label)
df = df[df["primary_label"].isin(CLASS_MAPPING.keys())]
print(f"✅ Filtered to {len(df)} records with valid diagnostic labels")

# Label encoding
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["primary_label"])

print(f"\n📋 Class distribution:")
for label, count in df["primary_label"].value_counts().items():
    print(f"   {label:30s}: {count:5d} samples ({count/len(df)*100:.1f}%)")


# ====
# 2. SIGNAL LOADING & PREPROCESSING (500Hz)
# ====

print("\n" + "=" * 70)
print("🔊 STEP 2: Loading and Preprocessing ECG Signals (500Hz)")
print("=" * 70)


def load_ecg_signal(filename_hr, data_dir, target_length=SIGNAL_LENGTH):
    """Load ECG signal from WFDB format at 500Hz"""
    try:
        import wfdb

        record_name = str(filename_hr).replace(".hea", "").replace(".dat", "")
        record_path = data_dir / record_name
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal

        current_len = signal.shape[0]
        if current_len < target_length:
            pad_width = ((0, target_length - current_len), (0, 0))
            signal = np.pad(signal, pad_width, mode="constant", constant_values=0)
        elif current_len > target_length:
            signal = signal[:target_length, :]

        return signal

    except Exception as e:
        print(f"⚠️  Error loading {filename_hr}: {e}")
        return None


def preprocess_signal(signal, target_length=SIGNAL_LENGTH):
    """Preprocess ECG signal with Z-score normalization"""
    if signal is None or len(signal) == 0:
        return None

    # Z-score normalization per lead
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = ((0, target_length - current_length), (0, 0))
        signal = np.pad(signal, pad_width, mode="constant", constant_values=0)
    elif current_length > target_length:
        signal = signal[:target_length, :]

    return signal


print("Loading ECG signals at 500Hz... (this may take a few minutes)")
signals = []
labels = []
valid_indices = []

for idx, row in df.iterrows():
    signal = load_ecg_signal(row["filename_hr"], DATA_DIR, target_length=SIGNAL_LENGTH)
    if signal is not None:
        processed = preprocess_signal(signal)
        if processed is not None and processed.shape == (SIGNAL_LENGTH, N_LEADS):
            signals.append(processed.astype(np.float32))
            labels.append(row["label_encoded"])
            valid_indices.append(idx)

    if (idx + 1) % 1000 == 0:
        print(f"   Processed {idx + 1}/{len(df)} records...")

X = np.array(signals, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\n✅ Successfully loaded {len(X)} ECG signals at 500Hz")
print(f"   Signal shape: {X.shape}   # (n_samples, 5000, 12)")
print(f"   Labels shape: {y.shape}")


# ====
# 3. TRAIN/VAL/TEST SPLIT
# ====

print("\n" + "=" * 70)
print("✂️  STEP 3: Creating Train/Validation/Test Splits")
print("=" * 70)

df_valid = df.loc[valid_indices]
if "strat_fold" in df_valid.columns:
    print("Using PTB-XL official stratified folds (1-8 train, 9 val, 10 test)...")
    train_mask = df_valid["strat_fold"].isin(range(1, 9))
    val_mask = df_valid["strat_fold"] == 9
    test_mask = df_valid["strat_fold"] == 10

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
else:
    print("Creating custom train/val/test splits (70/15/15)...")
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

# NEW: Create tf.data pipelines
print("\n🔄 Creating optimized tf.data pipelines...")
train_dataset = create_tf_dataset(
    X_train, y_train_cat, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    augment=USE_AUGMENTATION
)
val_dataset = create_tf_dataset(
    X_val, y_val_cat, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    augment=False
)
test_dataset = create_tf_dataset(
    X_test, y_test_cat, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    augment=False
)
print(f"✅ Pipelines created (augmentation={'enabled' if USE_AUGMENTATION else 'disabled'})")


# ====
# 4. MODEL ARCHITECTURE: ResNet1D
# ====

print("\n" + "=" * 70)
print("🏗️  STEP 4: Building ResNet1D Architecture (500Hz)")
print("=" * 70)


def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.2):
    """Residual block with skip connection"""
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


def build_resnet1d(input_shape=(SIGNAL_LENGTH, N_LEADS), num_classes=TARGET_CLASSES):
    """Build ResNet1D model for ECG classification"""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=512, stride=2)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


model = build_resnet1d(
    input_shape=(SIGNAL_LENGTH, N_LEADS), num_classes=num_classes
)

print("\n📐 Model Architecture:")
model.summary()

# Calculate class weights (still useful even with Focal Loss)
class_weights = {}
unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
for cls, count in zip(unique, counts):
    class_weights[cls] = total / (len(unique) * count)

print(f"\n⚖️  Class weights (for handling imbalance):")
for cls, weight in class_weights.items():
    print(f"   Class {cls}: {weight:.2f}")


# ====
# 5. MODEL COMPILATION & TRAINING (WITH FOCAL LOSS)
# ====

print("\n" + "=" * 70)
print("🎯 STEP 5: Compiling and Training Model (500Hz + Focal Loss)")
print("=" * 70)

# NEW: Use Focal Loss instead of categorical_crossentropy
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2.0, alpha=0.25),  # NEW: Focal Loss
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    ModelCheckpoint(
        str(OUTPUT_DIR / "model.weights.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    ),
]

print("\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Initial learning rate: {LEARNING_RATE}")
print(f"   Loss function: Focal Loss (gamma=2.0, alpha=0.25)")
print(f"   Augmentation: {'Enabled' if USE_AUGMENTATION else 'Disabled'}")

# NEW: Train with tf.data pipelines
history = model.fit(
    train_dataset,  # NEW: Use dataset instead of arrays
    validation_data=val_dataset,  # NEW: Use dataset
    epochs=EPOCHS,
    class_weight=class_weights,  # Still useful with Focal Loss
    callbacks=callbacks,
    verbose=2,
)


# ====
# 6. MODEL EVALUATION
# ====

print("\n" + "=" * 70)
print("📊 STEP 6: Evaluating Model Performance")
print("=" * 70)

# NEW: Evaluate with test dataset
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    test_dataset, verbose=0
)

print(f"\n🎯 Test Set Results (500Hz + Focal Loss + Augmentation):")
print(f"   Loss:      {test_loss:.4f}")
print(f"   Accuracy:  {test_acc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(
    f"   F1-Score:  {2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8):.4f}"
)

# Per-class accuracy
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

print(f"\n📋 Per-Class Accuracy:")
for cls in range(num_classes):
    mask = y_test_classes == cls
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == cls).mean()
        class_name = label_encoder.inverse_transform([cls])[0]
        print(f"   {class_name:30s}: {acc:.4f} ({mask.sum()} samples)")


# ====
# 7. SAVE ARTIFACTS
# ====

print("\n" + "=" * 70)
print("💾 STEP 7: Saving Model Artifacts")
print("=" * 70)

classes_file = OUTPUT_DIR / "classes.npy"
np.save(classes_file, label_encoder.classes_)
print(f"✅ Saved label classes to: {classes_file}")

weights_file = OUTPUT_DIR / "model.weights.h5"
print(f"✅ Model weights saved to: {weights_file}")

zip_file = OUTPUT_DIR / "output.zip"
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(weights_file, "model.weights.h5")
    zipf.write(classes_file, "classes.npy")

print(f"✅ Created output package: {zip_file}")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE! (Phase 1.5: 500Hz + Focal Loss + Augmentation)")
print("=" * 70)

if IS_KAGGLE:
    print("\n📥 To download your trained model:")
    print("   1. Look for 'output.zip' in the Output section (right panel)")
    print("   2. Click the download icon")
    print("   3. Extract the zip file to use model.weights.h5 and classes.npy")
else:
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")

print("This overpredicts the normal ECG over other classes due to undersampling:" \
"Test Set Results (Phase 1.5b: Tuned Focal Loss + Light Augmentation): Loss: 0.1131 Accuracy: 0.8988 Precision: 0.9155 Recall: 0.8752 F1-Score: 0.8949 Per Class Accuracy: AFIB : 0.0000 (8 samples) IVCD : 0.0000 (57 samples) LAFB : 0.7421 (159 samples) LPFB : 0.5000 (16 samples) NORM : 0.9650 (1856 samples) PACE : 0.7037 (27 samples) PVC : 0.5694 (72 samples) WPW : 0.3750 (8 samples)")

print("\n📖 References:")
print("   - Focal Loss: Lin et al. (2017) https://arxiv.org/abs/1708.02002")
print("   - ECG Augmentation: Iwana & Uchida (2021)")
print("   - PTB-XL Benchmark: Wagner et al. (2020) Nature Scientific Data\n")