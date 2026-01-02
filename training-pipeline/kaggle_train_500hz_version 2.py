"""
PTB-XL ECG Arrhythmia Classification - Phase 2 Memory-Optimized for Kaggle
====

Memory Optimizations:
- On-the-fly TF-native image generation (no matplotlib, no pre-stored arrays)
- 112x112 images with MobileNetV2 (lighter than EfficientNet)
- Mixed precision (float16) training
- Smaller batch size (16)
- Oversampling + focal loss from Phase 1.5c

Target: 90%+ accuracy within Kaggle's ~13GB RAM limit

Authors: Elissa Matlock, Eugene Ho, Jonah Zembower
"""

import os
import sys
import zipfile
from pathlib import Path
from collections import Counter

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
# ENABLE MIXED PRECISION (saves ~50% memory)
# ====
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision (float16) enabled")

# ====
# CONFIGURATION
# ====

IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path("./data/ptb-xl")
    OUTPUT_DIR = Path("./output")
    OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🔍 Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# Model configuration
TARGET_CLASSES = 14
SAMPLING_RATE = 500
SIGNAL_LENGTH = 5000
N_LEADS = 12

# MEMORY OPTIMIZED settings
BATCH_SIZE = 16  # Reduced from 32
IMAGE_SIZE = 112  # Reduced from 224
EPOCHS = 50
LEARNING_RATE = 0.0005

# Focal loss (tuned)
FOCAL_GAMMA = 1.0
FOCAL_ALPHA = 0.5

# Augmentation (light)
AUG_NOISE_STD = 0.01
AUG_SCALE_RANGE = (0.95, 1.05)

# Oversampling
MIN_SAMPLES_PER_CLASS = 100

# Two-stage training
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 45


# ====
# FOCAL LOSS
# ====

def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Focal Loss for class imbalance."""
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        focal_weight = K.pow(1.0 - y_pred, gamma)
        focal_loss_value = alpha * focal_weight * cross_entropy
        return K.sum(focal_loss_value, axis=-1)
    return loss_fn


# ====
# TF-NATIVE IMAGE GENERATION (no matplotlib, memory efficient)
# ====

def signal_to_image_tf(signal):
    """
    TensorFlow-native ECG image generation.
    Generates a 112x112x3 image from signal without matplotlib.
    """
    # Use Lead II (index 1) - most diagnostic value
    lead = signal[:, 1]
    
    # Normalize to [0, 1]
    lead_min = tf.reduce_min(lead)
    lead_max = tf.reduce_max(lead)
    lead_norm = (lead - lead_min) / (lead_max - lead_min + 1e-8)
    
    # Downsample 5000 -> IMAGE_SIZE points
    indices = tf.cast(tf.linspace(0.0, float(SIGNAL_LENGTH - 1), IMAGE_SIZE), tf.int32)
    lead_down = tf.gather(lead_norm, indices)
    
    # Create 2D image representation
    # Each column shows signal value as a Gaussian trace
    y_coords = tf.range(IMAGE_SIZE, dtype=tf.float32) / float(IMAGE_SIZE)
    lead_expanded = tf.reshape(lead_down, [1, IMAGE_SIZE])  # (1, 112)
    y_expanded = tf.reshape(y_coords, [IMAGE_SIZE, 1])  # (112, 1)
    
    # Create grayscale trace: bright where signal is, dark elsewhere
    # Invert y because image coordinates are top-down
    distance = tf.abs(y_expanded - (1.0 - lead_expanded))
    img = tf.exp(-distance * 15.0)  # Gaussian-like trace
    
    # Invert: white background, black trace (like real ECG paper)
    img = 1.0 - img
    
    # Expand to 3 channels (RGB)
    img = tf.expand_dims(img, -1)  # (112, 112, 1)
    img = tf.repeat(img, 3, axis=-1)  # (112, 112, 3)
    
    return img


# ====
# DUAL-STREAM TF.DATA PIPELINE (memory efficient)
# ====

def create_dual_stream_dataset(X_signals, y_labels, batch_size, shuffle=True, augment=False):
    """
    Memory-efficient dual-stream dataset.
    Generates images on-the-fly instead of storing in RAM.
    """
    
    def process_sample(signal, label):
        """Process single sample: generate image from signal."""
        image = signal_to_image_tf(signal)
        return {'signal_input': signal, 'image_input': image}, label
    
    def augment_signal_tf(inputs, label):
        """Light augmentation for signals."""
        signal = inputs['signal_input']
        image = inputs['image_input']
        
        # Add Gaussian noise to signal
        noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=AUG_NOISE_STD)
        signal = signal + noise
        
        # Amplitude scaling
        scale = tf.random.uniform([], AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
        signal = signal * scale
        
        # Regenerate image from augmented signal (keeps them in sync)
        image = signal_to_image_tf(signal)
        
        return {'signal_input': signal, 'image_input': image}, label
    
    dataset = tf.data.Dataset.from_tensor_slices((X_signals, y_labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X_signals), 10000), reshuffle_each_iteration=True)
    
    # Generate images on-the-fly
    dataset = dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Augment after batching (more efficient)
    if augment:
        dataset = dataset.map(
            lambda inputs, labels: (
                {
                    'signal_input': inputs['signal_input'] + tf.random.normal(tf.shape(inputs['signal_input']), 0.0, AUG_NOISE_STD),
                    'image_input': inputs['image_input']
                },
                labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ====
# DATA LOADING
# ====

print("\n" + "=" * 70)
print("📊 STEP 1: Loading PTB-XL Dataset")
print("=" * 70)

metadata_file = DATA_DIR / "ptbxl_database.csv"
if not metadata_file.exists():
    print(f"❌ ERROR: Cannot find metadata file at {metadata_file}")
    sys.exit(1)

df = pd.read_csv(metadata_file)
print(f"✅ Loaded {len(df)} ECG records from metadata")

df["scp_codes"] = df["scp_codes"].apply(lambda x: eval(x) if isinstance(x, str) else {})

CLASS_MAPPING = {
    "NORM": "Normal", "MI": "Myocardial Infarction", "STTC": "ST/T Change",
    "CD": "Conduction Disturbance", "HYP": "Hypertrophy", "AFIB": "Atrial Fibrillation",
    "LAFB": "Left Anterior Fascicular Block", "LPFB": "Left Posterior Fascicular Block",
    "RBBB": "Right Bundle Branch Block", "LBBB": "Left Bundle Branch Block",
    "WPW": "Wolff-Parkinson-White", "PVC": "Premature Ventricular Contraction",
    "PACE": "Pacemaker", "IVCD": "Intraventricular Conduction Delay",
}


def extract_primary_label(scp_dict):
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

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["primary_label"])

print(f"\n📋 Class distribution:")
for label, count in df["primary_label"].value_counts().items():
    print(f"   {label:30s}: {count:5d} samples ({count/len(df)*100:.1f}%)")


# ====
# SIGNAL LOADING (500Hz)
# ====

print("\n" + "=" * 70)
print("🔊 STEP 2: Loading ECG Signals (500Hz)")
print("=" * 70)


def load_ecg_signal(filename_hr, data_dir, target_length=SIGNAL_LENGTH):
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
        return None


def preprocess_signal(signal, target_length=SIGNAL_LENGTH):
    if signal is None or len(signal) == 0:
        return None
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = ((0, target_length - current_length), (0, 0))
        signal = np.pad(signal, pad_width, mode="constant", constant_values=0)
    elif current_length > target_length:
        signal = signal[:target_length, :]
    return signal


print("Loading ECG signals...")
signals = []
labels = []
valid_indices = []

for idx, row in df.iterrows():
    signal = load_ecg_signal(row["filename_hr"], DATA_DIR)
    if signal is not None:
        processed = preprocess_signal(signal)
        if processed is not None and processed.shape == (SIGNAL_LENGTH, N_LEADS):
            signals.append(processed.astype(np.float32))
            labels.append(row["label_encoded"])
            valid_indices.append(idx)

    if (idx + 1) % 2000 == 0:
        print(f"   Processed {idx + 1}/{len(df)} records...")

X = np.array(signals, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\n✅ Loaded {len(X)} ECG signals")
print(f"   Signal shape: {X.shape}")

# Free memory
del signals, labels
import gc
gc.collect()


# ====
# OVERSAMPLING MINORITY CLASSES
# ====

print("\n" + "=" * 70)
print("⚖️  STEP 3: Oversampling Minority Classes")
print("=" * 70)


def oversample_minority_classes(X, y, min_samples=MIN_SAMPLES_PER_CLASS):
    counter = Counter(y)
    X_list, y_list = list(X), list(y)
    
    for cls, count in counter.items():
        if count < min_samples:
            indices = np.where(y == cls)[0]
            n_dup = min_samples - count
            for idx in np.random.choice(indices, n_dup, replace=True):
                # Add with slight noise for variety
                aug_signal = X[idx] + np.random.normal(0, 0.01, X[idx].shape).astype(np.float32)
                X_list.append(aug_signal)
                y_list.append(cls)
            print(f"   Class {cls} ({label_encoder.inverse_transform([cls])[0]}): {count} -> {min_samples}")
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


X, y = oversample_minority_classes(X, y)
print(f"\n✅ After oversampling: {len(X)} samples")


# ====
# TRAIN/VAL/TEST SPLIT
# ====

print("\n" + "=" * 70)
print("✂️  STEP 4: Train/Val/Test Split")
print("=" * 70)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Free memory
del X, y, X_temp, y_temp
gc.collect()

num_classes = len(np.unique(y_train))
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Create tf.data pipelines (NO IMAGE ARRAYS IN MEMORY!)
print("\n🔄 Creating memory-efficient tf.data pipelines...")
train_dataset = create_dual_stream_dataset(X_train, y_train_cat, BATCH_SIZE, shuffle=True, augment=True)
val_dataset = create_dual_stream_dataset(X_val, y_val_cat, BATCH_SIZE, shuffle=False, augment=False)
test_dataset = create_dual_stream_dataset(X_test, y_test_cat, BATCH_SIZE, shuffle=False, augment=False)
print("✅ Pipelines created (images generated on-the-fly)")


# ====
# DUAL-STREAM MODEL (MobileNetV2 - lighter than EfficientNet)
# ====

print("\n" + "=" * 70)
print("🏗️  STEP 5: Building Dual-Stream Model")
print("=" * 70)


def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.2):
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


def build_dual_stream_model(signal_shape=(SIGNAL_LENGTH, N_LEADS), 
                            image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            num_classes=TARGET_CLASSES):
    """
    Dual-Stream Model:
    - Signal Branch: ResNet1D
    - Image Branch: MobileNetV2 (lighter than EfficientNet)
    - Fusion: Concatenate + Dense
    """
    
    # ==== SIGNAL BRANCH ====
    signal_input = layers.Input(shape=signal_shape, name='signal_input')
    
    x_sig = layers.Conv1D(64, kernel_size=7, strides=2, padding="same")(signal_input)
    x_sig = layers.BatchNormalization()(x_sig)
    x_sig = layers.Activation("relu")(x_sig)
    x_sig = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x_sig)
    
    x_sig = residual_block(x_sig, filters=64, stride=1)
    x_sig = residual_block(x_sig, filters=128, stride=2)
    x_sig = residual_block(x_sig, filters=256, stride=2)
    x_sig = residual_block(x_sig, filters=512, stride=2)
    
    x_sig = layers.GlobalAveragePooling1D()(x_sig)
    signal_features = layers.Dense(256, activation='relu', name='signal_features')(x_sig)
    signal_features = layers.Dropout(0.3)(signal_features)
    
    # ==== IMAGE BRANCH (MobileNetV2 - lighter) ====
    image_input = layers.Input(shape=image_shape, name='image_input')
    
    # Use MobileNetV2 instead of EfficientNet (much lighter)
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    mobilenet.trainable = False  # Freeze initially
    
    x_img = mobilenet(image_input)
    image_features = layers.Dense(256, activation='relu', name='image_features')(x_img)
    image_features = layers.Dropout(0.3)(image_features)
    
    # ==== FUSION ====
    fused = layers.Concatenate(name='fusion')([signal_features, image_features])
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    
    # Output (float32 for numerical stability with mixed precision)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(fused)
    
    model = models.Model(
        inputs=[signal_input, image_input],
        outputs=outputs,
        name='DualStream_ECG_Classifier'
    )
    
    return model, mobilenet


model, mobilenet_base = build_dual_stream_model(num_classes=num_classes)
print("\n📐 Model Summary:")
model.summary()


# ====
# CLASS WEIGHTS (sqrt-dampened)
# ====

class_weights = {}
unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
for cls, count in zip(unique, counts):
    class_weights[cls] = np.sqrt(total / (len(unique) * count))

print(f"\n⚖️  Class weights (sqrt-dampened):")
for cls, weight in sorted(class_weights.items()):
    print(f"   Class {cls}: {weight:.2f}")


# ====
# TWO-STAGE TRAINING
# ====

print("\n" + "=" * 70)
print("🎯 STEP 6: Two-Stage Training")
print("=" * 70)

# ==== STAGE 1: Frozen MobileNet ====
print("\n🔥 STAGE 1: Warmup (MobileNet frozen)")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')],
)

stage1_callbacks = [
    ModelCheckpoint(str(OUTPUT_DIR / 'model_stage1.weights.h5'), monitor='val_accuracy', 
                    save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights,
    callbacks=stage1_callbacks,
    verbose=2,
)

# ==== STAGE 2: Unfreeze MobileNet ====
print("\n🔥 STAGE 2: Fine-tuning (MobileNet unfrozen)")

mobilenet_base.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')],
)

stage2_callbacks = [
    ModelCheckpoint(str(OUTPUT_DIR / 'model.weights.h5'), monitor='val_accuracy', 
                    save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=stage2_callbacks,
    verbose=2,
)


# ====
# EVALUATION
# ====

print("\n" + "=" * 70)
print("📊 STEP 7: Evaluation")
print("=" * 70)

model.load_weights(str(OUTPUT_DIR / 'model.weights.h5'))

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset, verbose=0)

print(f"\n🎯 Test Results:")
print(f"   Loss:      {test_loss:.4f}")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8):.4f}")

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
# SAVE ARTIFACTS
# ====

print("\n" + "=" * 70)
print("💾 STEP 8: Saving Artifacts")
print("=" * 70)

classes_file = OUTPUT_DIR / "classes.npy"
np.save(classes_file, label_encoder.classes_)
print(f"✅ Saved: {classes_file}")

weights_file = OUTPUT_DIR / "model.weights.h5"
print(f"✅ Saved: {weights_file}")

zip_file = OUTPUT_DIR / "output.zip"
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(weights_file, "model.weights.h5")
    zipf.write(classes_file, "classes.npy")
print(f"✅ Created: {zip_file}")

# ====
# VISUALIZATION: Confusion Matrix & Training Curves
# ====

print("\n" + "=" * 70)
print("📈 STEP 9: Generating Visualizations")
print("=" * 70)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get class names
class_names = label_encoder.classes_

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_classes, y_pred_classes)
cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].tick_params(axis='x', rotation=45)

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: confusion_matrix.png")

# --- Training History ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Combine histories
history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

epochs_range = range(1, len(history['loss']) + 1)

# Loss
axes[0, 0].plot(epochs_range, history['loss'], 'b-', label='Train')
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val')
axes[0, 0].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Stage 2 Start')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs_range, history['accuracy'], 'b-', label='Train')
axes[0, 1].plot(epochs_range, history['val_accuracy'], 'r-', label='Val')
axes[0, 1].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Stage 2 Start')
axes[0, 1].set_title('Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(epochs_range, history['precision'], 'b-', label='Train')
axes[1, 0].plot(epochs_range, history['val_precision'], 'r-', label='Val')
axes[1, 0].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Stage 2 Start')
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(epochs_range, history['recall'], 'b-', label='Train')
axes[1, 1].plot(epochs_range, history['val_recall'], 'r-', label='Val')
axes[1, 1].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Stage 2 Start')
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training History (Two-Stage)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: training_history.png")

# --- Per-Class Performance Bar Chart ---
fig, ax = plt.subplots(figsize=(14, 6))

class_accuracies = []
class_samples = []
for cls in range(num_classes):
    mask = y_test_classes == cls
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == cls).mean()
        class_accuracies.append(acc)
        class_samples.append(mask.sum())
    else:
        class_accuracies.append(0)
        class_samples.append(0)

x = np.arange(len(class_names))
bars = ax.bar(x, class_accuracies, color='steelblue', edgecolor='black')

# Color bars by performance
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    if acc == 0:
        bar.set_color('red')
    elif acc < 0.5:
        bar.set_color('orange')
    elif acc < 0.8:
        bar.set_color('gold')
    else:
        bar.set_color('green')

# Add sample counts on bars
for i, (acc, n) in enumerate(zip(class_accuracies, class_samples)):
    ax.text(i, acc + 0.02, f'n={n}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy (Red=0%, Orange<50%, Gold<80%, Green≥80%)')
ax.set_ylim(0, 1.15)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: per_class_accuracy.png")

# --- Classification Report ---
print("\n📋 Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names, zero_division=0))

print("\n" + "=" * 70)
print("🎉 PHASE 2 TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Test Accuracy: {test_acc*100:.2f}%")
print("\n✅ Memory optimizations applied:")
print("   - On-the-fly TF-native image generation")
print("   - 112x112 images (vs 224x224)")
print("   - MobileNetV2 (vs EfficientNet)")
print("   - Mixed precision (float16)")
print("   - Batch size 16 (vs 32)")