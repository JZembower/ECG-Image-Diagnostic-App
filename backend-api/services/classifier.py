"""
ECG Classifier Service - Phase 2 Dual-Stream
Performs arrhythmia classification on ECG signals AND images using dual-stream TensorFlow model.

Architecture:
- Signal Branch: ResNet1D (5000, 12) @ 500Hz
- Image Branch: EfficientNetB0 (224, 224, 3)
- Fusion: Concatenated features -> Dense -> Softmax
"""

import os
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from scipy.signal import find_peaks
import cv2

# Configure logger
logger = logging.getLogger(__name__)

# Global model cache (singleton pattern)
_model = None
_classes = None

# Configuration (must match training script)
SIGNAL_LENGTH = 5000  # 10 seconds @ 500Hz
N_LEADS = 12
SAMPLING_RATE = 500  # Hz
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# ====
# 1. DUAL-STREAM MODEL ARCHITECTURE
# ====

def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.2):
    """Residual block for ResNet1D - must match training exactly"""
    try:
        from tensorflow.keras import layers
    except ImportError:
        import tensorflow.keras.layers as layers
    
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

def build_dual_stream_model(
    signal_shape=(SIGNAL_LENGTH, N_LEADS), 
    image_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
    num_classes=14
):
    """
    Build Phase 2 Dual-Stream Fusion Model.
    
    This MUST match the training architecture exactly for weights to load properly.
    
    Architecture:
    1. Signal Branch: ResNet1D (5000, 12) -> feature vector
    2. Image Branch: EfficientNetB0 (224, 224, 3) -> feature vector
    3. Fusion: Concatenate -> Dense layers -> Softmax
    """
    try:
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0
    except ImportError:
        import tensorflow.keras.layers as layers
        import tensorflow.keras.models as models
        from tensorflow.keras.applications import EfficientNetB0
    
    # ===== SIGNAL BRANCH: ResNet1D =====
    signal_input = layers.Input(shape=signal_shape, name='signal_input')
    
    # Initial convolution
    x_sig = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(signal_input)
    x_sig = layers.BatchNormalization()(x_sig)
    x_sig = layers.Activation('relu')(x_sig)
    x_sig = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x_sig)
    
    # Residual blocks
    x_sig = residual_block(x_sig, filters=64, stride=1)
    x_sig = residual_block(x_sig, filters=128, stride=2)
    x_sig = residual_block(x_sig, filters=256, stride=2)
    x_sig = residual_block(x_sig, filters=512, stride=2)
    
    # Global pooling
    x_sig = layers.GlobalAveragePooling1D()(x_sig)
    
    # Signal feature vector
    signal_features = layers.Dense(256, activation='relu', name='signal_features')(x_sig)
    signal_features = layers.Dropout(0.3)(signal_features)
    
    # ===== IMAGE BRANCH: EfficientNetB0 =====
    image_input = layers.Input(shape=image_shape, name='image_input')
    
    # Load pretrained EfficientNetB0 (ImageNet weights)
    efficientnet = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=image_input,
        pooling='avg'
    )
    
    # Image feature vector
    x_img = efficientnet.output
    image_features = layers.Dense(256, activation='relu', name='image_features')(x_img)
    image_features = layers.Dropout(0.3)(image_features)
    
    # ===== FUSION LAYER =====
    # Concatenate both feature vectors
    fused = layers.Concatenate(name='fusion')([signal_features, image_features])
    
    # Fusion dense layers
    fused = layers.Dense(512, activation='relu')(fused)
    fused = layers.Dropout(0.5)(fused)
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(fused)
    
    # Create model
    model = models.Model(
        inputs=[signal_input, image_input],
        outputs=outputs,
        name='DualStream_ECG_Classifier'
    )
    
    return model

# ====
# 2. MODEL LOADING & INFERENCE
# ====

def load_model():
    """
    Load TensorFlow dual-stream model and classes from disk (cached in memory).
    Uses singleton pattern to avoid reloading on each request.
    """
    global _model, _classes
    
    if _model is not None and _classes is not None:
        logger.debug("Using cached model")
        return _model, _classes
    
    try:
        import tensorflow as tf
        from pathlib import Path
        
        # Calculate base directory relative to THIS file
        BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Define paths dynamically
        default_model_path = BASE_DIR / "models" / "model.weights.h5"
        default_classes_path = BASE_DIR / "models" / "classes.npy"

        # Get model paths from environment or use defaults
        model_path = os.getenv("MODEL_PATH", str(default_model_path))
        classes_path = os.getenv("CLASSES_PATH", str(default_classes_path))
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please ensure 'model.weights.h5' is in the 'models' folder."
            )
        
        if not os.path.exists(classes_path):
            raise FileNotFoundError(
                f"Classes file not found at: {classes_path}\n"
                f"Please ensure 'classes.npy' is in the 'models' folder."
            )
        
        # Load classes FIRST (need count to build model)
        logger.info(f"Loading classes from {classes_path}...")
        _classes = np.load(classes_path, allow_pickle=True)
        logger.info(f"Classes loaded: {_classes}")
        num_classes = len(_classes)

        # Build architecture
        logger.info("Building Phase 2 Dual-Stream architecture...")
        _model = build_dual_stream_model(
            signal_shape=(SIGNAL_LENGTH, N_LEADS),
            image_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
            num_classes=num_classes
        )
        
        # Load weights
        logger.info(f"Loading weights from {model_path}...")
        _model.load_weights(model_path)
        logger.info("✅ Dual-stream model weights loaded successfully.")
        
        return _model, _classes
    
    except ImportError as e:
        logger.error(f"TensorFlow not installed: {e}")
        raise RuntimeError(
            "TensorFlow is not installed. Please install it with: pip install tensorflow>=2.14.0"
        )
    
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Preprocess signal for dual-stream model inference.
    
    Steps:
        1. Ensure shape is (5000, 12) - 500Hz sampling
        2. Apply Z-score normalization
        3. Expand dimensions for batch: (1, 5000, 12)
    """
    EXPECTED_SAMPLES = SIGNAL_LENGTH  # 5000
    
    # Ensure shape is correct
    if signal.shape != (EXPECTED_SAMPLES, N_LEADS):
        logger.warning(f"Signal shape {signal.shape} does not match expected ({EXPECTED_SAMPLES}, {N_LEADS}). Attempting to reshape...")
        
        # Try to handle different shapes
        if signal.ndim == 1:
            # Single lead, needs to be expanded to 12 leads
            signal = np.tile(signal[:EXPECTED_SAMPLES], (N_LEADS, 1)).T
        elif signal.shape[0] < EXPECTED_SAMPLES:
            # Pad with zeros if too short
            pad_length = EXPECTED_SAMPLES - signal.shape[0]
            signal = np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
        elif signal.shape[0] > EXPECTED_SAMPLES:
            # Truncate if too long
            signal = signal[:EXPECTED_SAMPLES, :]
        
        if signal.shape[1] != N_LEADS:
            # Replicate or truncate leads
            if signal.shape[1] < N_LEADS:
                signal = np.tile(signal, (1, N_LEADS))[:, :N_LEADS]
            else:
                signal = signal[:, :N_LEADS]
    
    # Z-score normalization
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    
    if np.any(np.abs(mean) > 1e-6) or np.any(np.abs(std - 1.0) > 0.5):
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        signal = (signal - mean) / std
    
    # Expand dimensions for batch
    signal_batch = np.expand_dims(signal, axis=0)  # Shape: (1, 5000, 12)
    
    return signal_batch

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess ECG image for EfficientNetB0.
    
    Steps:
        1. Resize to (224, 224)
        2. Convert to RGB if grayscale
        3. Normalize to [0, 1]
        4. Expand dimensions for batch: (1, 224, 224, 3)
    """
    # Resize to target size
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image / 255.0
    
    # Ensure float32
    image = image.astype(np.float32)
    
    # Expand dimensions for batch
    image_batch = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)
    
    return image_batch

def generate_placeholder_image(signal: np.ndarray) -> np.ndarray:
    """
    Generate placeholder ECG image from signal when no image is provided.
    
    This creates a simple visualization of the ECG waveform on a white background.
    """
    # Create white background
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.uint8) * 255
    
    # Plot first lead (Lead I) for simplicity
    lead_signal = signal[:, 0] if signal.ndim > 1 else signal
    
    # Normalize to image height
    signal_normalized = (lead_signal - lead_signal.min()) / (lead_signal.max() - lead_signal.min() + 1e-8)
    signal_scaled = (1 - signal_normalized) * (IMAGE_SIZE - 40) + 20  # Leave margins
    
    # Sample points to fit image width
    x_coords = np.linspace(0, IMAGE_SIZE - 1, len(lead_signal)).astype(int)
    y_coords = signal_scaled.astype(int)
    
    # Draw signal line
    for i in range(len(x_coords) - 1):
        cv2.line(img, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), (0, 0, 0), 2)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def classify_ecg(signal: np.ndarray, image: Optional[np.ndarray] = None) -> Dict:
    """
    Classify ECG using dual-stream model (signal + image).
    
    Args:
        signal: ECG signal array of shape (5000, 12) @ 500Hz
        image: Optional ECG image array of shape (H, W, 3) or (H, W)
               If None, a placeholder image will be generated from signal
    
    Returns:
        Dictionary with diagnosis, confidence, bpm, and all probabilities
    """
    try:
        # Load model (cached)
        model, classes = load_model()
        
        # Preprocess signal
        signal_preprocessed = preprocess_signal(signal)
        
        # Preprocess or generate image
        if image is None:
            logger.info("No image provided, generating placeholder from signal...")
            # Remove batch dimension if present for placeholder generation
            signal_for_image = signal_preprocessed[0] if signal_preprocessed.ndim == 3 else signal_preprocessed
            image_generated = generate_placeholder_image(signal_for_image)
            image_preprocessed = preprocess_image(image_generated)
        else:
            logger.info("Using provided ECG image...")
            image_preprocessed = preprocess_image(image)
        
        # Run inference
        logger.info("Running dual-stream model inference...")
        predictions = model.predict(
            [signal_preprocessed, image_preprocessed], 
            verbose=0
        )
        
        # Extract predictions
        probabilities = predictions[0]
        
        # Get top prediction
        top_class_idx = np.argmax(probabilities)
        top_confidence = float(probabilities[top_class_idx])
        
        # Decode class name
        if isinstance(classes, np.ndarray):
            diagnosis = str(classes[top_class_idx])
        else:
            diagnosis = f"Class_{top_class_idx}"
        
        logger.info(f"Prediction: {diagnosis} (confidence: {top_confidence:.3f})")
        
        # Get all probabilities
        all_probabilities = {}
        for idx, prob in enumerate(probabilities):
            try:
                class_name = str(classes[idx]) if idx < len(classes) else f"Class_{idx}"
            except:
                class_name = f"Class_{idx}"
            all_probabilities[class_name] = float(prob)
        
        # Calculate BPM
        bpm = calculate_bpm(signal)
        
        # Return results
        result = {
            "diagnosis": diagnosis,
            "confidence": top_confidence,
            "bpm": bpm,
            "all_probabilities": all_probabilities
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in classify_ecg: {e}", exc_info=True)
        raise

def calculate_bpm(signal: np.ndarray) -> int:
    """
    Calculate heart rate (BPM) from ECG signal using R-peak detection.
    
    Updated for 500Hz sampling rate.
    """
    try:
        # Use first lead if multi-lead
        if signal.ndim > 1:
            signal_1d = signal[:, 0]
        else:
            signal_1d = signal
        
        # Estimate peak height threshold
        threshold = np.mean(signal_1d) + 1.5 * np.std(signal_1d)
        
        # Minimum distance between peaks (assuming max HR of 200 BPM)
        # At 500Hz sampling, 200 BPM = 3.33 beats/sec
        # Minimum R-R interval = 60/200 = 0.3 sec = 150 samples @ 500Hz
        min_distance = 150
        
        # Find peaks
        peaks, _ = find_peaks(
            signal_1d,
            height=threshold,
            distance=min_distance
        )
        
        if len(peaks) < 2:
            # Not enough peaks, try lower threshold
            threshold = np.mean(signal_1d) + 0.5 * np.std(signal_1d)
            peaks, _ = find_peaks(
                signal_1d,
                height=threshold,
                distance=50
            )
        
        if len(peaks) < 2:
            logger.warning("Failed to detect R-peaks, using default BPM of 70")
            return 70
        
        # Calculate average R-R interval
        rr_intervals = np.diff(peaks)  # In samples
        avg_rr_interval = np.mean(rr_intervals)  # In samples
        
        # Convert to BPM (sampling rate is 500Hz)
        avg_rr_interval_sec = avg_rr_interval / SAMPLING_RATE
        bpm = 60.0 / avg_rr_interval_sec
        
        # Clamp values to realistic range
        if bpm < 40: bpm = 40
        if bpm > 200: bpm = 200
        
        return int(round(bpm))
    
    except Exception as e:
        logger.error(f"Error in calculate_bpm: {e}")
        return 70

# ====
# BACKWARD COMPATIBILITY: Signal-only inference
# ====

def classify_ecg_signal_only(signal: np.ndarray) -> Dict:
    """
    Convenience function for signal-only classification.
    Generates placeholder image internally.
    """
    return classify_ecg(signal=signal, image=None)
