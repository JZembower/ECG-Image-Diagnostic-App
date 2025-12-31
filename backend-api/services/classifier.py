"""
ECG Classifier Service
Performs arrhythmia classification on ECG signals using trained TensorFlow model.
"""

import os
import numpy as np
import logging
from typing import Dict, Optional
from scipy.signal import find_peaks

# Configure logger
logger = logging.getLogger(__name__)

# Global model cache (singleton pattern)
_model = None
_classes = None

# =============================================================================
# 1. MODEL ARCHITECTURE (The missing piece!)
# =============================================================================
def build_resnet1d(input_shape=(1000, 12), num_classes=14):
    """
    Builds the ResNet1D model architecture.
    This must match the training script exactly to load the weights.
    """
    try:
        from tensorflow.keras import layers, models
    except ImportError:
        import tensorflow.keras.layers as layers
        import tensorflow.keras.models as models

    inputs = layers.Input(shape=input_shape)
    
    # Entry Block
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Residual Blocks
    for filters in [64, 128, 256, 512]:
        # Shortcut path (skip connection)
        shortcut = layers.Conv1D(filters, 1, strides=2 if filters > 64 else 1, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        # Main path
        x = layers.Conv1D(filters, 3, strides=2 if filters > 64 else 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add & Activate
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# =============================================================================
# 2. MODEL LOADING & INFERENCE
# =============================================================================

def load_model():
    """
    Load TensorFlow model and classes from disk (cached in memory).
    Uses singleton pattern to avoid reloading on each request.
    """
    global _model, _classes
    
    if _model is not None and _classes is not None:
        logger.debug("Using cached model")
        return _model, _classes
    
    try:
        import tensorflow as tf
        from pathlib import Path
        
        # 1. Calculate the base directory relative to THIS file
        BASE_DIR = Path(__file__).resolve().parent.parent
        
        # 2. Define the paths dynamically
        default_model_path = BASE_DIR / "models" / "model.weights.h5"
        default_classes_path = BASE_DIR / "models" / "classes.npy"

        # Get model paths from environment or use our dynamic defaults
        model_path = os.getenv("MODEL_PATH", str(default_model_path))
        classes_path = os.getenv("CLASSES_PATH", str(default_classes_path))
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please ensure 'model.weights.h5' is in the 'backend-api/models' folder."
            )
        
        if not os.path.exists(classes_path):
            raise FileNotFoundError(
                f"Classes file not found at: {classes_path}\n"
                f"Please ensure 'classes.npy' is in the 'backend-api/models' folder."
            )
        
        # A. Load Classes FIRST (we need the count to build the model)
        logger.info(f"Loading classes from {classes_path}...")
        _classes = np.load(classes_path, allow_pickle=True)
        logger.info(f"Classes loaded: {_classes}")
        num_classes = len(_classes)

        # B. Build Architecture & Load Weights
        logger.info("Building ResNet1D architecture...")
        _model = build_resnet1d(input_shape=(1000, 12), num_classes=num_classes)
        
        logger.info(f"Loading weights from {model_path}...")
        _model.load_weights(model_path)
        logger.info("✅ Model weights loaded successfully.")
        
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
    Preprocess signal for model inference.
    Steps:
        1. Ensure shape is (1000, 12)
        2. Apply Z-score normalization
        3. Expand dimensions for batch: (1, 1000, 12)
    """
    # Ensure shape is correct
    if signal.shape != (1000, 12):
        logger.warning(f"Signal shape {signal.shape} does not match expected (1000, 12). Attempting to reshape...")
        
        # Try to handle different shapes
        if signal.ndim == 1:
            # Single lead, needs to be expanded to 12 leads
            signal = np.tile(signal[:1000], (12, 1)).T
        elif signal.shape[0] < 1000:
            # Pad with zeros if too short
            pad_length = 1000 - signal.shape[0]
            signal = np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
        elif signal.shape[0] > 1000:
            # Truncate if too long
            signal = signal[:1000, :]
        
        if signal.shape[1] != 12:
            # Replicate or truncate leads
            if signal.shape[1] < 12:
                signal = np.tile(signal, (1, 12))[:, :12]
            else:
                signal = signal[:, :12]
    
    # Ensure normalization (Z-score)
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    
    if np.any(np.abs(mean) > 1e-6) or np.any(np.abs(std - 1.0) > 0.5):
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        signal = (signal - mean) / std
    
    # Expand dimensions for batch
    signal_batch = np.expand_dims(signal, axis=0)  # Shape: (1, 1000, 12)
    
    return signal_batch

def classify_ecg(signal: np.ndarray) -> Dict:
    """
    Classify ECG signal using trained model.
    """
    try:
        # Load model (cached)
        model, classes = load_model()
        
        # Preprocess signal
        signal_preprocessed = preprocess_signal(signal)
        
        # Run inference
        logger.info("Running model inference...")
        predictions = model.predict(signal_preprocessed, verbose=0)
        
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
        # At 100Hz sampling, 200 BPM = 3.33 beats/sec = 30 samples between peaks
        min_distance = 30 
        
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
                distance=20
            )
        
        if len(peaks) < 2:
            logger.warning("Failed to detect R-peaks, using default BPM of 70")
            return 70
        
        # Calculate average R-R interval
        rr_intervals = np.diff(peaks)  # In samples
        avg_rr_interval = np.mean(rr_intervals)  # In samples
        
        # Convert to BPM (sampling rate is 100Hz)
        avg_rr_interval_sec = avg_rr_interval / 100.0
        bpm = 60.0 / avg_rr_interval_sec
        
        # Clamp values to realistic range
        if bpm < 40: bpm = 40
        if bpm > 200: bpm = 200
        
        return int(round(bpm))
    
    except Exception as e:
        logger.error(f"Error in calculate_bpm: {e}")
        return 70