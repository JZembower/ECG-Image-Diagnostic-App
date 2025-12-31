"""
Services package for ECG Diagnosis Backend.

Contains:
- digitizer: Image-to-signal conversion using OpenCV
- classifier: Arrhythmia classification using TensorFlow
"""

from .digitizer import extract_signal_from_image, assess_signal_quality
from .classifier import classify_ecg, load_model, calculate_bpm

__all__ = [
    'extract_signal_from_image',
    'assess_signal_quality',
    'classify_ecg',
    'load_model',
    'calculate_bpm',
]

__version__ = '1.0.0'
