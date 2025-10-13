# ECG Diagnostic Image Analysis

## Overview

We have created an innovative mobile application designed to empower users with real-time ECG (Electrocardiogram) diagnostic analysis using just their smartphone. By simply taking a photo of an ECG readout (e.g., from a monitor, printout, or wearable device), the app processes the image to extract heart rate and detect potential arrhythmias. This tool aims to provide quick, preliminary insights into heart health, making it accessible for personal use, remote monitoring, or basic use with oversight in medical settings.

Contributors:

- Elissa Matlock
- Eugene Ho
- Jonah Zembower

## Key features

- Heart Rate Detection: Calculates beats per minute (BPM) from the captured ECG waveform.
- Arrhythmia Classification
  - Atrial Fibrillation (AFib)
  - Right Bundle Branch Block (RBBB)
  - Left Bundle Branch Block (LBBB)
  - ST Elevation
  - ST Depression
  - AV Block
  - Myocardial Infarction (MI)
  - Wolff-Parkinson-White Syndrome (WPW)
  - Premature Ventricular Contraction (PVC)
  - Idioventricular Rhythm
  - Junctional Rhythm
  - Fusion Beats
  - Normal Sinus Rhythm
- User-Friendly Interface: Simple photo capture, instant analysis, and easy-to-understand results with visualizations.
- Privacy-Focused: All processing can be done on-device (with optional cloud support for advanced models).

## Dataset

The core machine learning model is trained on the PTB-XL dataset, a large, publicly available electrocardiography dataset sourced from PhysioNet. PTB-XL contains ~22,000 clinical 12-lead ECG records from 18,885 patients, annotated with diagnostic codes for various cardiac conditions. We processed this dataset to create a 14-class classification system, focusing on common arrhythmias and abnormalities.

Source: PhysioNet (PTB-XL v1.0.3)
Size: ~22,000 ECG records (sampled at 100Hz or 500Hz)
Classes: The various diagnostic categories
The dataset was preprocessed to handle waveform files (.dat/.hea) via WFDB, with signals padded/cropped to a fixed length (e.g., 5000 samples at 500Hz for 10-second clips).

## Model Training and Notebook

The backend ML model is a custom SE-ResNet-50 architecture adapted for 1D ECG signals, implemented in TensorFlow/Keras. 
Key Notebook: ptbxl-all.ipynb
This Jupyter notebook demonstrates the full pipeline for training the model on PTB-XL:

- Data Loading & Preprocessing: Parses SCP codes, maps to 14 classes using direct mappings and regex patterns, loads WFDB signals, and creates TensorFlow datasets with augmentation (e.g., random crop, flip, noise).
- Class Balancing: Computes class weights to handle imbalanced data (e.g., Normal rhythms dominate).
- Model Architecture: Builds a SE-ResNet-50 with bottleneck blocks, trained with cosine decay learning rate and Adam optimizer.
- Training: Fits the model over 25 epochs with early stopping, checkpointing, and validation monitoring.
- Evaluation: Achieves test accuracy of ~70% on the held-out test set.
- Outputs: Saves the trained model (best.keras or best.h5) and class labels (label_classes.npy).

## To run the notebook

- Install dependencies: pip install wfdb tensorflow numpy pandas sklearn
- Download PTB-XL from PhysioNet and run it with the necessary GPUs.
- Execute cells sequentially for data prep, model building, and training.

## How does it work?

- Image Capture: User takes a photo of an ECG strip via the phone camera.
- Signal Extraction: (Future/Planned) Use computer vision (e.g., OpenCV) to detect grid lines, extract waveform traces, and digitize into a time-series signal.
- Preprocessing: Normalize/resample the signal to match training data (e.g., 500Hz, 10s clip).
- Analysis:
  - Heart Rate: Detect R-peaks to conduct BPM.
  - Arrhythmia Detection: Feed the signal into the trained SE-ResNet model for classification.
  - Results: Display BPM, predicted class (with confidence), and waveform visualization. Alerts for high-risk detections (e.g., AFib).

## Dependencies

Python 3.10+
TensorFlow 2.15+
WFDB, NumPy, Pandas, Scikit-learn

## Disclaimer

This app is for educational and informational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for diagnosis and treatment.
