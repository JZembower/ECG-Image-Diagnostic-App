# ECG Arrhythmia Detection System

> **An AI-powered mobile ECG diagnostic system for real-time arrhythmia detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![React Native](https://img.shields.io/badge/React_Native-0.72+-61DAFB.svg)](https://reactnative.dev/)

![ECG App UI](mobile-app/assets/ECG%20App%20UI.png)

---

## 🎯 Overview

This project implements an end-to-end ECG arrhythmia detection system that:

1. **Captures** ECG images via smartphone camera
2. **Digitizes** paper ECG traces into digital signals
3. **Classifies** 14 types of cardiac arrhythmias using deep learning
4. **Provides** instant diagnostic insights on mobile devices

### Key Features

- **14-Class Arrhythmia Detection:**
  - Normal Sinus Rhythm
  - Atrial Fibrillation (AFib)
  - Myocardial Infarction (MI)
  - Right/Left Bundle Branch Block (RBBB/LBBB)
  - ST Elevation/Depression
  - Hypertrophy
  - Conduction Disturbances
  - And more...

- **Mobile-First Design:**
  - React Native app for iOS/Android
  - Real-time ECG capture and calibration
  - Manual R-peak marking and BPM calculation
  - User-friendly interface with visual feedback

- **Production-Ready ML Pipeline:**
  - ResNet1D architecture optimized for 1D ECG signals
  - Trained on 22,000+ clinical ECG records (PTB-XL dataset)
  - Handles class imbalance with weighted loss
  - Achieves ~70% accuracy on 14-class classification

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MOBILE APP (React Native)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Camera       │→ │ Calibration  │→ │ R-Peak      │            │
│  │ Capture      │  │ & BPM Calc   │  │ Marking     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ECG Image
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Image        │→ │ Signal       │→ │ Arrhythmia  │            │
│  │ Digitizer    │  │ Preprocessor │  │ Classifier  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Predictions
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ML MODEL (ResNet1D on PTB-XL)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Input:       │→ │ ResNet1D     │→ │ Output:     │            │
│  │ 12-lead ECG  │  │ Architecture │  │ 14 Classes  │            │
│  │ (1000×12)    │  │              │  │ + Confidence│            │ 
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ecg-diagnosis-system/
├── mobile-app/                    # React Native mobile application
│   ├── src/
│   │   ├── screens/
│   │   │   ├── index.tsx         # Main ECG analyzer screen
│   │   │   ├── App.js            # Alternative implementation
│   │   │   └── _layout.tsx       # Tab navigation layout
│   │   └── components/           # Reusable UI components
│   └── README_ORIGINAL.md        # Original project documentation
│
├── backend-api/                   # FastAPI backend (Phase 3)
│   ├── main.py                   # API endpoints (to be built)
│   ├── services/                 # Business logic
│   │   ├── digitizer.py          # ECG image → signal conversion
│   │   └── classifier.py         # ML model inference
│   └── models/                   # Model artifacts directory
│       ├── model_weights.h5      # Trained ResNet1D weights
│       └── classes.npy           # Label encoder classes
│
├── training-pipeline/             # Kaggle training scripts (Phase 1-2)
│   ├── kaggle_train.py           # ⭐ Main training script
│   └── requirements.txt          # Python dependencies
│
└── README.md                      # This file
```

---

## 🚀 Phase 1 & 2: Model Training on Kaggle

This section guides you through training the ResNet1D model on Kaggle using the PTB-XL dataset.

### Prerequisites

- **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com/) (free)
- **GPU Access**: Kaggle provides free GPU resources (T4/P100)
- **PTB-XL Dataset**: Available on PhysioNet and Kaggle

---

### 📊 Dataset Setup

#### Option 1: Use Existing Kaggle Dataset (Recommended)

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Search for **"PTB-XL"** or **"PTB-XL EKG Dataset"**
3. Find a dataset like `khyeh0719/ptb-xl-dataset` or similar
4. Click "New Notebook" to create a notebook with the dataset attached

#### Option 2: Upload PTB-XL Manually

1. Download PTB-XL from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Extract the archive (you'll get ~22GB of data)
3. Upload to Kaggle Datasets:
   - Go to [Your Datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload the PTB-XL folder
   - Name it `ptbxl-ekg` for compatibility

---

### 🎯 Running the Training Script

#### Step 1: Create a Kaggle Notebook

1. Navigate to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Select **"Notebook"** (not Script)
4. Enable GPU:
   - Click "Accelerator" in right panel
   - Select "GPU T4 x2" or "GPU P100"

#### Step 2: Add PTB-XL Dataset

1. In the right panel, click **"+ Add Data"**
2. Search for the PTB-XL dataset you found/uploaded
3. Click "Add" to mount it to `/kaggle/input/ptbxl-ekg/`

#### Step 3: Copy & Paste the Training Script

1. Open `training-pipeline/kaggle_train.py` from this repository
2. **Copy the entire script** (all ~500 lines)
3. **Paste into a Kaggle notebook cell**
4. Run the cell!

**What Gets Saved:**

1. **`model.weights.h5`** (15-20 MB)
   - Trained ResNet1D model weights
   - Compatible with TensorFlow/Keras 2.10+
   - Input shape: `(batch, 1000, 12)`
   - Output shape: `(batch, 14)` - 14 class probabilities

2. **`classes.npy`** (<1 KB)
   - Label encoder classes
   - Maps model output indices to disease names
   - Example: `[0] → "Normal", [1] → "AFib", ...`

3. **`output.zip`** (15-20 MB)
   - Combines both files above
   - Ready for download and deployment

---

## 📚 Dataset & References

### PTB-XL Dataset

- **Source:** [PhysioNet - PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)
- **Size:** 21,837 clinical ECG records
- **Patients:** 18,885 unique patients
- **Sampling Rates:** 100Hz and 500Hz versions
- **Leads:** 12-lead ECG (I, II, III, aVR, aVL, aVF, V1-V6)
- **Labels:** SCP-ECG diagnostic codes mapped to superclasses
- **License:** ODC Open Database License v1.0

### Additional References

- **ECG Digitization:** [ECG-Digitiser GitHub](https://github.com/felixkrones/ECG-Digitiser)
- **Synthetic ECG Images:** [GenECG Dataset](https://huggingface.co/datasets/edcci/GenECG)
- **ResNet Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## 👥 Contributors

- **Elissa Matlock** - Machine Learning & Model Architecture
- **Eugene Ho** - Backend API & Integration
- **Jonah Zembower** - Mobile App Development

---

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.**

- NOT intended for clinical diagnosis or treatment decisions
- NOT a replacement for professional medical equipment or advice
- Always consult a licensed healthcare provider for medical concerns
- Accuracy may vary based on image quality and device capabilities
