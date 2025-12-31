# ECG Arrhythmia Detection System

> **An AI-powered mobile ECG diagnostic system for real-time arrhythmia detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![React Native](https://img.shields.io/badge/React_Native-0.72+-61DAFB.svg)](https://reactnative.dev/)

![ECG App UI](C:\\Users\\jrzem\\OneDrive\\Majors\\Coding Applications\\Projects\\CMU HealthCare\\ECG-Image-Diagnostic-App\\mobile-app\\assets\\ECG App UI.png)

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Phase 1 & 2: Model Training on Kaggle](#phase-1--2-model-training-on-kaggle)
  - [Dataset Setup](#dataset-setup)
  - [Running the Training Script](#running-the-training-script)
  - [Expected Training Time & Output](#expected-training-time--output)
  - [Downloading the Model](#downloading-the-model)
- [Phase 3: Backend API (Coming Soon)](#phase-3-backend-api-coming-soon)
- [Phase 4: Mobile App Integration (Coming Soon)](#phase-4-mobile-app-integration-coming-soon)
- [Technical Specifications](#technical-specifications)
- [Dataset & References](#dataset--references)
- [Contributors](#contributors)

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
│                      MOBILE APP (React Native)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Camera       │→ │ Calibration  │→ │ R-Peak      │           │
│  │ Capture      │  │ & BPM Calc   │  │ Marking     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ECG Image
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Image        │→ │ Signal       │→ │ Arrhythmia  │           │
│  │ Digitizer    │  │ Preprocessor │  │ Classifier  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Predictions
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ML MODEL (ResNet1D on PTB-XL)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Input:       │→ │ ResNet1D     │→ │ Output:     │           │
│  │ 12-lead ECG  │  │ Architecture │  │ 14 Classes  │           │
│  │ (1000×12)    │  │              │  │ + Confidence│           │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
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
│   │   │   ├── explore.tsx       # Explore/info screen
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

#### Step 4: Monitor Training

The script will automatically:
- ✅ Detect Kaggle environment
- ✅ Load PTB-XL metadata and signals
- ✅ Preprocess ECG data (normalize, pad/truncate)
- ✅ Build ResNet1D architecture
- ✅ Train with early stopping and learning rate scheduling
- ✅ Evaluate on test set
- ✅ Save model artifacts to `/kaggle/working/`

**Expected Console Output:**
```
🔍 Environment: Kaggle
📁 Data directory: /kaggle/input/ptbxl-ekg
📁 Output directory: /kaggle/working

======================================================================
📊 STEP 1: Loading PTB-XL Dataset
======================================================================
✅ Loaded 21837 ECG records from metadata
✅ Filtered to 21837 records with valid diagnostic labels

📋 Class distribution:
   NORM                          : 9528 samples (43.6%)
   MI                            : 5486 samples (25.1%)
   ...

======================================================================
🔊 STEP 2: Loading and Preprocessing ECG Signals
======================================================================
Loading ECG signals... (this may take a few minutes)
   Processed 1000/21837 records...
   Processed 2000/21837 records...
   ...

✅ Successfully loaded 21837 ECG signals
   Signal shape: (21837, 1000, 12)
   Labels shape: (21837,)

======================================================================
🎯 STEP 5: Compiling and Training Model
======================================================================
🚀 Starting training...
Epoch 1/50
   loss: 1.8234 - accuracy: 0.4521 - val_loss: 1.5432 - val_accuracy: 0.5234
Epoch 2/50
   loss: 1.4521 - accuracy: 0.5678 - val_loss: 1.3211 - val_accuracy: 0.6012
...

======================================================================
📊 STEP 6: Evaluating Model Performance
======================================================================
🎯 Test Set Results:
   Loss:      1.2345
   Accuracy:  0.7021
   Precision: 0.6834
   Recall:    0.6912
   F1-Score:  0.6873

======================================================================
💾 STEP 7: Saving Model Artifacts
======================================================================
✅ Saved label classes to: /kaggle/working/classes.npy
✅ Model weights saved to: /kaggle/working/model_weights.h5
✅ Created output package: /kaggle/working/output.zip

======================================================================
🎉 TRAINING COMPLETE!
======================================================================
```

---

### ⏱️ Expected Training Time & Output

| **Metric**               | **Value**                          |
|--------------------------|------------------------------------|
| **Total Training Time**  | ~45-90 minutes (GPU T4 x2)         |
| **Epochs**               | 30-50 (early stopping)             |
| **Test Accuracy**        | ~65-75%                            |
| **Model Size**           | ~15-20 MB (model_weights.h5)       |
| **Output Files**         | `model_weights.h5`, `classes.npy`  |

**What Gets Saved:**

1. **`model_weights.h5`** (15-20 MB)
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

### 📥 Downloading the Model

#### Step 1: Locate Output Files

1. Look at the **Output** panel in Kaggle (bottom-right)
2. You should see:
   - `model_weights.h5`
   - `classes.npy`
   - `output.zip`

#### Step 2: Download

**Option A: Download Individual Files**
- Click the **⋮** (three dots) next to each file
- Select **"Download"**

**Option B: Download output.zip (Recommended)**
- Click the **⋮** next to `output.zip`
- Select **"Download"**
- Extract locally to get both files

#### Step 3: Verify Files

```bash
# Check file sizes
ls -lh model_weights.h5 classes.npy

# Expected output:
# -rw-r--r-- 1 user user  18M Dec 28 12:00 model_weights.h5
# -rw-r--r-- 1 user user 512B Dec 28 12:00 classes.npy
```

#### Step 4: Test Model Locally (Optional)

```python
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model('model_weights.h5')

# Load classes
classes = np.load('classes.npy', allow_pickle=True)
print(f"Classes: {classes}")

# Test with dummy data
dummy_ecg = np.random.randn(1, 1000, 12)
predictions = model.predict(dummy_ecg)
predicted_class = classes[np.argmax(predictions)]
print(f"Predicted: {predicted_class} (confidence: {predictions.max():.2%})")
```

---

## 🔧 Phase 3: Backend API (Coming Soon)

> **Status:** 🚧 To be implemented in next phase

### Planned Features

- **FastAPI REST API** for mobile app communication
- **ECG Image Digitizer** using OpenCV/computer vision
- **ML Model Inference Service** with TensorFlow Serving
- **Preprocessing Pipeline** for signal normalization
- **Response Caching** for faster repeated queries

### Placeholder Structure

```
backend-api/
├── main.py                  # FastAPI app with endpoints
├── services/
│   ├── digitizer.py         # Image → signal conversion
│   └── classifier.py        # Model inference
├── models/
│   ├── model_weights.h5     # Copy from Kaggle output
│   └── classes.npy          # Copy from Kaggle output
├── requirements.txt         # Backend dependencies
└── Dockerfile               # For deployment
```

### Planned API Endpoints

```python
POST /api/v1/analyze-ecg
{
  "image_base64": "...",      # Base64-encoded ECG image
  "calibration": {
    "mm_per_pixel": 0.025,
    "paper_speed": 25
  }
}

Response:
{
  "predicted_class": "Atrial Fibrillation",
  "confidence": 0.89,
  "all_predictions": [
    {"class": "AFib", "probability": 0.89},
    {"class": "Normal", "probability": 0.05},
    ...
  ],
  "heart_rate_bpm": 92,
  "recommendation": "Irregular rhythm detected. Consult cardiologist."
}
```

---

## 📱 Phase 4: Mobile App Integration (Coming Soon)

> **Status:** 🚧 Mobile UI complete, backend integration pending

### Current Status

✅ **Completed:**
- React Native app with ECG camera capture
- Manual grid calibration (two-point calibration)
- R-peak marking and BPM calculation
- Beautiful UI with real-time feedback

❌ **Pending:**
- Backend API integration
- Image upload to server
- Display ML model predictions
- Offline mode with cached results

### Integration Steps (Next Phase)

1. **Connect to Backend API**
   ```typescript
   const analyzeECG = async (imageUri: string, calibration: any) => {
     const response = await fetch('https://your-api.com/api/v1/analyze-ecg', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
         image_base64: await imageToBase64(imageUri),
         calibration: calibration
       })
     });
     return response.json();
   };
   ```

2. **Display Predictions**
   - Show predicted arrhythmia class
   - Display confidence percentage
   - Highlight concerning findings (AFib, MI, etc.)

3. **Add History Tracking**
   - Store past ECG analyses locally
   - Allow users to review trends over time

---

## 🔬 Technical Specifications

### Model Architecture: ResNet1D

```
Input: (1000, 12) - 10 seconds @ 100Hz, 12 ECG leads

Initial Conv1D (64 filters, kernel=7, stride=2)
    ↓
Residual Block 1 (64 filters)
    ↓
Residual Block 2 (128 filters, stride=2)
    ↓
Residual Block 3 (256 filters, stride=2)
    ↓
Residual Block 4 (512 filters, stride=2)
    ↓
Global Average Pooling
    ↓
Dense (256) → ReLU → Dropout(0.5)
    ↓
Dense (128) → ReLU → Dropout(0.3)
    ↓
Dense (14) → Softmax

Output: (14,) - Probability distribution over 14 classes
```

**Residual Block Structure:**
```
x → Conv1D → BatchNorm → ReLU → Dropout → Conv1D → BatchNorm
    ↓                                                  ↓
    └──────────────── Skip Connection ────────────────┴→ Add → ReLU
```

### Signal Preprocessing

1. **Loading:** Read WFDB format (`.hea` + `.dat` files)
2. **Normalization:** Z-score normalization per lead
3. **Resampling:** Standardize to 100Hz (if needed)
4. **Padding/Truncation:** Ensure exactly 1000 samples (10 seconds)
5. **Lead Selection:** Use all 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)

### Training Configuration

| **Parameter**         | **Value**                       |
|-----------------------|---------------------------------|
| Optimizer             | Adam (lr=0.001)                |
| Loss Function         | Categorical Crossentropy       |
| Batch Size            | 32                             |
| Epochs                | 50 (early stopping)            |
| Learning Rate Decay   | ReduceLROnPlateau (factor=0.5) |
| Class Weighting       | Inverse frequency weighting    |
| Train/Val/Test Split  | 70% / 15% / 15%                |

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

### Citation

If you use PTB-XL in research, please cite:

```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and others},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={154},
  year={2020},
  publisher={Nature Publishing Group}
}
```

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

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🚀 Quick Start Checklist

- [ ] Phase 1-2: Train model on Kaggle
  - [ ] Create Kaggle account
  - [ ] Add PTB-XL dataset
  - [ ] Run `kaggle_train.py`
  - [ ] Download `output.zip`
- [ ] Phase 3: Build backend API
  - [ ] Set up FastAPI project
  - [ ] Implement image digitizer
  - [ ] Deploy model inference service
- [ ] Phase 4: Connect mobile app
  - [ ] Integrate backend API calls
  - [ ] Test end-to-end workflow
  - [ ] Deploy to App Store / Google Play

---

## 📞 Support

For questions or issues:
- Open a GitHub Issue
- Email: [your-email@example.com]
- Check our [Wiki](https://github.com/your-repo/wiki) for FAQs

---

**Made with ❤️ by the ECG Diagnosis Team**
