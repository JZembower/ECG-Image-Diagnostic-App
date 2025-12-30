# ECG Diagnosis Backend API

Production-ready FastAPI service that converts ECG images to diagnostic results using computer vision and deep learning.

## 🏗️ Architecture Overview

The backend consists of three main components:

1. **FastAPI Server (`main.py`)**: RESTful API with CORS, validation, and error handling
2. **Digitizer Service (`services/digitizer.py`)**: OpenCV-based ECG signal extraction from images
3. **Classifier Service (`services/classifier.py`)**: TensorFlow-based arrhythmia classification

### Processing Pipeline

```
ECG Image → Digitizer → Signal (1000, 12) → Classifier → Diagnosis + BPM + Confidence
```

**Digitizer Steps:**
1. Preprocess image (grayscale, blur, adaptive threshold)
2. Remove grid lines (morphological operations)
3. Extract ECG trace (contour detection)
4. Convert pixels to physical units (mV, seconds)
5. Resample to 100Hz (1000 samples)
6. Create 12-lead signal (replicate or extract)
7. Normalize (Z-score)
8. Assess signal quality

**Classifier Steps:**
1. Load cached model (ResNet1D trained on PTB-XL)
2. Preprocess signal (ensure shape, normalization)
3. Run inference (14-class arrhythmia classification)
4. Calculate BPM (R-peak detection)
5. Return diagnosis with confidence

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Docker** (optional, for containerized deployment)
- **Model Files**: `model.weights.h5` and `classes.npy` (see [Model Setup](#model-setup))

## 🚀 Quick Start

### Option 1: Local Development (Python Virtual Environment)

```bash
# Navigate to backend-api directory
cd .../ecg-diagnosis-system/backend-api

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up model files (see Model Setup section below)
# Place model.weights.h5 and classes.npy in models/ directory

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server will start at http://localhost:8000
# API docs: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Option 2: Docker Deployment

```bash
# Build Docker image
docker build -t ecg-backend:latest .

# Run container (mount models directory)
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name ecg-backend \
  ecg-backend:latest

# Check logs
docker logs -f ecg-backend

# Stop container
docker stop ecg-backend
```

### Option 3: Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧠 Model Setup

The backend requires trained model files from the training pipeline:

### Step 1: Train the Model (or Download from Kaggle)

**Option A: Train on Kaggle**
1. Upload `training-pipeline/kaggle_train.py` to Kaggle
2. Add PTB-XL dataset to notebook
3. Run the script (generates `model.weights.h5` and `classes.npy`)
4. Download `output.zip` from Kaggle output

**Option B: Use Pre-trained Model**
- If you have pre-trained files, skip to Step 2

### Step 2: Place Model Files

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Extract and place files
# From Kaggle output.zip:
unzip output.zip
mv model.weights.h5 models/
mv classes.npy models/

# Verify files
ls -lh models/
# Should show:
# - model.weights.h5 (~2-10MB depending on architecture)
# - classes.npy (~few KB)
```

### Step 3: Verify Model Loading

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_file_exists": true,
#   "classes_file_exists": true,
#   ...
# }
```

## 📡 API Documentation

### Endpoints

#### `GET /` - API Information
Returns basic API information and documentation links.

```bash
curl http://localhost:8000/
```

#### `GET /health` - Health Check
Check if API is running and model is loaded.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_file_exists": true,
  "classes_file_exists": true
}
```

#### `POST /predict` - ECG Diagnosis
Upload ECG image and get diagnostic prediction.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (required): ECG image file (JPG, PNG, etc.)
  - `calibration_points` (optional): JSON string with pixel-to-physical mappings

**Example with curl:**
```bash
# Simple prediction (no calibration)
curl -X POST \
  -F "image=@path/to/ecg_image.jpg" \
  http://localhost:8000/predict

# With calibration points
curl -X POST \
  -F "image=@path/to/ecg_image.jpg" \
  -F 'calibration_points=[{"x":100,"y":200,"value":1.0,"unit":"mV"}]' \
  http://localhost:8000/predict
```

**Example with Python:**
```python
import requests

url = "http://localhost:8000/predict"

# Upload image
with open("ecg_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

result = response.json()
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"BPM: {result['bpm']}")
print(f"Signal Quality: {result['signal_quality']}")
```

**Response:**
```json
{
  "diagnosis": "Atrial Fibrillation",
  "confidence": 0.95,
  "bpm": 72,
  "signal_quality": "good",
  "all_probabilities": {
    "Normal": 0.02,
    "Atrial Fibrillation": 0.95,
    "1st Degree AV Block": 0.01,
    ...
  }
}
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing the API

### Test with Sample ECG Image

```bash
# Download sample ECG image (if you don't have one)
# You can use the ptbxl-all-waves.html file to view sample ECGs

# Test prediction
curl -X POST \
  -F "image=@sample_ecg.jpg" \
  http://localhost:8000/predict \
  | jq '.'
```

### Expected Output
```json
{
  "diagnosis": "Normal",
  "confidence": 0.87,
  "bpm": 75,
  "signal_quality": "good",
  "all_probabilities": {
    "Normal": 0.87,
    "Atrial Fibrillation": 0.05,
    ...
  }
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```env
# Model paths
MODEL_PATH=.../ecg-diagnosis-system/backend-api/models/model.weights.h5
CLASSES_PATH=.../ecg-diagnosis-system/backend-api/models/classes.npy

# Logging
LOG_LEVEL=INFO

# CORS (comma-separated origins)
CORS_ORIGINS=*

# Server (optional, defaults shown)
HOST=0.0.0.0
PORT=8000
```

### Load Environment Variables

```bash
# Install python-dotenv
pip install python-dotenv

# Load in code (already implemented in main.py)
from dotenv import load_dotenv
load_dotenv()
```

## 📊 Signal Quality Assessment

The digitizer assesses signal quality using multiple metrics:

- **SNR (Signal-to-Noise Ratio)**: High-frequency noise analysis
- **Baseline Wander**: Low-frequency drift detection
- **Signal Variability**: Standard deviation check
- **Clipping Detection**: Saturation check

**Quality Levels:**
- `excellent`: SNR > 20dB, minimal baseline wander
- `good`: SNR > 15dB, acceptable quality
- `fair`: SNR > 10dB, usable but noisy
- `poor`: Low SNR, unreliable results

## 🐛 Troubleshooting

### Issue: Model files not found

**Error:**
```
FileNotFoundError: Model file not found: /path/to/model.weights.h5
```

**Solution:**
1. Ensure `model.weights.h5` and `classes.npy` are in the `models/` directory
2. Check file permissions: `chmod 644 models/*.{h5,npy}`
3. Verify paths in `.env` or environment variables

### Issue: Failed to extract ECG signal

**Error:**
```
Failed to extract ECG signal from image: No ECG trace detected
```

**Possible Causes:**
- Image quality too poor
- ECG trace not visible/clear
- Grid removal removed the trace

**Solutions:**
1. Use higher resolution images
2. Ensure ECG trace is dark on light background
3. Adjust preprocessing parameters in `digitizer.py`

### Issue: TensorFlow not installed

**Error:**
```
TensorFlow is not installed
```

**Solution:**
```bash
pip install tensorflow>=2.14.0

# Or for CPU-only (smaller size):
pip install tensorflow-cpu>=2.14.0
```

### Issue: OpenCV import error

**Error:**
```
ImportError: libGL.so.1: cannot open shared object file
```

### Issue: Poor prediction accuracy

**Possible Causes:**
- Poor signal quality
- Image not properly digitized
- Model not trained properly

**Solutions:**
1. Check signal quality in response (should be "good" or "excellent")
2. Use calibration points for better digitization
3. Retrain model with more data
4. Verify model files are correct version

## 🔐 Security Considerations

### Production Deployment Checklist

- [ ] Use HTTPS (TLS/SSL certificates)
- [ ] Restrict CORS origins (don't use `*` in production)
- [ ] Implement authentication (JWT, OAuth2)
- [ ] Rate limiting (e.g., using `slowapi`)
- [ ] Input validation and sanitization
- [ ] Log sensitive operations (HIPAA compliance if medical use)
- [ ] Use non-root user in Docker (already implemented)
- [ ] Regular security updates

### Example: Add JWT Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement token verification
    if not verify_jwt(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict(...):
    # Endpoint now requires valid JWT
    ...
```

## 📦 Project Structure

```
backend-api/
├── main.py                 # FastAPI app entry point
├── services/
│   ├── __init__.py
│   ├── digitizer.py        # Image-to-signal conversion
│   └── classifier.py       # Model inference
├── models/                 # Model files (not in git)
│   ├── model.weights.h5
│   └── classes.npy
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Multi-container setup
├── .env.example            # Environment variable template
├── .gitignore              # Ignore model files, venv, etc.
└── README.md               # This file
```

## 🚦 Performance Optimization

### Tips for Production

1. **Use TensorFlow Lite** for faster inference:
   ```bash
   # Convert model to TFLite
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   ```

2. **Enable model caching** (already implemented):
   - Model loaded once and cached in memory
   - Subsequent requests reuse cached model

3. **Use Gunicorn** for production server:
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

4. **Add Redis caching** for frequent predictions:
   ```bash
   pip install redis
   # Cache predictions based on image hash
   ```

## 📝 License

This project is for educational and research purposes only. **Not intended for clinical use.**

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check [main project README](../README.md) for more details
- Review training pipeline documentation

## 🎯 Next Steps

- [ ] Implement proper 12-lead extraction from standard layouts
- [ ] Add support for different ECG paper speeds and scales
- [ ] Implement calibration using provided calibration points
- [ ] Add unit tests for digitizer and classifier
- [ ] Add integration tests for API endpoints
- [ ] Implement caching for predictions
- [ ] Add monitoring and metrics (Prometheus)
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Mobile app integration (Phase 4)

---

**Note**: This backend assumes localhost refers to the computer running the service. For remote access or mobile app integration, you'll need to deploy on a server with a public IP or domain name.
