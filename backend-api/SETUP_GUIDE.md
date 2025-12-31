# Quick Setup Guide - ECG Diagnosis Backend

This guide will help you get the backend up and running in minutes.

## 📦 What You'll Need

1. **Python 3.10+** installed
2. **Model files** (from Kaggle training pipeline)
   - `model.weights.h5`
   - `classes.npy`

## 🚀 Quick Start (5 Steps)

### Step 1: Navigate to Backend Directory

```bash
cd .../ecg-diagnosis-system/backend-api
```

### Step 2: Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will install TensorFlow (~500MB) and other dependencies. It may take 5-10 minutes.

### Step 4: Add Model Files

Place your trained model files in the `models/` directory:

```bash
# Example: Copy from downloads or training pipeline output
cp /path/to/model.weights.h5 models/
cp /path/to/classes.npy models/

# Verify files
ls -lh models/
```

**Don't have model files yet?** 
- Run the training pipeline: `.../ecg-diagnosis-system/training-pipeline/kaggle_train.py` on Kaggle
- Or download pre-trained models (if available)
- See [main README](../README.md) for training instructions

### Step 5: Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## ✅ Verify Setup

### Test 1: Health Check

Open a new terminal:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_file_exists": true,
  "classes_file_exists": true
}
```

### Test 2: API Documentation

Open browser: http://localhost:8000/docs

You should see interactive API documentation (Swagger UI).

### Test 3: Make a Prediction (with sample image)

```bash
# Download sample ECG image or use one you have
curl -X POST \
  -F "image=@/path/to/ecg_image.jpg" \
  http://localhost:8000/predict
```

Expected response:
```json
{
  "diagnosis": "Normal",
  "confidence": 0.87,
  "bpm": 75,
  "signal_quality": "good",
  "all_probabilities": {...}
}
```

## 🔧 Troubleshooting

### Issue: Dependencies won't install

**Error**: `ERROR: Could not find a version that satisfies the requirement...`

**Solution**:
```bash
# Update pip
pip install --upgrade pip

# Try installing TensorFlow separately
pip install tensorflow>=2.14.0

# Then install remaining dependencies
pip install -r requirements.txt
```

### Issue: Model files not found

**Error**: `FileNotFoundError: Model file not found`

**Solution**:
1. Ensure files are in `models/` directory
2. Check file names exactly: `model.weights.h5` and `classes.npy`
3. Verify file permissions: `chmod 644 models/*`

### Issue: Port already in use

**Error**: `[Errno 98] Address already in use`

**Solution**:
```bash
# Use a different port
uvicorn main:app --port 8001

# Or kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

### Issue: TensorFlow not working

**Error**: Various TensorFlow import errors

**Solution**:
```bash
# Try CPU-only version (smaller, works everywhere)
pip uninstall tensorflow
pip install tensorflow-cpu>=2.14.0
```

## 🐳 Docker Alternative

If you prefer Docker (no Python setup needed):

```bash
# Build image
docker build -t ecg-backend .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name ecg-backend \
  ecg-backend

# View logs
docker logs -f ecg-backend

# Test
curl http://localhost:8000/health
```

## 📱 Connect Mobile App

Once backend is running, update your mobile app configuration:

```typescript
// In mobile app config
const API_URL = "http://YOUR_IP_ADDRESS:8000";

// For local testing on same machine:
const API_URL = "http://localhost:8000";

// For testing from mobile device on same network:
const API_URL = "http://192.168.1.X:8000";  // Replace with your IP
```

**Find your IP address**:
```bash
# Linux/Mac
ifconfig | grep "inet "

# Or
hostname -I
```

## 🧪 Run Setup Test

We've included a test script to verify everything is working:

```bash
python test_setup.py
```

This will check:
- ✓ All required packages are installed
- ✓ All required files are present
- ✓ Model files exist
- ✓ Services can be imported

## 📚 Next Steps

1. **Test with real ECG images**: Try uploading different ECG images
2. **Review API docs**: http://localhost:8000/docs
3. **Integrate with mobile app**: See Phase 4 instructions
4. **Deploy to production**: See deployment guide in README.md
5. **Monitor performance**: Add logging and metrics

## 💡 Tips

- **Development mode**: Use `--reload` flag for auto-reload on code changes
- **Production mode**: Use `gunicorn` with multiple workers (see README.md)
- **Logging**: Set `LOG_LEVEL=DEBUG` in `.env` for verbose logs
- **CORS**: Update `CORS_ORIGINS` in `.env` for security

## 📖 Documentation

- **Full README**: [README.md](README.md)
- **API Docs**: http://localhost:8000/docs
- **Main Project**: [../README.md](../README.md)
- **Training Pipeline**: [../training-pipeline/README.md](../training-pipeline/README.md)

## 🆘 Still Having Issues?

1. Check logs for detailed error messages
2. Review troubleshooting section in README.md
3. Verify Python version: `python --version` (should be 3.10+)
4. Ensure all dependencies installed: `pip list`
5. Try running test_setup.py for diagnostic info

---

**Success?** You should now have a running ECG diagnosis backend! 🎉

**Next**: Integrate with mobile app or test predictions with sample ECG images.
