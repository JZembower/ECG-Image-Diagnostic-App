# ECG Analyzer Mobile App

A professional React Native mobile application for AI-powered ECG diagnosis with real-time backend integration. This app allows users to capture ECG images, calibrate measurements, and receive instant diagnostic analysis powered by a deep learning backend.

## 📱 Features

- **Image Capture**: Take photos or select from gallery
- **Smart Calibration**: Optional grid calibration for accurate measurements
- **Real-time Backend Integration**: Connects to FastAPI backend for AI diagnosis
- **Professional UI/UX**: Medical-grade design with clinical color palette
- **Comprehensive Error Handling**: User-friendly error messages and recovery options
- **Loading States**: Professional progress indicators during analysis
- **Detailed Results**: Diagnosis, BPM, confidence levels, and signal quality
- **Backend Health Monitoring**: Real-time connection status

## 🏗️ Architecture

### Project Structure

```
mobile-app/
├── src/
│   └── screens/
│       ├── index.tsx          # Main ECG analyzer screen (enhanced)
│       ├── explore.tsx         # Explore/documentation screen
│       ├── _layout.tsx         # Tab navigation layout
│       └── App.js             # Legacy app file (kept for reference)
├── services/
│   └── api.js                 # Backend API communication service
├── utils/
│   ├── logger.js              # Logging utility
│   └── errorHandler.js        # Error handling utility
├── components/
│   ├── ErrorBoundary.js       # React error boundary
│   └── LoadingIndicator.js    # Loading components
├── config/
│   └── config.js              # Configuration management
├── package.json               # Dependencies and scripts
└── README.md                  # This file
```

### Component Hierarchy

```
ECGAnalyzer (index.tsx)
├── LoadingIndicator
│   └── ProgressLoadingIndicator
├── Image Capture Section
│   ├── Camera Button
│   ├── Gallery Button
│   └── Clear Button
├── Configuration Controls
│   ├── Grid Size Input
│   ├── Paper Speed Input
│   └── Analyze Button
├── Image Display
│   ├── Selected Image
│   └── Calibration Overlay (SVG)
├── Calibration Status
├── Error Display (conditional)
└── Results Display (conditional)
    ├── Diagnosis Card
    ├── Heart Rate Display
    ├── Metrics Grid
    ├── Technical Details
    └── Medical Disclaimer
```

## 🚀 Getting Started

### Prerequisites

- **Node.js**: v16 or higher
- **npm** or **yarn**: Latest version
- **Expo CLI**: `npm install -g expo-cli`
- **iOS Simulator** (Mac only) or **Android Emulator**
- **Physical Device** (optional, recommended for testing)
- **Backend API**: Running at configured URL (default: http://localhost:8000)

### Installation

1. **Clone the repository**:
   ```bash
   cd /path/to/ecg-diagnosis-system/mobile-app
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure backend URL**:
   
   Edit `config/config.js` to set your backend API URL:
   ```javascript
   const config = {
     development: {
       API_BASE_URL: 'http://localhost:8000',  // Change this to your backend URL
       // ...
     },
   };
   ```

   **Important Notes**:
   - For **iOS Simulator**: Use `http://localhost:8000`
   - For **Android Emulator**: Use `http://10.0.2.2:8000`
   - For **Physical Device**: Use your computer's local IP (e.g., `http://192.168.1.100:8000`)

4. **Start the backend server** (in another terminal):
   ```bash
   cd ../backend-api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Running the App

1. **Start Expo development server**:
   ```bash
   npm start
   # or
   expo start
   ```

2. **Run on specific platform**:
   ```bash
   # iOS (Mac only)
   npm run ios
   # or
   expo start --ios

   # Android
   npm run android
   # or
   expo start --android

   # Web (limited functionality)
   npm run web
   ```

3. **Run on physical device**:
   - Install **Expo Go** app from App Store or Play Store
   - Scan the QR code shown in terminal
   - Ensure device is on the same network as development machine

## 🔧 Configuration

### Environment Configuration

The app supports three environments: **development**, **staging**, and **production**.

#### Development Configuration

Edit `config/config.js`:

```javascript
development: {
  API_BASE_URL: 'http://localhost:8000',
  API_TIMEOUT: 30000,
  ENABLE_AUTO_DETECTION: true,
  LOG_LEVEL: 'debug',
  ENABLE_LOGGING: true,
}
```

#### Production Configuration

For production builds, update the production section:

```javascript
production: {
  API_BASE_URL: 'https://api.ecg-diagnosis.com',
  API_TIMEOUT: 30000,
  LOG_LEVEL: 'error',
  ENABLE_LOGGING: false,
}
```

### Runtime URL Override

For testing with different backend URLs without changing code:

```javascript
import { setApiUrl } from './config/config';

// Override at app startup
setApiUrl('http://192.168.1.100:8000');
```

## 📖 Usage Guide

### Basic Workflow

1. **Capture or Select ECG Image**:
   - Tap "📸 Take Photo" to use camera
   - Tap "📁 Gallery" to select from photo library
   - Ensure ECG waveform is clearly visible

2. **Optional: Calibrate Grid**:
   - Tap opposite corners of one grid square (usually 5mm x 5mm)
   - First tap: marks first corner
   - Second tap: completes calibration
   - Skip calibration if grid is not visible (backend uses defaults)

3. **Configure Settings** (optional):
   - Set **Grid Size**: Standard is 5mm, adjust if different
   - Set **Paper Speed**: Standard is 25mm/s, common values are 25 or 50

4. **Analyze ECG**:
   - Tap "🧠 Analyze ECG" button
   - Wait for backend processing (typically 5-15 seconds)
   - View results

5. **Review Results**:
   - **Diagnosis**: Primary condition detected
   - **Heart Rate**: BPM with classification (Normal/Bradycardia/Tachycardia)
   - **Confidence**: AI model confidence level
   - **Signal Quality**: Quality of extracted signal

### Backend Health Check

- Green dot (●): Backend connected and healthy
- Red dot (●): Backend offline or unreachable
- Tap 🔄 icon to retry connection

### Error Recovery

If analysis fails:
1. Check error message for specific issue
2. Tap "Try Again" to retry with same image
3. If problem persists:
   - Check backend is running
   - Verify network connection
   - Try with a different, clearer image

## 🧪 Testing

### Testing with Sample Images

1. **Prepare sample ECG images**:
   - Use provided test images from `../test-images/` directory
   - Or download ECG samples from medical databases

2. **Test calibration**:
   - Use images with clear grid lines
   - Verify calibration accuracy by checking mm/pixel value

3. **Test without calibration**:
   - Try images without visible grids
   - Backend should use default calibration

### Testing Backend Integration

1. **Test with backend running**:
   ```bash
   # Terminal 1: Start backend
   cd backend-api
   uvicorn main:app --reload

   # Terminal 2: Start mobile app
   cd mobile-app
   npm start
   ```

2. **Test error scenarios**:
   - Stop backend server (test offline error)
   - Send invalid image (test validation error)
   - Send low-quality image (test quality error)

### Manual Testing Checklist

- [ ] Camera permission request
- [ ] Gallery permission request
- [ ] Image capture works
- [ ] Image selection works
- [ ] Calibration tap detection
- [ ] Calibration calculation
- [ ] Clear calibration button
- [ ] Backend health check
- [ ] Analyze button (enabled/disabled states)
- [ ] Loading indicator during analysis
- [ ] Progress steps display
- [ ] Successful analysis results
- [ ] Error handling and display
- [ ] Retry after error
- [ ] Clear all functionality
- [ ] Results display formatting
- [ ] Confidence color coding
- [ ] BPM classification

## 🛠️ Development

### Code Style

- **Language**: TypeScript for screens, JavaScript for services/utils
- **Formatting**: 2 spaces indentation
- **Comments**: JSDoc style for functions
- **Naming**: camelCase for functions/variables, PascalCase for components

### Adding New Features

1. **Add service functions** in `services/api.js`
2. **Add utilities** in `utils/` directory
3. **Update UI** in `src/screens/index.tsx`
4. **Test thoroughly** with backend integration

### Logging

Use the logger utility for consistent logging:

```javascript
import logger from '../../utils/logger';

// Different log levels
logger.debug('Debug information', { data });
logger.info('General information', { data });
logger.warn('Warning message', { data });
logger.error('Error occurred', { error: err.message });

// Special logging
logger.logUserAction('button_clicked', { button: 'analyze' });
logger.logApiCall('/predict', 'POST', { hasCalibration: true });
```

### Error Handling

Use the error handler utility:

```javascript
import { handleError, createErrorAlert } from '../../utils/errorHandler';

try {
  // Your code
} catch (err) {
  const errorAlert = handleError(err, 'performing action');
  Alert.alert(errorAlert.title, errorAlert.message);
}
```

## 🏭 Building for Production

### iOS Build

1. **Configure for production**:
   ```bash
   expo build:ios
   ```

2. **Follow Expo prompts** to configure bundle identifier and provisioning

3. **Submit to App Store** via Expo or manually

### Android Build

1. **Configure for production**:
   ```bash
   expo build:android
   ```

2. **Choose APK or AAB** format

3. **Download and distribute** or submit to Play Store

### Environment Variables

For production builds, ensure:
- `API_BASE_URL` points to production backend
- `LOG_LEVEL` is set to 'error'
- `ENABLE_LOGGING` is false

## 🐛 Troubleshooting

### Common Issues

#### "Unable to connect to diagnostic server"

**Solutions**:
- Verify backend is running: `curl http://localhost:8000/health`
- Check API URL in `config/config.js`
- For Android emulator, use `http://10.0.2.2:8000` instead of `localhost`
- For physical device, use computer's local IP address

#### "Camera permission denied"

**Solutions**:
- Grant camera permission in device settings
- Restart the app
- On iOS: Settings → Privacy → Camera → [App Name]
- On Android: Settings → Apps → [App Name] → Permissions

#### "Analysis taking too long"

**Solutions**:
- Check backend server logs for errors
- Ensure image is not too large (< 10MB)
- Try with a smaller/compressed image
- Check network connection speed

#### Image not displaying after capture

**Solutions**:
- Check image picker permissions
- Verify image URI is valid
- Check console logs for errors
- Try gallery selection instead of camera

### Debug Mode

Enable detailed logging:

```javascript
// In config/config.js
LOG_LEVEL: 'debug',
ENABLE_LOGGING: true,
```

View logs:
- **Metro bundler console**: Check terminal running `npm start`
- **Device logs**: Use React Native Debugger or Flipper
- **Expo logs**: Check Expo Dev Tools in browser

## 📚 API Integration

### Backend Endpoints Used

#### Health Check
```
GET /health
Response: { status: "healthy", model_loaded: true }
```

#### ECG Analysis
```
POST /predict
Body: multipart/form-data
  - image: file
  - calibration_points: JSON (optional)
  - paper_speed: number (optional)
  - grid_size: number (optional)

Response: {
  diagnosis: string,
  bpm: number,
  confidence: number,
  signal_quality: string,
  all_probabilities: object
}
```

### Request Format

```javascript
// Calibration points format
{
  point1: { x: 100, y: 50 },
  point2: { x: 200, y: 150 }
}

// FormData structure
const formData = new FormData();
formData.append('image', {
  uri: imageUri,
  type: 'image/jpeg',
  name: 'ecg.jpg',
});
formData.append('calibration_points', JSON.stringify(calibrationPoints));
formData.append('paper_speed', '25');
formData.append('grid_size', '5');
```

## 🔐 Security & Privacy

- **Local Processing**: No data is stored on servers (stateless backend)
- **Network Security**: Use HTTPS in production
- **Permissions**: Only requests necessary permissions (camera, photos)
- **Data Privacy**: Images are processed in real-time and not persisted

## 📄 License

This project is part of the ECG Diagnosis System. See main project README for license information.

## 🤝 Contributing

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull request with clear description

## 👥 Authors

- **Original Contributors**: Elissa Matlock, Eugene Ho, Jonah Zembower
- **Enhanced Mobile Integration**: AI-powered improvements

## 📞 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Check backend API documentation
3. Review console logs for error details
4. Contact development team

## 🚧 Future Enhancements

- [ ] User account system
- [ ] Analysis history storage
- [ ] Report export (PDF)
- [ ] Share results with doctors
- [ ] Multi-image batch analysis
- [ ] Offline mode with local model
- [ ] Real-time ECG monitoring
- [ ] Integration with wearable devices
- [ ] Multi-language support
- [ ] Accessibility improvements

---

**⚠️ Medical Disclaimer**

This application is for educational and informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
