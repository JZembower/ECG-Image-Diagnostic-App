/**
 * Configuration Module
 * * Manages API endpoints and app configuration across different environments.
 * Supports development, staging, and production environments.
 */

// Environment detection
const ENV = __DEV__ ? 'development' : process.env.NODE_ENV || 'production';

// Configuration for different environments
const config = {
  development: {
    // -----------------------------------------------------------
    // CHANGE: Updated localhost to your specific Local IP
    // This allows your physical phone to see your laptop backend.
    // -----------------------------------------------------------
    API_BASE_URL: 'http://192.168.68.59:8000', 
    API_TIMEOUT: 30000, // 30 seconds
    ENABLE_AUTO_DETECTION: true,
    LOG_LEVEL: 'debug',
    ENABLE_LOGGING: true,
  },
  staging: {
    API_BASE_URL: 'https://staging-api.ecg-diagnosis.com',
    API_TIMEOUT: 30000,
    ENABLE_AUTO_DETECTION: true,
    LOG_LEVEL: 'info',
    ENABLE_LOGGING: true,
  },
  production: {
    API_BASE_URL: 'https://api.ecg-diagnosis.com',
    API_TIMEOUT: 30000,
    ENABLE_AUTO_DETECTION: true,
    LOG_LEVEL: 'error',
    ENABLE_LOGGING: false,
  },
};

// Get current environment configuration
const currentConfig = config[ENV];

// App constants
export const APP_VERSION = '1.0.0';
export const APP_NAME = 'ECG Analyzer';

// API Configuration
export const API_BASE_URL = currentConfig.API_BASE_URL;
export const API_TIMEOUT = currentConfig.API_TIMEOUT;
export const ENABLE_AUTO_DETECTION = currentConfig.ENABLE_AUTO_DETECTION;

// Logging Configuration
export const LOG_LEVEL = currentConfig.LOG_LEVEL;
export const ENABLE_LOGGING = currentConfig.ENABLE_LOGGING;

// ECG Analysis Constants
export const ECG_CONSTANTS = {
  DEFAULT_PAPER_SPEED: 25, // mm/s
  DEFAULT_GRID_SIZE: 5, // mm
  MIN_PEAKS_REQUIRED: 2,
  MIN_BPM: 40,
  MAX_BPM: 200,
  NORMAL_BPM_RANGE: { min: 60, max: 100 },
};

// Image Configuration
export const IMAGE_CONFIG = {
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  QUALITY: 0.8, // 80% quality for compression
  SUPPORTED_FORMATS: ['jpg', 'jpeg', 'png'],
};

// Export environment for runtime checks
export const CURRENT_ENV = ENV;

// Helper function to override API URL at runtime (useful for testing)
let apiUrlOverride = null;

export const setApiUrl = (url) => {
  apiUrlOverride = url;
};

export const getApiUrl = () => {
  return apiUrlOverride || API_BASE_URL;
};

// Export full config object for debugging
export const getFullConfig = () => ({
  environment: ENV,
  apiBaseUrl: getApiUrl(),
  apiTimeout: API_TIMEOUT,
  enableAutoDetection: ENABLE_AUTO_DETECTION,
  logLevel: LOG_LEVEL,
  enableLogging: ENABLE_LOGGING,
  appVersion: APP_VERSION,
  appName: APP_NAME,
});

export default {
  API_BASE_URL: getApiUrl(),
  API_TIMEOUT,
  ENABLE_AUTO_DETECTION,
  LOG_LEVEL,
  ENABLE_LOGGING,
  APP_VERSION,
  APP_NAME,
  ECG_CONSTANTS,
  IMAGE_CONFIG,
  CURRENT_ENV,
  setApiUrl,
  getApiUrl,
  getFullConfig,
};