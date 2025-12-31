/**
 * API Service Module
 * 
 * Handles all backend communication for ECG diagnosis.
 * Provides functions for uploading ECG images, checking backend health,
 * and handling API responses.
 */

import { getApiUrl, API_TIMEOUT } from '../config/config';
import logger from '../utils/logger';

/**
 * API Error Class
 * Custom error class for API-related errors
 */
export class ApiError extends Error {
  constructor(message, statusCode, originalError = null) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.originalError = originalError;
  }
}

/**
 * Format calibration points for API
 * Converts app's calibration format to backend API format
 * 
 * @param {Array} points - Array of calibration points [{x, y}, {x, y}]
 * @returns {Object} Formatted calibration data
 */
export const formatCalibrationPoints = (points) => {
  if (!points || points.length < 2) {
    return null;
  }

  return {
    point1: { x: points[0].x, y: points[0].y },
    point2: { x: points[1].x, y: points[1].y },
  };
};

/**
 * Check Backend Health
 * Verifies that the backend API is running and accessible
 * 
 * @returns {Promise<Object>} Health status response
 * @throws {ApiError} If backend is not accessible
 */
export const checkBackendHealth = async () => {
  const url = `${getApiUrl()}/health`;
  
  logger.info('Checking backend health', { url });
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for health check
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new ApiError(
        'Backend health check failed',
        response.status
      );
    }
    
    const data = await response.json();
    logger.info('Backend health check successful', data);
    
    return data;
  } catch (error) {
    if (error.name === 'AbortError') {
      logger.error('Backend health check timeout', { url });
      throw new ApiError(
        'Backend server is not responding',
        0,
        error
      );
    }
    
    logger.error('Backend health check failed', { error: error.message });
    throw new ApiError(
      'Unable to connect to diagnostic server',
      0,
      error
    );
  }
};

/**
 * Upload ECG Image for Analysis
 * Main function to send ECG image and calibration points to backend
 * 
 * @param {string} imageUri - Local URI of the ECG image
 * @param {Array} calibrationPoints - Array of calibration points (optional)
 * @param {Object} options - Additional options (paperSpeed, gridSize, etc.)
 * @returns {Promise<Object>} Analysis result from backend
 * @throws {ApiError} If upload or analysis fails
 */
export const uploadECG = async (imageUri, calibrationPoints = null, options = {}) => {
  const url = `${getApiUrl()}/predict`;
  
  logger.info('Uploading ECG for analysis', {
    url,
    hasCalibration: !!calibrationPoints,
    options,
  });
  
  try {
    // Create FormData
    const formData = new FormData();
    
    // Add image file
    const imageFilename = imageUri.split('/').pop();
    const imageType = imageFilename.split('.').pop();
    
    formData.append('image', {
      uri: imageUri,
      type: `image/${imageType}`,
      name: imageFilename,
    });
    
    // Add calibration points if provided
    if (calibrationPoints && calibrationPoints.length >= 2) {
      const formattedCalibration = formatCalibrationPoints(calibrationPoints);
      formData.append('calibration_points', JSON.stringify(formattedCalibration));
      logger.debug('Added calibration points', formattedCalibration);
    }
    
    // Add additional options if provided
    if (options.paperSpeed) {
      formData.append('paper_speed', options.paperSpeed.toString());
    }
    if (options.gridSize) {
      formData.append('grid_size', options.gridSize.toString());
    }
    
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
    
    // Send request
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        // Note: Don't set Content-Type header - let the browser set it with boundary
      },
      body: formData,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    // Handle response
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || 'Analysis failed';
      
      logger.error('API request failed', {
        status: response.status,
        message: errorMessage,
      });
      
      throw new ApiError(errorMessage, response.status);
    }
    
    // Parse successful response
    const result = await response.json();
    
    logger.info('ECG analysis successful', {
      diagnosis: result.diagnosis,
      bpm: result.bpm,
      confidence: result.confidence,
    });
    
    return result;
  } catch (error) {
    if (error.name === 'AbortError') {
      logger.error('ECG upload timeout', { url });
      throw new ApiError(
        'Analysis is taking too long. Please try again with a clearer image.',
        0,
        error
      );
    }
    
    if (error instanceof ApiError) {
      throw error;
    }
    
    logger.error('ECG upload failed', { error: error.message });
    throw new ApiError(
      'Failed to analyze ECG. Please check your connection and try again.',
      0,
      error
    );
  }
};

/**
 * Handle API Error
 * Converts technical API errors to user-friendly messages
 * 
 * @param {Error} error - The error object
 * @returns {string} User-friendly error message
 */
export const handleApiError = (error) => {
  logger.error('Handling API error', { error });
  
  if (error instanceof ApiError) {
    // Network errors
    if (error.statusCode === 0) {
      if (error.message.includes('not responding')) {
        return 'Unable to connect to diagnostic server. Please check that the backend is running.';
      }
      if (error.message.includes('timeout') || error.message.includes('taking too long')) {
        return 'Analysis is taking too long. Please try again with a clearer image.';
      }
      return 'Unable to connect to diagnostic server. Please check your internet connection.';
    }
    
    // Client errors (4xx)
    if (error.statusCode >= 400 && error.statusCode < 500) {
      if (error.statusCode === 400) {
        return 'Invalid image or calibration data. Please try again with a clearer ECG image.';
      }
      if (error.statusCode === 422) {
        return 'Image quality too low. Please retake the photo with better lighting and focus.';
      }
      return error.message || 'Invalid request. Please check your input and try again.';
    }
    
    // Server errors (5xx)
    if (error.statusCode >= 500) {
      return 'Diagnostic service is temporarily unavailable. Please try again later.';
    }
    
    return error.message;
  }
  
  // Generic errors
  if (error.message) {
    return error.message;
  }
  
  return 'An unexpected error occurred. Please try again.';
};

/**
 * Validate Image File
 * Checks if the image meets requirements before upload
 * 
 * @param {string} imageUri - Local URI of the image
 * @returns {Promise<boolean>} True if valid, throws error if not
 */
export const validateImage = async (imageUri) => {
  // Basic URI validation
  if (!imageUri) {
    throw new Error('No image selected');
  }
  
  // Check file extension
  const extension = imageUri.split('.').pop().toLowerCase();
  const validExtensions = ['jpg', 'jpeg', 'png'];
  
  if (!validExtensions.includes(extension)) {
    throw new Error('Invalid image format. Please use JPG or PNG.');
  }
  
  logger.debug('Image validation passed', { imageUri, extension });
  return true;
};

/**
 * Test API Connection
 * Helper function to test the API connection with detailed diagnostics
 * 
 * @returns {Promise<Object>} Connection test results
 */
export const testConnection = async () => {
  const results = {
    success: false,
    apiUrl: getApiUrl(),
    timestamp: new Date().toISOString(),
    errors: [],
  };
  
  try {
    const health = await checkBackendHealth();
    results.success = true;
    results.healthData = health;
    results.message = 'Successfully connected to backend';
  } catch (error) {
    results.success = false;
    results.errors.push(error.message);
    results.message = handleApiError(error);
  }
  
  return results;
};

export default {
  uploadECG,
  checkBackendHealth,
  formatCalibrationPoints,
  handleApiError,
  validateImage,
  testConnection,
  ApiError,
};
