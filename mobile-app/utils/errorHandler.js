/**
 * Error Handler Utility
 * 
 * Provides user-friendly error messages and error handling helpers
 * for various error scenarios in the application.
 */

import logger from './logger';

/**
 * Error categories for better error handling
 */
export const ERROR_CATEGORIES = {
  NETWORK: 'network',
  API: 'api',
  VALIDATION: 'validation',
  PERMISSION: 'permission',
  IMAGE: 'image',
  ANALYSIS: 'analysis',
  UNKNOWN: 'unknown',
};

/**
 * Get error category from error object
 * 
 * @param {Error} error - Error object
 * @returns {string} Error category
 */
const getErrorCategory = (error) => {
  if (!error) return ERROR_CATEGORIES.UNKNOWN;
  
  const message = error.message?.toLowerCase() || '';
  
  if (message.includes('network') || message.includes('connection') || message.includes('fetch')) {
    return ERROR_CATEGORIES.NETWORK;
  }
  if (message.includes('api') || message.includes('server') || message.includes('backend')) {
    return ERROR_CATEGORIES.API;
  }
  if (message.includes('invalid') || message.includes('validation')) {
    return ERROR_CATEGORIES.VALIDATION;
  }
  if (message.includes('permission') || message.includes('denied')) {
    return ERROR_CATEGORIES.PERMISSION;
  }
  if (message.includes('image') || message.includes('photo')) {
    return ERROR_CATEGORIES.IMAGE;
  }
  if (message.includes('analysis') || message.includes('diagnosis')) {
    return ERROR_CATEGORIES.ANALYSIS;
  }
  
  return ERROR_CATEGORIES.UNKNOWN;
};

/**
 * Get user-friendly error message
 * Converts technical error messages to user-friendly ones
 * 
 * @param {Error} error - Error object
 * @param {string} context - Context where error occurred
 * @returns {string} User-friendly error message
 */
export const getUserFriendlyMessage = (error, context = '') => {
  if (!error) return 'An unexpected error occurred';
  
  const category = getErrorCategory(error);
  const originalMessage = error.message || 'Unknown error';
  
  logger.error('Processing error for user display', {
    category,
    originalMessage,
    context,
  });
  
  // Return specific messages based on category
  switch (category) {
    case ERROR_CATEGORIES.NETWORK:
      return 'Unable to connect to the diagnostic server. Please check your internet connection and try again.';
      
    case ERROR_CATEGORIES.API:
      if (originalMessage.includes('timeout') || originalMessage.includes('taking too long')) {
        return 'The analysis is taking longer than expected. Please try again with a clearer image.';
      }
      if (originalMessage.includes('unavailable')) {
        return 'The diagnostic service is temporarily unavailable. Please try again in a few moments.';
      }
      return 'There was a problem communicating with the diagnostic server. Please try again.';
      
    case ERROR_CATEGORIES.VALIDATION:
      if (originalMessage.includes('calibration')) {
        return 'Invalid calibration points. Please recalibrate by tapping two opposite corners of a grid square.';
      }
      if (originalMessage.includes('peaks')) {
        return 'Not enough R-peaks detected. Please mark at least 2 R-peaks on the ECG waveform.';
      }
      return originalMessage;
      
    case ERROR_CATEGORIES.PERMISSION:
      if (originalMessage.includes('camera')) {
        return 'Camera permission is required to take photos. Please enable camera access in your device settings.';
      }
      if (originalMessage.includes('library') || originalMessage.includes('photos')) {
        return 'Photo library permission is required. Please enable photo access in your device settings.';
      }
      return 'Permission denied. Please check your app permissions in device settings.';
      
    case ERROR_CATEGORIES.IMAGE:
      if (originalMessage.includes('quality')) {
        return 'Image quality is too low for accurate analysis. Please retake the photo with better lighting and focus.';
      }
      if (originalMessage.includes('format')) {
        return 'Invalid image format. Please use JPG or PNG images.';
      }
      if (originalMessage.includes('size')) {
        return 'Image file is too large. Please compress the image or take a new photo.';
      }
      return 'There was a problem with the image. Please try taking a new photo.';
      
    case ERROR_CATEGORIES.ANALYSIS:
      if (originalMessage.includes('signal')) {
        return 'Unable to extract ECG signal from the image. Please ensure the ECG waveform is clearly visible.';
      }
      return 'Analysis failed. Please try again with a clearer ECG image.';
      
    default:
      // For unknown errors, try to return the original message if it's user-friendly
      if (originalMessage.length < 100 && !originalMessage.includes('Error:')) {
        return originalMessage;
      }
      return `An unexpected error occurred${context ? ` while ${context}` : ''}. Please try again.`;
  }
};

/**
 * Get action suggestions for error recovery
 * Provides user with actionable steps to resolve the error
 * 
 * @param {Error} error - Error object
 * @returns {Array<string>} List of suggested actions
 */
export const getErrorSuggestions = (error) => {
  const category = getErrorCategory(error);
  
  switch (category) {
    case ERROR_CATEGORIES.NETWORK:
      return [
        'Check your internet connection',
        'Ensure the backend server is running',
        'Try again in a few moments',
      ];
      
    case ERROR_CATEGORIES.API:
      return [
        'Verify the backend server is running',
        'Check the API URL configuration',
        'Try again later',
      ];
      
    case ERROR_CATEGORIES.VALIDATION:
      return [
        'Review the calibration instructions',
        'Ensure you have marked enough R-peaks',
        'Clear and retry the calibration',
      ];
      
    case ERROR_CATEGORIES.PERMISSION:
      return [
        'Open device Settings',
        'Find this app in Settings',
        'Enable the required permissions',
        'Restart the app',
      ];
      
    case ERROR_CATEGORIES.IMAGE:
      return [
        'Retake the photo with better lighting',
        'Ensure the ECG is in focus',
        'Use a higher resolution camera',
        'Avoid glare and shadows',
      ];
      
    case ERROR_CATEGORIES.ANALYSIS:
      return [
        'Ensure the ECG waveform is clearly visible',
        'Check that grid lines are visible',
        'Try with a different ECG image',
        'Improve image quality',
      ];
      
    default:
      return [
        'Try again',
        'Restart the app if problem persists',
        'Contact support if error continues',
      ];
  }
};

/**
 * Create error alert data
 * Prepares error information for display in an alert/modal
 * 
 * @param {Error} error - Error object
 * @param {string} context - Context where error occurred
 * @returns {Object} Error alert data with title, message, and actions
 */
export const createErrorAlert = (error, context = '') => {
  const category = getErrorCategory(error);
  const message = getUserFriendlyMessage(error, context);
  const suggestions = getErrorSuggestions(error);
  
  let title = 'Error';
  
  switch (category) {
    case ERROR_CATEGORIES.NETWORK:
      title = 'Connection Error';
      break;
    case ERROR_CATEGORIES.API:
      title = 'Service Error';
      break;
    case ERROR_CATEGORIES.VALIDATION:
      title = 'Validation Error';
      break;
    case ERROR_CATEGORIES.PERMISSION:
      title = 'Permission Required';
      break;
    case ERROR_CATEGORIES.IMAGE:
      title = 'Image Error';
      break;
    case ERROR_CATEGORIES.ANALYSIS:
      title = 'Analysis Error';
      break;
    default:
      title = 'Unexpected Error';
  }
  
  return {
    title,
    message,
    suggestions,
    category,
    canRetry: [ERROR_CATEGORIES.NETWORK, ERROR_CATEGORIES.API, ERROR_CATEGORIES.ANALYSIS].includes(category),
    needsPermission: category === ERROR_CATEGORIES.PERMISSION,
  };
};

/**
 * Handle and log error
 * Centralized error handling function
 * 
 * @param {Error} error - Error object
 * @param {string} context - Context where error occurred
 * @param {Function} callback - Optional callback with error alert data
 */
export const handleError = (error, context = '', callback = null) => {
  logger.error(`Error in ${context || 'application'}`, {
    error: error.message,
    stack: error.stack,
  });
  
  const alertData = createErrorAlert(error, context);
  
  if (callback) {
    callback(alertData);
  }
  
  return alertData;
};

/**
 * Retry helper with exponential backoff
 * 
 * @param {Function} fn - Async function to retry
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} delay - Initial delay in milliseconds
 * @returns {Promise<any>} Result of the function
 */
export const retryWithBackoff = async (fn, maxRetries = 3, delay = 1000) => {
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (attempt < maxRetries) {
        const backoffDelay = delay * Math.pow(2, attempt);
        logger.warn(`Retry attempt ${attempt + 1}/${maxRetries} after ${backoffDelay}ms`, {
          error: error.message,
        });
        await new Promise(resolve => setTimeout(resolve, backoffDelay));
      }
    }
  }
  
  throw lastError;
};

export default {
  ERROR_CATEGORIES,
  getUserFriendlyMessage,
  getErrorSuggestions,
  createErrorAlert,
  handleError,
  retryWithBackoff,
};
