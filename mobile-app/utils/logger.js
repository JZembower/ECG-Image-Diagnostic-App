/**
 * Logger Utility
 * 
 * Provides consistent logging throughout the application with different log levels.
 * Can be configured to log to console, remote logging service, or file.
 */

import { LOG_LEVEL, ENABLE_LOGGING } from '../config/config';

// Log levels with priorities
const LOG_LEVELS = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  none: 4,
};

// Get current log level priority
const currentLogLevel = LOG_LEVELS[LOG_LEVEL] || LOG_LEVELS.info;

/**
 * Format log message with timestamp and context
 * 
 * @param {string} level - Log level
 * @param {string} message - Log message
 * @param {Object} context - Additional context data
 * @returns {Object} Formatted log entry
 */
const formatLog = (level, message, context = {}) => {
  return {
    timestamp: new Date().toISOString(),
    level,
    message,
    context,
  };
};

/**
 * Should log based on current log level
 * 
 * @param {string} level - Level to check
 * @returns {boolean} Whether to log
 */
const shouldLog = (level) => {
  if (!ENABLE_LOGGING) return false;
  const levelPriority = LOG_LEVELS[level] || LOG_LEVELS.info;
  return levelPriority >= currentLogLevel;
};

/**
 * Log to console with appropriate styling
 * 
 * @param {string} level - Log level
 * @param {Object} logEntry - Formatted log entry
 */
const logToConsole = (level, logEntry) => {
  const prefix = `[${logEntry.timestamp}] [${level.toUpperCase()}]`;
  const message = `${prefix} ${logEntry.message}`;
  
  switch (level) {
    case 'debug':
      console.log(message, logEntry.context);
      break;
    case 'info':
      console.info(message, logEntry.context);
      break;
    case 'warn':
      console.warn(message, logEntry.context);
      break;
    case 'error':
      console.error(message, logEntry.context);
      break;
    default:
      console.log(message, logEntry.context);
  }
};

/**
 * Debug level logging
 * For detailed debugging information
 * 
 * @param {string} message - Log message
 * @param {Object} context - Additional context
 */
export const debug = (message, context = {}) => {
  if (!shouldLog('debug')) return;
  
  const logEntry = formatLog('debug', message, context);
  logToConsole('debug', logEntry);
};

/**
 * Info level logging
 * For general information about app operations
 * 
 * @param {string} message - Log message
 * @param {Object} context - Additional context
 */
export const info = (message, context = {}) => {
  if (!shouldLog('info')) return;
  
  const logEntry = formatLog('info', message, context);
  logToConsole('info', logEntry);
};

/**
 * Warning level logging
 * For potentially harmful situations
 * 
 * @param {string} message - Log message
 * @param {Object} context - Additional context
 */
export const warn = (message, context = {}) => {
  if (!shouldLog('warn')) return;
  
  const logEntry = formatLog('warn', message, context);
  logToConsole('warn', logEntry);
};

/**
 * Error level logging
 * For error events that might still allow the app to continue
 * 
 * @param {string} message - Log message
 * @param {Object} context - Additional context (should include error object)
 */
export const error = (message, context = {}) => {
  if (!shouldLog('error')) return;
  
  const logEntry = formatLog('error', message, context);
  logToConsole('error', logEntry);
  
  // In production, you might want to send errors to a remote logging service
  // Example: sendToRemoteLogger(logEntry);
};

/**
 * Log user action
 * Special logging for user interactions
 * 
 * @param {string} action - Action name
 * @param {Object} details - Action details
 */
export const logUserAction = (action, details = {}) => {
  info(`User action: ${action}`, { action, ...details });
};

/**
 * Log API call
 * Special logging for API requests
 * 
 * @param {string} endpoint - API endpoint
 * @param {string} method - HTTP method
 * @param {Object} details - Additional details
 */
export const logApiCall = (endpoint, method, details = {}) => {
  info(`API call: ${method} ${endpoint}`, { endpoint, method, ...details });
};

/**
 * Log performance metric
 * For tracking performance measurements
 * 
 * @param {string} metric - Metric name
 * @param {number} value - Metric value
 * @param {string} unit - Unit of measurement
 */
export const logPerformance = (metric, value, unit = 'ms') => {
  debug(`Performance: ${metric} = ${value}${unit}`, { metric, value, unit });
};

/**
 * Create a logger instance with a prefix
 * Useful for component-specific logging
 * 
 * @param {string} prefix - Prefix for all log messages
 * @returns {Object} Logger instance with prefixed methods
 */
export const createLogger = (prefix) => ({
  debug: (message, context) => debug(`[${prefix}] ${message}`, context),
  info: (message, context) => info(`[${prefix}] ${message}`, context),
  warn: (message, context) => warn(`[${prefix}] ${message}`, context),
  error: (message, context) => error(`[${prefix}] ${message}`, context),
  logUserAction: (action, details) => logUserAction(`[${prefix}] ${action}`, details),
});

// Export default logger object
export default {
  debug,
  info,
  warn,
  error,
  logUserAction,
  logApiCall,
  logPerformance,
  createLogger,
};
