/**
 * Loading Indicator Component
 * 
 * Professional loading indicator with customizable messages and styles.
 * Used during API calls and long-running operations.
 */

import React from 'react';
import {
  View,
  Text,
  ActivityIndicator,
  StyleSheet,
  Modal,
  TouchableOpacity,
} from 'react-native';

const LoadingIndicator = ({
  visible = false,
  message = 'Loading...',
  submessage = '',
  showCancel = false,
  onCancel = null,
  transparent = false,
}) => {
  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="fade"
      statusBarTranslucent={true}
    >
      <View style={[styles.container, transparent && styles.transparentContainer]}>
        <View style={styles.content}>
          {/* Loading Spinner */}
          <ActivityIndicator size="large" color="#EF4444" style={styles.spinner} />
          
          {/* Main Message */}
          <Text style={styles.message}>{message}</Text>
          
          {/* Submessage */}
          {submessage ? (
            <Text style={styles.submessage}>{submessage}</Text>
          ) : null}
          
          {/* Cancel Button */}
          {showCancel && onCancel ? (
            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </Modal>
  );
};

/**
 * Inline Loading Indicator
 * Smaller loading indicator for inline use (not modal)
 */
export const InlineLoadingIndicator = ({
  message = 'Loading...',
  size = 'small',
  color = '#EF4444',
}) => {
  return (
    <View style={styles.inlineContainer}>
      <ActivityIndicator size={size} color={color} />
      {message ? (
        <Text style={styles.inlineMessage}>{message}</Text>
      ) : null}
    </View>
  );
};

/**
 * Progress Loading Indicator
 * Shows loading with progress steps
 */
export const ProgressLoadingIndicator = ({
  visible = false,
  currentStep = 0,
  totalSteps = 3,
  steps = ['Uploading image...', 'Extracting signal...', 'Running diagnosis...'],
  onCancel = null,
}) => {
  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="fade"
      statusBarTranslucent={true}
    >
      <View style={styles.container}>
        <View style={styles.content}>
          {/* Loading Spinner */}
          <ActivityIndicator size="large" color="#EF4444" style={styles.spinner} />
          
          {/* Progress Steps */}
          <View style={styles.progressContainer}>
            {steps.map((step, index) => (
              <View key={index} style={styles.progressStep}>
                <View style={[
                  styles.progressDot,
                  index < currentStep && styles.progressDotComplete,
                  index === currentStep && styles.progressDotActive,
                ]} />
                <Text style={[
                  styles.progressText,
                  index === currentStep && styles.progressTextActive,
                  index < currentStep && styles.progressTextComplete,
                ]}>
                  {step}
                </Text>
              </View>
            ))}
          </View>
          
          {/* Progress Percentage */}
          <Text style={styles.progressPercentage}>
            {Math.round((currentStep / totalSteps) * 100)}%
          </Text>
          
          {/* Cancel Button */}
          {onCancel ? (
            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  transparentContainer: {
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
  },
  content: {
    backgroundColor: 'rgba(30, 41, 59, 0.98)',
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
    minWidth: 280,
    maxWidth: 360,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.2)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 12,
  },
  spinner: {
    marginBottom: 20,
    transform: [{ scale: 1.2 }],
  },
  message: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  submessage: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    lineHeight: 20,
    marginTop: 4,
  },
  cancelButton: {
    marginTop: 20,
    paddingVertical: 10,
    paddingHorizontal: 24,
    backgroundColor: 'rgba(71, 85, 105, 0.5)',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#475569',
  },
  cancelButtonText: {
    color: '#E2E8F0',
    fontSize: 14,
    fontWeight: '600',
  },
  // Inline styles
  inlineContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    justifyContent: 'center',
  },
  inlineMessage: {
    marginLeft: 12,
    fontSize: 16,
    color: '#94A3B8',
  },
  // Progress styles
  progressContainer: {
    width: '100%',
    marginTop: 16,
  },
  progressStep: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  progressDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#475569',
    marginRight: 12,
  },
  progressDotActive: {
    backgroundColor: '#EF4444',
    transform: [{ scale: 1.2 }],
  },
  progressDotComplete: {
    backgroundColor: '#10B981',
  },
  progressText: {
    fontSize: 14,
    color: '#64748B',
    flex: 1,
  },
  progressTextActive: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  progressTextComplete: {
    color: '#94A3B8',
  },
  progressPercentage: {
    fontSize: 24,
    fontWeight: '700',
    color: '#EF4444',
    marginTop: 16,
  },
});

export default LoadingIndicator;
