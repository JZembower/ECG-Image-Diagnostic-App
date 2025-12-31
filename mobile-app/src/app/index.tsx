/**
 * ECG Analyzer - Main Screen
 * 
 * Enhanced React Native screen with full backend integration,
 * professional UI/UX, and comprehensive error handling.
 */

import * as ImagePicker from 'expo-image-picker';
import React, { useState, useEffect } from 'react';
import {
  Alert,
  Dimensions,
  Image,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

// Import services and utilities
import { uploadECG, checkBackendHealth, validateImage } from '../../services/api';
import { handleError } from '../../utils/errorHandler';
import logger from '../../utils/logger';
import LoadingIndicator, { ProgressLoadingIndicator } from '../../components/LoadingIndicator';
import { ECG_CONSTANTS } from '../../config/config';

const SCREEN_WIDTH = Dimensions.get('window').width;
const IMAGE_HEIGHT = 250;

interface Point {
  x: number;
  y: number;
}

interface AnalysisResult {
  diagnosis: string;
  bpm: number;
  confidence: number;
  signal_quality: string;
  all_probabilities?: Record<string, number>;
}

export default function ECGAnalyzer() {
  // Image and calibration state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([]);
  const [mmPerPixel, setMmPerPixel] = useState<number | null>(null);
  const [gridSizeMm, setGridSizeMm] = useState(ECG_CONSTANTS.DEFAULT_GRID_SIZE);
  const [paperSpeed, setPaperSpeed] = useState(ECG_CONSTANTS.DEFAULT_PAPER_SPEED);
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  
  // Backend health check
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);
  
  // Error state
  const [error, setError] = useState<string | null>(null);
  
  const [imageLayout, setImageLayout] = useState({ width: 0, height: 0 });

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealthStatus();
  }, []);

  /**
   * Check Backend Health Status
   */
  const checkBackendHealthStatus = async () => {
    setCheckingHealth(true);
    logger.info('Checking backend health status');
    
    try {
      await checkBackendHealth();
      setBackendHealthy(true);
      logger.info('Backend is healthy');
    } catch (err: any) {
      setBackendHealthy(false);
      logger.error('Backend health check failed', { error: err.message });
      
      // Show warning but don't block the app
      Alert.alert(
        'Backend Connection',
        'Unable to connect to the diagnostic server. Please ensure the backend is running at the configured URL.',
        [{ text: 'OK' }]
      );
    } finally {
      setCheckingHealth(false);
    }
  };

  /**
   * Handle Camera Capture
   */
  const handleCameraCapture = async () => {
    logger.logUserAction('camera_capture_initiated');
    
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    
    if (permissionResult.granted === false) {
      const errorAlert = handleError(
        new Error('Camera permission denied'),
        'requesting camera permission'
      );
      Alert.alert(errorAlert.title, errorAlert.message);
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
    });

    if (!result.canceled) {
      const imageUri = result.assets[0].uri;
      logger.info('Image captured', { uri: imageUri });
      setSelectedImage(imageUri);
      resetAnalysis();
    }
  };

  /**
   * Handle Image Picker
   */
  const handleImagePicker = async () => {
    logger.logUserAction('image_picker_initiated');
    
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
    });

    if (!result.canceled) {
      const imageUri = result.assets[0].uri;
      logger.info('Image selected from gallery', { uri: imageUri });
      setSelectedImage(imageUri);
      resetAnalysis();
    }
  };

  /**
   * Reset Analysis State
   */
  const resetAnalysis = () => {
    setCalibrationPoints([]);
    setMmPerPixel(null);
    setAnalysisResult(null);
    setError(null);
    setAnalysisProgress(0);
  };

  /**
   * Clear All Data
   */
  const clearAll = () => {
    logger.logUserAction('clear_all');
    setSelectedImage(null);
    resetAnalysis();
  };

  /**
   * Convert touch coordinates to image coordinates
   */
  const viewToImageCoords = (vx: number, vy: number): Point => {
    if (!imageLayout.width || !imageLayout.height) return { x: vx, y: vy };
    return { x: vx, y: vy };
  };

  /**
   * Handle Image Press for Calibration
   */
  const handleImagePress = (event: any) => {
    if (!selectedImage) return;
    
    const { locationX, locationY } = event.nativeEvent;
    const pt = viewToImageCoords(locationX, locationY);

    // Collect 2 calibration taps (opposite corners of ONE grid square)
    if (calibrationPoints.length < 2) {
      const next = [...calibrationPoints, pt];
      setCalibrationPoints(next);
      logger.debug('Calibration point added', { point: pt, total: next.length });

      if (next.length === 2) {
        const dx = next[1].x - next[0].x;
        const dy = next[1].y - next[0].y;
        const pixelDist = Math.sqrt(dx * dx + dy * dy);
        const mmPerPx = gridSizeMm / pixelDist;
        setMmPerPixel(mmPerPx);
        
        logger.info('Grid calibrated', { mmPerPixel: mmPerPx, gridSizeMm });
        
        Alert.alert(
          '✅ Grid Calibrated!',
          `Calibration complete for ${gridSizeMm}mm grid square.\\n\\nScale: ${mmPerPx.toFixed(4)} mm/pixel\\n\\nYou can now analyze the ECG.`
        );
      } else {
        Alert.alert(
          'Grid Calibration - Step 1',
          `First corner marked!\\n\\nNow tap the OPPOSITE corner of the same ${gridSizeMm}mm grid square.`
        );
      }
    }
  };

  /**
   * Clear Calibration Only
   */
  const clearCalibration = () => {
    logger.logUserAction('clear_calibration');
    setCalibrationPoints([]);
    setMmPerPixel(null);
    Alert.alert('Calibration Cleared', 'You can now recalibrate the grid.');
  };

  /**
   * Analyze ECG - Main Function
   */
  const analyzeECG = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or capture an ECG image first.');
      return;
    }

    logger.logUserAction('analyze_ecg_initiated', {
      hasCalibration: !!mmPerPixel,
      gridSizeMm,
      paperSpeed,
    });

    setError(null);
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    try {
      // Step 1: Validate image
      setAnalysisProgress(1);
      await validateImage(selectedImage);
      logger.debug('Image validation passed');

      // Step 2: Upload and analyze
      setAnalysisProgress(2);
      
      const options = {
        paperSpeed,
        gridSize: gridSizeMm,
      };

      logger.info('Uploading ECG to backend', { options });
      
      const result = await uploadECG(
        selectedImage,
        calibrationPoints.length === 2 ? calibrationPoints : null,
        options
      );

      // Step 3: Process results
      setAnalysisProgress(3);
      logger.info('Analysis complete', { result });
      
      setAnalysisResult(result);
      
      Alert.alert(
        '✅ Analysis Complete',
        `Diagnosis: ${result.diagnosis}\\nBPM: ${result.bpm}\\nConfidence: ${(result.confidence * 100).toFixed(1)}%`
      );

    } catch (err: any) {
      logger.error('ECG analysis failed', { error: err.message });
      
      const errorAlert = handleError(err, 'analyzing ECG');
      setError(errorAlert.message);
      
      Alert.alert(
        errorAlert.title,
        errorAlert.message,
        [
          {
            text: 'Retry',
            onPress: () => analyzeECG(),
            style: 'default',
          },
          {
            text: 'Cancel',
            style: 'cancel',
          },
        ]
      );
    } finally {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
    }
  };

  /**
   * Get confidence color based on confidence level
   */
  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return '#10B981'; // Green
    if (confidence >= 0.7) return '#F59E0B'; // Amber
    return '#EF4444'; // Red
  };

  /**
   * Get BPM classification
   */
  const classifyBPM = (bpm: number) => {
    if (bpm < ECG_CONSTANTS.NORMAL_BPM_RANGE.min) {
      return { label: 'Bradycardia', color: '#F59E0B' };
    }
    if (bpm > ECG_CONSTANTS.NORMAL_BPM_RANGE.max) {
      return { label: 'Tachycardia', color: '#EF4444' };
    }
    return { label: 'Normal', color: '#10B981' };
  };

  /**
   * Render Calibration Overlay
   */
  const renderCalibrationOverlay = () => {
    if (!selectedImage) return null;

    return (
      <Svg
        style={StyleSheet.absoluteFill}
        width={imageLayout.width}
        height={IMAGE_HEIGHT}
        pointerEvents="none"
      >
        {/* Calibration points & line */}
        {calibrationPoints.map((p, i) => (
          <Circle
            key={`cal-${i}`}
            cx={p.x}
            cy={p.y}
            r={8}
            fill="#3B82F6"
            stroke="#FFFFFF"
            strokeWidth="2"
          />
        ))}
        {calibrationPoints.length === 2 && (
          <Line
            x1={calibrationPoints[0].x}
            y1={calibrationPoints[0].y}
            x2={calibrationPoints[1].x}
            y2={calibrationPoints[1].y}
            stroke="#3B82F6"
            strokeWidth={3}
          />
        )}
      </Svg>
    );
  };

  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" translucent={false} />
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.iconContainer}>
            <Text style={styles.heartIcon}>❤️</Text>
          </View>
          <Text style={styles.title}>ECG Analyzer</Text>
          <Text style={styles.subtitle}>
            AI-powered ECG diagnosis with real-time backend integration
          </Text>
          
          {/* Backend Status Indicator */}
          {!checkingHealth && (
            <View style={styles.backendStatus}>
              <View style={[
                styles.statusDot,
                { backgroundColor: backendHealthy ? '#10B981' : '#EF4444' }
              ]} />
              <Text style={styles.backendStatusText}>
                {backendHealthy ? 'Backend Connected' : 'Backend Offline'}
              </Text>
              <TouchableOpacity onPress={checkBackendHealthStatus}>
                <Text style={styles.refreshIcon}>🔄</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* Main Card */}
        <View style={styles.card}>
          {/* Action Buttons */}
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.primaryBtn} onPress={handleCameraCapture}>
              <Text style={styles.primaryBtnText}>📸 Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryBtn} onPress={handleImagePicker}>
              <Text style={styles.secondaryBtnText}>📁 Gallery</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.clearBtn} onPress={clearAll}>
              <Text style={styles.clearBtnText}>🗑️</Text>
            </TouchableOpacity>
          </View>

          {/* Configuration Controls */}
          <View style={styles.controlsSection}>
            <View style={styles.inputRow}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Grid Size (mm)</Text>
                <TextInput
                  style={styles.input}
                  keyboardType="numeric"
                  placeholder="5"
                  placeholderTextColor="#64748B"
                  value={String(gridSizeMm)}
                  onChangeText={(t) => setGridSizeMm(parseFloat(t) || ECG_CONSTANTS.DEFAULT_GRID_SIZE)}
                  editable={!isAnalyzing}
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Paper Speed (mm/s)</Text>
                <TextInput
                  style={styles.input}
                  keyboardType="numeric"
                  placeholder="25"
                  placeholderTextColor="#64748B"
                  value={String(paperSpeed)}
                  onChangeText={(t) => setPaperSpeed(parseFloat(t) || ECG_CONSTANTS.DEFAULT_PAPER_SPEED)}
                  editable={!isAnalyzing}
                />
              </View>
            </View>
            
            <TouchableOpacity
              style={[styles.analyzeBtn, !selectedImage && styles.analyzeBtnDisabled]}
              onPress={analyzeECG}
              disabled={!selectedImage || isAnalyzing}
            >
              <Text style={styles.analyzeBtnText}>
                {isAnalyzing ? '⏳ Analyzing...' : '🧠 Analyze ECG'}
              </Text>
            </TouchableOpacity>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} style={styles.scrollContent}>
            {/* Image Section */}
            <View
              style={styles.imageContainer}
              onLayout={(event) => {
                const { width, height } = event.nativeEvent.layout;
                setImageLayout({ width, height });
              }}
            >
              {selectedImage ? (
                <TouchableOpacity
                  activeOpacity={1}
                  onPress={handleImagePress}
                  style={styles.imageWrapper}
                  disabled={isAnalyzing || calibrationPoints.length >= 2}
                >
                  <Image
                    source={{ uri: selectedImage }}
                    style={styles.image}
                    resizeMode="contain"
                  />
                  {renderCalibrationOverlay()}
                </TouchableOpacity>
              ) : (
                <View style={styles.placeholder}>
                  <Text style={styles.placeholderIcon}>📷</Text>
                  <Text style={styles.placeholderTitle}>Ready to Analyze ECG</Text>
                  <Text style={styles.placeholderText}>
                    1. Take or select an ECG photo{'\n'}
                    2. Optionally calibrate by tapping grid corners{'\n'}
                    3. Tap Analyze for AI-powered diagnosis
                  </Text>
                </View>
              )}
            </View>

            {/* Calibration Status */}
            {selectedImage && (
              <View style={styles.statusSection}>
                <Text style={styles.sectionTitle}>Calibration Status</Text>
                
                <View style={styles.statusGrid}>
                  <View style={styles.statusItem}>
                    <Text style={styles.statusLabel}>Grid Calibration</Text>
                    <Text
                      style={[
                        styles.statusValue,
                        { color: mmPerPixel ? '#10B981' : '#94A3B8' }
                      ]}
                    >
                      {mmPerPixel ? '✅ Complete' : '⚪ Optional'}
                    </Text>
                  </View>
                  <View style={styles.statusItem}>
                    <Text style={styles.statusLabel}>Points Marked</Text>
                    <Text style={[styles.statusValue, { color: '#94A3B8' }]}>
                      {calibrationPoints.length}/2
                    </Text>
                  </View>
                </View>

                {mmPerPixel && (
                  <TouchableOpacity
                    style={styles.clearCalibrationBtn}
                    onPress={clearCalibration}
                    disabled={isAnalyzing}
                  >
                    <Text style={styles.clearCalibrationText}>🔄 Reset Calibration</Text>
                  </TouchableOpacity>
                )}
                
                {!mmPerPixel && calibrationPoints.length === 0 && (
                  <View style={styles.infoBox}>
                    <Text style={styles.infoText}>
                      💡 Tap opposite corners of a grid square to calibrate, or skip and analyze without calibration.
                    </Text>
                  </View>
                )}
              </View>
            )}

            {/* Error Display */}
            {error && (
              <View style={styles.errorSection}>
                <Text style={styles.errorIcon}>⚠️</Text>
                <Text style={styles.errorText}>{error}</Text>
                <TouchableOpacity style={styles.retryBtn} onPress={analyzeECG}>
                  <Text style={styles.retryBtnText}>Try Again</Text>
                </TouchableOpacity>
              </View>
            )}

            {/* Results Section */}
            {analysisResult && (
              <View style={styles.resultsSection}>
                <View style={styles.resultsHeader}>
                  <Text style={styles.checkIcon}>✅</Text>
                  <Text style={styles.resultsTitle}>Diagnosis Results</Text>
                </View>

                {/* Diagnosis Display */}
                <View style={styles.diagnosisSection}>
                  <Text style={styles.diagnosisLabel}>Diagnosis</Text>
                  <Text style={styles.diagnosisValue}>{analysisResult.diagnosis}</Text>
                </View>

                {/* Heart Rate Display */}
                <View style={styles.heartRateSection}>
                  <Text style={styles.heartRateLabel}>Heart Rate</Text>
                  <View style={styles.heartRateDisplay}>
                    <Text style={styles.heartRateValue}>{Math.round(analysisResult.bpm)}</Text>
                    <Text style={styles.heartRateUnit}>BPM</Text>
                  </View>
                  <View
                    style={[
                      styles.classificationBadge,
                      { backgroundColor: classifyBPM(analysisResult.bpm).color }
                    ]}
                  >
                    <Text style={styles.badgeText}>
                      {classifyBPM(analysisResult.bpm).label}
                    </Text>
                  </View>
                </View>

                {/* Confidence & Quality */}
                <View style={styles.metricsGrid}>
                  <View style={styles.metricItem}>
                    <Text style={styles.metricLabel}>Confidence</Text>
                    <Text
                      style={[
                        styles.metricValue,
                        { color: getConfidenceColor(analysisResult.confidence) }
                      ]}
                    >
                      {(analysisResult.confidence * 100).toFixed(1)}%
                    </Text>
                  </View>
                  <View style={styles.metricItem}>
                    <Text style={styles.metricLabel}>Signal Quality</Text>
                    <Text style={styles.metricValue}>{analysisResult.signal_quality}</Text>
                  </View>
                </View>

                {/* Calibration Info */}
                {mmPerPixel && (
                  <View style={styles.technicalSection}>
                    <Text style={styles.sectionTitle}>Calibration Details</Text>
                    <Text style={styles.technicalText}>
                      Grid: {gridSizeMm}mm | Scale: {mmPerPixel.toFixed(4)} mm/px
                    </Text>
                    <Text style={styles.technicalText}>Paper Speed: {paperSpeed} mm/s</Text>
                  </View>
                )}

                {/* Disclaimer */}
                <View style={styles.disclaimer}>
                  <Text style={styles.disclaimerText}>
                    ⚠️ This analysis is for educational purposes only. Always consult a
                    healthcare professional for medical advice.
                  </Text>
                </View>
              </View>
            )}
          </ScrollView>
        </View>
      </ScrollView>

      {/* Loading Indicator */}
      <ProgressLoadingIndicator
        visible={isAnalyzing}
        currentStep={analysisProgress}
        totalSteps={3}
        steps={[
          'Validating image...',
          'Uploading to backend...',
          'Running AI diagnosis...',
        ]}
      />
    </>
  );
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  header: {
    alignItems: 'center',
    paddingTop: 20,
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  iconContainer: {
    backgroundColor: 'rgba(30, 41, 59, 0.8)',
    padding: 20,
    borderRadius: 50,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: 'rgba(239, 68, 68, 0.3)',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 10,
  },
  heartIcon: {
    fontSize: 36,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 6,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 12,
  },
  backendStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(30, 41, 59, 0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginTop: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  backendStatusText: {
    fontSize: 12,
    color: '#CBD5E1',
    marginRight: 6,
  },
  refreshIcon: {
    fontSize: 14,
  },
  card: {
    flex: 1,
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 24,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.2)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 12,
  },
  buttonRow: {
    flexDirection: 'row',
    padding: 16,
    paddingBottom: 0,
    gap: 10,
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: '#EF4444',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  primaryBtnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 14,
  },
  secondaryBtn: {
    flex: 1,
    backgroundColor: '#475569',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#64748B',
  },
  secondaryBtnText: {
    color: '#E2E8F0',
    fontWeight: '600',
    fontSize: 14,
  },
  clearBtn: {
    backgroundColor: 'rgba(71, 85, 105, 0.5)',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#475569',
  },
  clearBtnText: {
    fontSize: 16,
  },
  controlsSection: {
    padding: 16,
    paddingTop: 12,
  },
  inputRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 12,
  },
  inputGroup: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  inputLabel: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '500',
    marginBottom: 4,
  },
  input: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '600',
  },
  analyzeBtn: {
    backgroundColor: '#10B981',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#10B981',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  analyzeBtnDisabled: {
    backgroundColor: '#475569',
    shadowOpacity: 0,
  },
  analyzeBtnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 15,
  },
  scrollContent: {
    flex: 1,
  },
  imageContainer: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 14,
    overflow: 'hidden',
  },
  imageWrapper: {
    position: 'relative',
  },
  image: {
    width: '100%',
    height: IMAGE_HEIGHT,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#475569',
  },
  placeholder: {
    minHeight: IMAGE_HEIGHT,
    borderRadius: 14,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#475569',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
  },
  placeholderIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  placeholderTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#E2E8F0',
    marginBottom: 8,
  },
  placeholderText: {
    color: '#94A3B8',
    textAlign: 'center',
    lineHeight: 20,
    fontSize: 13,
  },
  statusSection: {
    margin: 16,
    marginTop: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#E2E8F0',
    marginBottom: 10,
  },
  statusGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  statusItem: {
    flex: 1,
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  statusLabel: {
    fontSize: 11,
    color: '#94A3B8',
    marginBottom: 4,
  },
  statusValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  clearCalibrationBtn: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.3)',
  },
  clearCalibrationText: {
    color: '#FCD34D',
    fontWeight: '600',
    fontSize: 12,
  },
  infoBox: {
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    padding: 12,
    borderRadius: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#3B82F6',
  },
  infoText: {
    color: '#93C5FD',
    fontSize: 12,
    lineHeight: 18,
  },
  errorSection: {
    margin: 16,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 14,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  errorText: {
    color: '#FECACA',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 16,
  },
  retryBtn: {
    backgroundColor: '#EF4444',
    paddingVertical: 10,
    paddingHorizontal: 24,
    borderRadius: 10,
  },
  retryBtnText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 14,
  },
  resultsSection: {
    margin: 16,
    marginTop: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  checkIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  diagnosisSection: {
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  diagnosisLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 6,
  },
  diagnosisValue: {
    fontSize: 22,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  heartRateSection: {
    alignItems: 'center',
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  heartRateLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 6,
  },
  heartRateDisplay: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 10,
  },
  heartRateValue: {
    fontSize: 40,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  heartRateUnit: {
    fontSize: 14,
    color: '#94A3B8',
    marginLeft: 6,
  },
  classificationBadge: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
  },
  badgeText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 12,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  metricItem: {
    flex: 1,
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  metricLabel: {
    fontSize: 11,
    color: '#94A3B8',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  technicalSection: {
    marginBottom: 12,
  },
  technicalText: {
    color: '#CBD5E1',
    fontSize: 12,
    marginBottom: 3,
  },
  disclaimer: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    padding: 10,
    borderRadius: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#F59E0B',
  },
  disclaimerText: {
    fontSize: 11,
    color: '#FCD34D',
    lineHeight: 16,
  },
});
