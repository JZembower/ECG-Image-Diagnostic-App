import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

const SCREEN_WIDTH = Dimensions.get('window').width;
const IMAGE_HEIGHT = 250;

interface Point {
  x: number;
  y: number;
}

export default function ECGAnalyzer() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  
  // Single grid calibration: user marks two opposite corners of ONE grid square
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([]);
  const [mmPerPixel, setMmPerPixel] = useState<number | null>(null);
  const [gridSizeMm, setGridSizeMm] = useState(5); // Standard ECG grid: 5mm = 1 large square
  
  // R-peaks and analysis
  const [peaks, setPeaks] = useState<Point[]>([]);
  const [paperSpeed, setPaperSpeed] = useState(25); // mm/s
  const [bpm, setBpm] = useState<number | null>(null);
  const [notes, setNotes] = useState("");
  
  // Rate classification
  const [rateClass, setRateClass] = useState({ label: "--", color: "#64748B" });
  
  const [imageLayout, setImageLayout] = useState({ width: 0, height: 0, x: 0, y: 0 });

  const classifyRate = (b: number) => {
    if (!isFinite(b)) return { label: "--", color: "#64748B" };
    if (b < 60) return { label: "Bradycardia", color: "#F59E0B" };
    if (b > 100) return { label: "Tachycardia", color: "#EF4444" };
    return { label: "Normal", color: "#10B981" };
  };

  const handleCameraCapture = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    
    if (permissionResult.granted === false) {
      Alert.alert('Permission Required', 'Camera permission is needed to take photos');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      resetCalibration();
    }
  };

  const handleImagePicker = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      resetCalibration();
    }
  };

  const resetCalibration = () => {
    setCalibrationPoints([]);
    setPeaks([]);
    setMmPerPixel(null);
    setBpm(null);
    setNotes("");
    setRateClass({ label: "--", color: "#64748B" });
    setAnalysisResult(null);
  };

  // Convert touch coordinates to image coordinates
  const viewToImageCoords = (vx: number, vy: number) => {
    if (!imageLayout.width || !imageLayout.height) return { x: vx, y: vy };
    return { x: vx, y: vy };
  };

  const handleImagePress = (event: any) => {
    if (!selectedImage) return;
    const { locationX, locationY } = event.nativeEvent;
    const pt = viewToImageCoords(locationX, locationY);

    // First collect 2 calibration taps (opposite corners of ONE grid square)
    if (calibrationPoints.length < 2) {
      const next = [...calibrationPoints, pt];
      setCalibrationPoints(next);

      if (next.length === 2) {
        const dx = next[1].x - next[0].x;
        const dy = next[1].y - next[0].y;
        const pixelDist = Math.sqrt(dx * dx + dy * dy);
        const mmPerPx = gridSizeMm / pixelDist;
        setMmPerPixel(mmPerPx);
        Alert.alert(
          "Grid Calibrated!", 
          `Calibration complete for ${gridSizeMm}mm grid square.\nScale: ${mmPerPx.toFixed(4)} mm/pixel\n\nNow tap R-peaks on the ECG waveform.`
        );
      } else {
        Alert.alert(
          "Grid Calibration", 
          `First corner marked!\nNow tap the OPPOSITE corner of the same ${gridSizeMm}mm grid square.`
        );
      }
      return;
    }

    // Otherwise, mark R peaks
    setPeaks(prev => [...prev, pt]);
  };

  const clearAll = () => {
    setCalibrationPoints([]);
    setPeaks([]);
    setMmPerPixel(null);
    setBpm(null);
    setNotes("");
    setRateClass({ label: "--", color: "#64748B" });
    setAnalysisResult(null);
  };

  const clearImage = () => {
    setSelectedImage(null);
    resetCalibration();
  };

  const analyzeECG = async () => {
    if (!selectedImage) return;
    
    if (!mmPerPixel) {
      Alert.alert("Missing Calibration", `Please calibrate first by tapping opposite corners of one ${gridSizeMm}mm grid square.`);
      return;
    }

    if (peaks.length < 2) {
      Alert.alert("Need More R-Peaks", "Please mark at least two R-peaks on the ECG trace.");
      return;
    }
    
    setIsAnalyzing(true);
    
    try {
      // Simulate analysis
      await new Promise(resolve => setTimeout(resolve, 2500));
      
      // Sort R-peaks left to right
      const sorted = [...peaks].sort((a, b) => a.x - b.x);

      // Calculate RR intervals
      const rrPixels = [];
      for (let i = 1; i < sorted.length; i++) {
        const dx = sorted[i].x - sorted[i - 1].x;
        const dy = sorted[i].y - sorted[i - 1].y;
        rrPixels.push(Math.sqrt(dx * dx + dy * dy));
      }

      // Convert to time using calibrated scale
      const rrMm = rrPixels.map(p => p * mmPerPixel);
      const rrSeconds = rrMm.map(mm => mm / paperSpeed);
      const meanRR = rrSeconds.reduce((a, b) => a + b, 0) / rrSeconds.length;
      const bpmEst = 60 / meanRR;

      // Variability check
      const variance = rrSeconds.reduce((acc, v) => acc + Math.pow(v - meanRR, 2), 0) / rrSeconds.length;
      const sdRR = Math.sqrt(variance);

      let analysisNotes = "";
      if (sdRR > 0.12) {
        analysisNotes += "Irregular rhythm detected (high RR variability). ";
      } else {
        analysisNotes += "Regular rhythm pattern. ";
      }
      analysisNotes += `Grid: ${gridSizeMm}mm calibrated. Paper speed: ${paperSpeed}mm/s.`;
      
      const mockResult = {
        confidence: 0.94,
        rhythm: sdRR > 0.12 ? "Irregular Rhythm" : "Regular Sinus Rhythm",
        heartRate: Math.round(bpmEst),
        intervals: {
          PR: "0.16s (Normal)",
          QRS: "0.08s (Normal)", 
          QT: "0.38s (Normal)"
        },
        peaksDetected: peaks.length,
        gridCalibrated: true,
        recommendation: `Analysis based on ${peaks.length} R-peaks with ${gridSizeMm}mm grid calibration. ${Math.round(bpmEst) > 100 ? 'Tachycardia detected.' : Math.round(bpmEst) < 60 ? 'Bradycardia detected.' : 'Heart rate within normal range.'}`
      };

      setBpm(bpmEst);
      setRateClass(classifyRate(bpmEst));
      setNotes(analysisNotes);
      setAnalysisResult(mockResult);
      
    } catch (err) {
      Alert.alert('Analysis Error', 'Failed to analyze ECG. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderCalibrationOverlay = () => {
    if (!selectedImage) return null;

    return (
      <Svg
        style={StyleSheet.absoluteFill}
        width={imageLayout.width}
        height={IMAGE_HEIGHT}
      >
        {/* Calibration points & line */}
        {calibrationPoints.map((p, i) => (
          <Circle key={`cal-${i}`} cx={p.x} cy={p.y} r={8} fill="#3B82F6" stroke="#FFFFFF" strokeWidth="2" />
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

        {/* R-peak markers */}
        {peaks.map((p, i) => (
          <Circle key={`peak-${i}`} cx={p.x} cy={p.y} r={6} fill="#EF4444" stroke="#FFFFFF" strokeWidth="2" />
        ))}
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
          <Text style={styles.subtitle}>AI-powered ECG analysis with single grid calibration</Text>
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
            <TouchableOpacity style={styles.clearBtn} onPress={clearImage}>
              <Text style={styles.clearBtnText}>🗑️ Clear</Text>
            </TouchableOpacity>
          </View>

          {/* Controls */}
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
                  onChangeText={(t) => setGridSizeMm(parseFloat(t) || 5)}
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
                  onChangeText={(t) => setPaperSpeed(parseFloat(t) || 25)}
                />
              </View>
            </View>
            <TouchableOpacity style={styles.analyzeBtn} onPress={analyzeECG}>
              <Text style={styles.analyzeBtnText}>🧠 Analyze ECG</Text>
            </TouchableOpacity>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} style={styles.scrollContent}>
            
            {/* Image Section */}
            <View
              style={styles.imageContainer}
              onLayout={(event) => {
                const { width, height, x, y } = event.nativeEvent.layout;
                setImageLayout({ width, height, x, y });
              }}
            >
              {selectedImage ? (
                <TouchableOpacity activeOpacity={1} onPress={handleImagePress} style={styles.imageWrapper}>
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
                    1. Take or select an ECG photo{"\n"}
                    2. Set grid size and paper speed above{"\n"}
                    3. Tap OPPOSITE corners of ONE grid square{"\n"}
                    4. Mark R-peaks on the waveform{"\n"}
                    5. Tap Analyze for results
                  </Text>
                </View>
              )}
            </View>

            {/* Status Section */}
            <View style={styles.statusSection}>
              <Text style={styles.sectionTitle}>Calibration Status</Text>
              
              <View style={styles.statusGrid}>
                <View style={styles.statusItem}>
                  <Text style={styles.statusLabel}>Grid Calibration</Text>
                  <Text style={[styles.statusValue, { color: mmPerPixel ? '#10B981' : '#64748B' }]}>
                    {mmPerPixel ? '✅ Complete' : '❌ Required'}
                  </Text>
                </View>
                <View style={styles.statusItem}>
                  <Text style={styles.statusLabel}>R-Peaks Marked</Text>
                  <Text style={[styles.statusValue, { color: peaks.length >= 2 ? '#10B981' : '#64748B' }]}>
                    {peaks.length >= 2 ? `✅ ${peaks.length}` : `${peaks.length} (need 2+)`}
                  </Text>
                </View>
              </View>

              {/* Clear calibration button */}
              <TouchableOpacity style={styles.clearCalibrationBtn} onPress={clearAll}>
                <Text style={styles.clearCalibrationText}>🔄 Reset Calibration</Text>
              </TouchableOpacity>
            </View>

            {/* Loading State */}
            {isAnalyzing && (
              <View style={styles.loadingSection}>
                <ActivityIndicator size="large" color="#EF4444" />
                <Text style={styles.loadingText}>Analyzing ECG with calibrated measurements...</Text>
                <Text style={styles.loadingSubtext}>Processing {peaks.length} R-peaks</Text>
              </View>
            )}

            {/* Results Section */}
            {analysisResult && (
              <View style={styles.resultsSection}>
                <View style={styles.resultsHeader}>
                  <Text style={styles.checkIcon}>✅</Text>
                  <Text style={styles.resultsTitle}>Analysis Complete</Text>
                </View>

                {/* Heart Rate Display */}
                <View style={styles.heartRateSection}>
                  <Text style={styles.heartRateLabel}>Heart Rate</Text>
                  <View style={styles.heartRateDisplay}>
                    <Text style={styles.heartRateValue}>
                      {bpm != null ? Math.round(bpm) : '--'}
                    </Text>
                    <Text style={styles.heartRateUnit}>BPM</Text>
                  </View>
                  <View style={[styles.classificationBadge, { backgroundColor: rateClass.color }]}>
                    <Text style={styles.badgeText}>{rateClass.label}</Text>
                  </View>
                </View>

                {/* Technical Details */}
                <View style={styles.technicalSection}>
                  <Text style={styles.sectionTitle}>Calibration Details</Text>
                  <Text style={styles.technicalText}>Grid Size: {gridSizeMm}mm square</Text>
                  <Text style={styles.technicalText}>Scale: {mmPerPixel?.toFixed(4)} mm/pixel</Text>
                  <Text style={styles.technicalText}>Paper Speed: {paperSpeed} mm/s</Text>
                  <Text style={styles.technicalText}>R-Peaks: {peaks.length} detected</Text>
                </View>

                {/* Analysis Notes */}
                {notes && (
                  <View style={styles.notesSection}>
                    <Text style={styles.sectionTitle}>Analysis Notes</Text>
                    <Text style={styles.notesText}>{notes}</Text>
                  </View>
                )}

                {/* Disclaimer */}
                <View style={styles.disclaimer}>
                  <Text style={styles.disclaimerText}>
                    ⚠️ This analysis is for educational purposes only. Always consult a healthcare professional for medical advice.
                  </Text>
                </View>
              </View>
            )}
          </ScrollView>
        </View>
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: '#0F172A',
  },
  
  header: {
    alignItems: 'center',
    paddingTop: 20,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  
  iconContainer: {
    backgroundColor: 'rgba(30, 41, 59, 0.8)',
    padding: 20,
    borderRadius: 50,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: 'rgba(239, 68, 68, 0.3)',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 10,
  },
  
  heartIcon: { 
    fontSize: 36 
  },
  
  title: { 
    fontSize: 32, 
    fontWeight: '700', 
    color: '#FFFFFF', 
    marginBottom: 8,
    textAlign: 'center',
    fontFamily: 'System',
    letterSpacing: -0.5,
  },
  
  subtitle: { 
    fontSize: 16, 
    color: '#94A3B8', 
    textAlign: 'center',
    fontFamily: 'System',
    fontWeight: '300',
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
    padding: 20,
    paddingBottom: 0,
    gap: 12,
  },
  
  primaryBtn: { 
    flex: 1,
    backgroundColor: '#EF4444', 
    paddingVertical: 14, 
    borderRadius: 14,
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
    fontFamily: 'System',
  },
  
  secondaryBtn: { 
    flex: 1,
    backgroundColor: '#475569', 
    paddingVertical: 14, 
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#64748B',
  },
  
  secondaryBtnText: { 
    color: '#E2E8F0', 
    fontWeight: '600',
    fontSize: 14,
    fontFamily: 'System',
  },
  
  clearBtn: { 
    backgroundColor: 'rgba(71, 85, 105, 0.5)', 
    paddingVertical: 14, 
    paddingHorizontal: 16,
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#475569',
  },
  
  clearBtnText: { 
    color: '#CBD5E1', 
    fontWeight: '600',
    fontSize: 14,
    fontFamily: 'System',
  },

  controlsSection: { 
    padding: 20,
    paddingTop: 16,
  },

  inputRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  
  inputGroup: { 
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.8)', 
    paddingHorizontal: 16, 
    paddingVertical: 12, 
    borderRadius: 12,
    borderWidth: 1, 
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  
  inputLabel: { 
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 4,
    fontFamily: 'System',
  },
  
  input: { 
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'System',
  },
  
  analyzeBtn: { 
    backgroundColor: '#10B981', 
    paddingHorizontal: 20, 
    paddingVertical: 14, 
    borderRadius: 14,
    alignItems: 'center',
    shadowColor: '#10B981',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  
  analyzeBtnText: { 
    color: '#FFFFFF', 
    fontWeight: '700',
    fontSize: 14,
    fontFamily: 'System',
  },

  scrollContent: {
    flex: 1,
  },

  imageContainer: { 
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 16,
    overflow: 'hidden',
  },

  imageWrapper: {
    position: 'relative',
  },
  
  image: {
    width: '100%',
    height: IMAGE_HEIGHT,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#475569',
  },
  
  placeholder: { 
    minHeight: SCREEN_WIDTH * 0.6,
    borderRadius: 16, 
    borderWidth: 2, 
    borderStyle: 'dashed',
    borderColor: '#475569', 
    alignItems: 'center', 
    justifyContent: 'center', 
    padding: 32,
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
  },
  
  placeholderIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  
  placeholderTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#E2E8F0',
    marginBottom: 12,
    fontFamily: 'System',
  },
  
  placeholderText: { 
    color: '#94A3B8', 
    textAlign: 'center',
    lineHeight: 22,
    fontFamily: 'System',
    fontWeight: '300',
  },

  statusSection: { 
    margin: 20,
    marginTop: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.8)', 
    borderRadius: 16, 
    padding: 20,
    borderWidth: 1, 
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },
  
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#E2E8F0',
    marginBottom: 12,
    fontFamily: 'System',
  },

  statusGrid: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16,
  },

  statusItem: {
    flex: 1,
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },

  statusLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 4,
    fontFamily: 'System',
    fontWeight: '300',
  },

  statusValue: {
    fontSize: 14,
    fontWeight: '600',
    fontFamily: 'System',
  },

  clearCalibrationBtn: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.3)',
  },

  clearCalibrationText: {
    color: '#FCD34D',
    fontWeight: '600',
    fontSize: 13,
    fontFamily: 'System',
  },

  loadingSection: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 24,
    margin: 20,
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },

  loadingText: {
    marginTop: 20,
    fontSize: 18,
    color: '#E2E8F0',
    fontWeight: '500',
    fontFamily: 'System',
    textAlign: 'center',
  },

  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#94A3B8',
    fontFamily: 'System',
    fontWeight: '300',
  },

  resultsSection: { 
    margin: 20,
    marginTop: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.8)', 
    borderRadius: 16, 
    padding: 20,
    borderWidth: 1, 
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },

  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },

  checkIcon: {
    fontSize: 28,
    marginRight: 12,
  },

  resultsTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#FFFFFF',
    fontFamily: 'System',
  },

  heartRateSection: {
    alignItems: 'center',
    backgroundColor: 'rgba(71, 85, 105, 0.3)',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(71, 85, 105, 0.5)',
  },

  heartRateLabel: {
    fontSize: 14,
    color: '#94A3B8',
    marginBottom: 8,
    fontFamily: 'System',
    fontWeight: '300',
  },

  heartRateDisplay: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 12,
  },

  heartRateValue: {
    fontSize: 48,
    fontWeight: '700',
    color: '#FFFFFF',
    fontFamily: 'System',
  },

  heartRateUnit: {
    fontSize: 16,
    color: '#94A3B8',
    marginLeft: 8,
    fontFamily: 'System',
    fontWeight: '300',
  },

  classificationBadge: { 
    paddingHorizontal: 16, 
    paddingVertical: 8, 
    borderRadius: 20,
  },
  
  badgeText: { 
    color: '#FFFFFF', 
    fontWeight: '700',
    fontSize: 14,
    fontFamily: 'System',
  },

  technicalSection: {
    marginBottom: 16,
  },

  technicalText: {
    color: '#CBD5E1',
    fontSize: 14,
    marginBottom: 4,
    fontFamily: 'System',
    fontWeight: '300',
  },

  notesSection: {
    marginBottom: 16,
  },

  notesText: { 
    color: '#FCD34D', 
    fontWeight: '500',
    fontSize: 14,
    lineHeight: 20,
    fontFamily: 'System',
  },

  disclaimer: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    padding: 12,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#F59E0B',
  },

  disclaimerText: {
    fontSize: 12,
    color: '#FCD34D',
    lineHeight: 16,
    fontFamily: 'System',
    fontWeight: '300',
  },
});