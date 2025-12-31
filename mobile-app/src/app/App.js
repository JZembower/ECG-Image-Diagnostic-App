import * as ImagePicker from "expo-image-picker";
import { useRef, useState } from "react";
import { Alert, Dimensions, Image, ScrollView, StatusBar, StyleSheet, Text, TextInput, TouchableOpacity, View } from "react-native";
import Svg, { Circle, Line } from "react-native-svg";

const SCREEN_WIDTH = Dimensions.get("window").width;

export default function App() {
  const [image, setImage] = useState(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  // Calibration: two taps on adjacent BOLD grid lines (5 mm apart)
  const [calibrationPoints, setCalibrationPoints] = useState([]); // [{x, y}, {x, y}]
  const [mmPerPixel, setMmPerPixel] = useState(null);

  // R-peaks: tap on R peaks (2+)
  const [rPeaks, setRPeaks] = useState([]); // [{x,y}, ...]
  const [paperSpeed, setPaperSpeed] = useState(25); // mm/s
  const [bpm, setBpm] = useState(null);
  const [notes, setNotes] = useState("");

  // Rate classification badge
  const [rateClass, setRateClass] = useState({ label: "--", color: "#64748b" });
  const classifyRate = (b) => {
    if (!isFinite(b)) return { label: "--", color: "#64748b" };
    if (b < 60) return { label: "Bradycardia", color: "#F59E0B" };   // amber
    if (b > 100) return { label: "Tachycardia", color: "#EF4444" };  // red
    return { label: "Normal", color: "#10B981" };                    // green
  };

  const imgRef = useRef(null);

  const resetStateForNewImage = (uri) => {
    setImage(uri);
    setMmPerPixel(null);
    setCalibrationPoints([]);
    setRPeaks([]);
    setBpm(null);
    setNotes("");
    setRateClass({ label: "--", color: "#64748b" });
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission required", "Please grant photo library access.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });
    if (!result.canceled) {
      const asset = result.assets[0];
      resetStateForNewImage(asset.uri);
      Image.getSize(asset.uri, (w, h) => {
        setImgSize({ width: w, height: h });
      });
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission required", "Please grant camera access.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 1 });
    if (!result.canceled) {
      const asset = result.assets[0];
      resetStateForNewImage(asset.uri);
      Image.getSize(asset.uri, (w, h) => {
        setImgSize({ width: w, height: h });
      });
    }
  };

  // Convert touch coordinates (view) -> image coordinates
  const viewToImageCoords = (vx, vy) => {
    if (!displaySize.width || !displaySize.height || !imgSize.width || !imgSize.height) return { x: vx, y: vy };
    const ratio = displaySize.width / imgSize.width;
    const scaledHeight = imgSize.height * ratio;
    const offsetY = (displaySize.height - scaledHeight) / 2;
    const ix = vx / ratio;
    const iy = (vy - offsetY) / ratio;
    return { x: ix, y: iy };
  };

  const handleTap = (evt) => {
    if (!image) return;
    const { locationX, locationY } = evt.nativeEvent;
    const pt = viewToImageCoords(locationX, locationY);

    // First collect 2 calibration taps (5 mm apart on bold grid lines)
    if (calibrationPoints.length < 2) {
      const next = [...calibrationPoints, pt];
      setCalibrationPoints(next);

      if (next.length === 2) {
        const dx = next[1].x - next[0].x;
        const dy = next[1].y - next[0].y;
        const pixelDist = Math.sqrt(dx * dx + dy * dy);
        const mmPerPx = 5 / pixelDist; // 5 mm between bold lines
        setMmPerPixel(mmPerPx);
        Alert.alert("Calibration set", "Calibration complete (5 mm between your taps). Now tap R-peaks.");
      }
      return;
    }

    // Otherwise, mark R peaks
    setRPeaks((prev) => [...prev, pt]);
  };

  const clearAll = () => {
    setCalibrationPoints([]);
    setRPeaks([]);
    setMmPerPixel(null);
    setBpm(null);
    setNotes("");
    setRateClass({ label: "--", color: "#64748b" });
  };

  const analyze = () => {
    if (!mmPerPixel) {
      Alert.alert("Missing calibration", "Tap two adjacent bold grid lines first to calibrate (5 mm apart).");
      return;
    }
    if (rPeaks.length < 2) {
      Alert.alert("Need more points", "Tap at least two R-peaks on the ECG trace.");
      return;
    }

    // Sort R-peaks left->right to compute RR intervals
    const sorted = [...rPeaks].sort((a, b) => a.x - b.x);

    const rrPixels = [];
    for (let i = 1; i < sorted.length; i++) {
      const dx = sorted[i].x - sorted[i - 1].x;
      const dy = sorted[i].y - sorted[i - 1].y;
      rrPixels.push(Math.sqrt(dx * dx + dy * dy));
    }

    const rrMm = rrPixels.map((p) => p * mmPerPixel);
    const rrSeconds = rrMm.map((mm) => mm / paperSpeed); // paperSpeed mm/s
    const meanRR = rrSeconds.reduce((a, b) => a + b, 0) / rrSeconds.length;
    const bpmEst = 60 / meanRR;

    // Simple variability check
    const variance = rrSeconds.reduce((acc, v) => acc + Math.pow(v - meanRR, 2), 0) / rrSeconds.length;
    const sdRR = Math.sqrt(variance);

    // Irregularity flag only (rate classification now shown as a badge)
    let flag = "";
    if (sdRR > 0.12) flag += "Irregular rhythm (high RR variability). ";

    setBpm(bpmEst);
    setRateClass(classifyRate(bpmEst));
    setNotes(flag || "Regular rhythm by simple RR variability check.");
  };

  // Draw overlay (calibration + R-peak markers)
  const drawOverlay = () => {
    if (!displaySize.width || !displaySize.height) return null;
    const ratio = displaySize.width / (imgSize.width || 1);
    const scaledHeight = (imgSize.height || 1) * ratio;
    const offsetY = (displaySize.height - scaledHeight) / 2;

    const mapX = (ix) => ix * ratio;
    const mapY = (iy) => iy * ratio + offsetY;

    const sortedR = [...rPeaks].sort((a, b) => a.x - b.x);

    return (
      <Svg pointerEvents="none" width={displaySize.width} height={displaySize.height} style={StyleSheet.absoluteFill}>
        {/* Calibration points & line */}
        {calibrationPoints.map((p, i) => (
          <Circle key={`cal-${i}`} cx={mapX(p.x)} cy={mapY(p.y)} r={8} fill="#3B82F6" stroke="#FFFFFF" strokeWidth="2" />
        ))}
        {calibrationPoints.length === 2 && (
          <Line
            x1={mapX(calibrationPoints[0].x)}
            y1={mapY(calibrationPoints[0].y)}
            x2={mapX(calibrationPoints[1].x)}
            y2={mapY(calibrationPoints[1].y)}
            stroke="#3B82F6"
            strokeWidth={3}
          />
        )}

        {/* R-peak markers & connecting lines */}
        {sortedR.map((p, i) => (
          <Circle key={`r-${i}`} cx={mapX(p.x)} cy={mapY(p.y)} r={8} fill="#EF4444" stroke="#FFFFFF" strokeWidth="2" />
        ))}
        {sortedR.length > 1 &&
          sortedR.slice(1).map((p, i) => {
            const prev = sortedR[i];
            return (
              <Line
                key={`rl-${i}`}
                x1={mapX(prev.x)}
                y1={mapY(prev.y)}
                x2={mapX(p.x)}
                y2={mapY(p.y)}
                stroke="#EF4444"
                strokeWidth={2}
                opacity={0.7}
              />
            );
          })}
      </Svg>
    );
  };

  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" translucent={false} />
      <View style={styles.container}>
        
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.iconContainer}>
            <Text style={styles.heartIcon}>❤️</Text>
          </View>
          <Text style={styles.title}>ECG Analyzer</Text>
          <Text style={styles.subtitle}>AI-powered ECG analysis with manual calibration</Text>
        </View>

        {/* Main Card */}
        <View style={styles.card}>
          
          {/* Action Buttons */}
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.primaryBtn} onPress={takePhoto}>
              <Text style={styles.primaryBtnText}>📸 Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryBtn} onPress={pickImage}>
              <Text style={styles.secondaryBtnText}>📁 Gallery</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.clearBtn} onPress={clearAll}>
              <Text style={styles.clearBtnText}>🗑️ Clear</Text>
            </TouchableOpacity>
          </View>

          {/* Controls */}
          <View style={styles.controlsSection}>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Paper speed (mm/s)</Text>
              <TextInput
                style={styles.input}
                keyboardType="numeric"
                placeholder="25"
                placeholderTextColor="#64748B"
                value={String(paperSpeed)}
                onChangeText={(t) => setPaperSpeed(parseFloat(t) || 25)}
              />
            </View>
            <TouchableOpacity style={styles.analyzeBtn} onPress={analyze}>
              <Text style={styles.analyzeBtnText}>🧠 Analyze ECG</Text>
            </TouchableOpacity>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} style={styles.scrollContent}>
            
            {/* Image Section */}
            <View
              style={styles.imageContainer}
              onLayout={(e) => {
                const { width } = e.nativeEvent.layout;
                const aspect = imgSize.width && imgSize.height ? imgSize.height / imgSize.width : 0.66;
                const height = width * (aspect || 0.66);
                setDisplaySize({ width, height });
              }}
            >
              {image ? (
                <TouchableOpacity activeOpacity={1} onPress={handleTap} style={styles.imageWrapper}>
                  <Image
                    ref={imgRef}
                    source={{ uri: image }}
                    style={[styles.image, { width: displaySize.width, height: displaySize.height }]}
                    resizeMode="contain"
                  />
                  {drawOverlay()}
                </TouchableOpacity>
              ) : (
                <View style={styles.placeholder}>
                  <Text style={styles.placeholderIcon}>📷</Text>
                  <Text style={styles.placeholderTitle}>Ready to Analyze ECG</Text>
                  <Text style={styles.placeholderText}>
                    1. Take or select an ECG photo{"\n"}
                    2. Tap two bold grid lines (5mm apart){"\n"}
                    3. Mark R-peaks on the waveform{"\n"}
                    4. Tap Analyze for results
                  </Text>
                </View>
              )}
            </View>

            {/* Results Section */}
            <View style={styles.resultsSection}>
              <Text style={styles.sectionTitle}>Analysis Status</Text>
              
              <View style={styles.statusGrid}>
                <View style={styles.statusItem}>
                  <Text style={styles.statusLabel}>Calibration</Text>
                  <Text style={[styles.statusValue, { color: mmPerPixel ? '#10B981' : '#64748B' }]}>
                    {mmPerPixel ? '✅ Set' : '❌ Required'}
                  </Text>
                </View>
                <View style={styles.statusItem}>
                  <Text style={styles.statusLabel}>R-Peaks</Text>
                  <Text style={[styles.statusValue, { color: rPeaks.length >= 2 ? '#10B981' : '#64748B' }]}>
                    {rPeaks.length >= 2 ? `✅ ${rPeaks.length}` : `${rPeaks.length} marked`}
                  </Text>
                </View>
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
              {mmPerPixel && (
                <View style={styles.technicalSection}>
                  <Text style={styles.sectionTitle}>Technical Details</Text>
                  <Text style={styles.technicalText}>
                    Scale: {mmPerPixel.toFixed(4)} mm/pixel
                  </Text>
                  <Text style={styles.technicalText}>
                    Paper Speed: {paperSpeed} mm/s
                  </Text>
                </View>
              )}

              {/* Analysis Notes */}
              {notes && (
                <View style={styles.notesSection}>
                  <Text style={styles.sectionTitle}>Notes</Text>
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
          </ScrollView>
        </View>
      </View>
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

  resultsSection: { 
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
    marginBottom: 20,
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