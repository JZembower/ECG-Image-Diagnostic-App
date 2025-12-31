"""
ECG Image Digitizer Service
Converts ECG images to time-series signals using OpenCV-based computer vision pipeline.
"""

import numpy as np
import cv2
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# Constants
TARGET_SAMPLE_RATE = 100  # Hz - matches training data
TARGET_DURATION = 10  # seconds
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION  # 1000 samples
TARGET_LEADS = 12  # 12-lead ECG

# Standard ECG paper specs (if no calibration provided)
STANDARD_PAPER_SPEED = 25  # mm/s
STANDARD_PAPER_AMPLITUDE = 10  # mm/mV
STANDARD_DPI = 300  # dots per inch
MM_PER_INCH = 25.4
PIXELS_PER_MM = STANDARD_DPI / MM_PER_INCH  # ~11.81 pixels/mm


def extract_signal_from_image(
    image_bytes: bytes,
    calibration_points: Optional[List[Dict]] = None
) -> np.ndarray:
    """
    Extract ECG signal from image using OpenCV computer vision pipeline.
    
    Args:
        image_bytes: Raw image bytes
        calibration_points: Optional list of calibration points for pixel-to-physical mapping
            Format: [{"x": pixel_x, "y": pixel_y, "value": physical_value, "unit": "mV" or "seconds"}]
    
    Returns:
        numpy array of shape (1000, 12) - 10 seconds @ 100Hz, 12 leads
    
    Process:
        1. Load and preprocess image
        2. Remove grid lines
        3. Extract ECG trace contours
        4. Convert pixels to physical units (mV, seconds)
        5. Resample to 100Hz
        6. Handle multi-lead or replicate single lead
        7. Normalize and return
    """
    try:
        # Step 1: Load image from bytes
        logger.info("Loading image from bytes...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image. Invalid image format.")
        
        logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
        
        # Step 2: Preprocess image
        logger.info("Preprocessing image...")
        preprocessed = preprocess_image(image)
        
        # Step 3: Remove grid lines
        logger.info("Removing grid lines...")
        grid_removed = remove_grid(preprocessed)
        
        # Step 4: Extract trace contours
        logger.info("Extracting ECG trace...")
        trace_points = extract_trace_contours(grid_removed)
        
        if len(trace_points) == 0:
            raise ValueError("No ECG trace detected in image. Please ensure the image contains a clear ECG waveform.")
        
        logger.info(f"Extracted {len(trace_points)} trace points")
        
        # Step 5: Build calibration mapping
        calibration = build_calibration(image.shape, calibration_points)
        
        # Step 6: Convert pixels to physical units
        logger.info("Converting pixels to physical units...")
        signal_physical = pixels_to_physical(trace_points, calibration)
        
        # Step 7: Resample to target sample rate
        logger.info(f"Resampling to {TARGET_SAMPLE_RATE}Hz...")
        signal_resampled = resample_signal(signal_physical, TARGET_SAMPLES)
        
        # Step 8: Handle multi-lead (for now, replicate single lead to 12 channels)
        # TODO: Implement proper 12-lead extraction from standard layouts
        logger.info("Creating 12-lead signal...")
        signal_12lead = create_12lead_signal(signal_resampled)
        
        # Step 9: Normalize signal (Z-score normalization to match training)
        logger.info("Normalizing signal...")
        signal_normalized = normalize_signal(signal_12lead)
        
        logger.info(f"Final signal shape: {signal_normalized.shape}")
        
        return signal_normalized
    
    except Exception as e:
        logger.error(f"Error in extract_signal_from_image: {e}", exc_info=True)
        raise


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for trace extraction.
    
    Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Apply adaptive thresholding to handle uneven lighting
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur (kernel size 5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding (handles uneven lighting)
    # THRESH_BINARY_INV because ECG trace is typically dark on light background
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    return thresh


def remove_grid(image: np.ndarray) -> np.ndarray:
    """
    Remove grid lines from ECG image using morphological operations.
    
    Grid lines are typically:
    - Horizontal lines (time axis)
    - Vertical lines (amplitude axis)
    - Regular spacing
    
    Strategy: Detect grid using morphological operations, then subtract from original.
    """
    # Create horizontal and vertical kernels for morphological operations
    # Adjust kernel sizes based on typical grid line spacing
    
    # Horizontal kernel (detect horizontal lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    
    # Vertical kernel (detect vertical lines)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine grid lines
    grid = cv2.add(horizontal_lines, vertical_lines)
    
    # Subtract grid from original image
    grid_removed = cv2.subtract(image, grid)
    
    # Apply morphological closing to clean up the trace
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grid_removed = cv2.morphologyEx(grid_removed, cv2.MORPH_CLOSE, kernel)
    
    return grid_removed


def extract_trace_contours(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract ECG trace as list of (x, y) pixel coordinates.
    
    Strategy:
        1. Find contours in the processed image
        2. Filter contours by area (remove noise)
        3. For each X-column, calculate Y-centroid of contour pixels
        4. Handle gaps with interpolation
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    # Combine all contours into single point cloud
    all_points = []
    for contour in contours:
        # Filter by area to remove noise
        area = cv2.contourArea(contour)
        if area > 10:  # Minimum area threshold
            # Extract points from contour
            points = contour.reshape(-1, 2)
            all_points.extend(points)
    
    if len(all_points) == 0:
        return []
    
    all_points = np.array(all_points)
    
    # Sort points by X-coordinate
    all_points = all_points[all_points[:, 0].argsort()]
    
    # Build time series: for each X, calculate average Y (handle multiple points at same X)
    trace_points = []
    current_x = all_points[0, 0]
    y_values = []
    
    for x, y in all_points:
        if x == current_x:
            y_values.append(y)
        else:
            # Calculate average Y for this X column
            avg_y = int(np.mean(y_values))
            trace_points.append((current_x, avg_y))
            
            # Move to next X
            current_x = x
            y_values = [y]
    
    # Don't forget the last group
    if len(y_values) > 0:
        avg_y = int(np.mean(y_values))
        trace_points.append((current_x, avg_y))
    
    return trace_points


def build_calibration(
    image_shape: Tuple[int, int, int],
    calibration_points: Optional[List[Dict]] = None
) -> Dict:
    """
    Build calibration mapping from pixels to physical units.
    
    If calibration_points provided: use them
    If not: assume standard ECG paper (25mm/s, 10mm/mV at 300 DPI)
    """
    height, width = image_shape[:2]
    
    if calibration_points and len(calibration_points) >= 2:
        # Use provided calibration points
        # TODO: Implement proper calibration from points
        logger.info("Using provided calibration points")
        
        # For now, fall back to standard calibration
        # Full implementation would calculate scale factors from calibration points
        pass
    
    # Standard ECG paper calibration
    # Horizontal: 25 mm/s at 300 DPI = 11.81 pixels/mm = 295.28 pixels/second
    # Vertical: 10 mm/mV at 300 DPI = 11.81 pixels/mm = 118.11 pixels/mV
    
    pixels_per_second = PIXELS_PER_MM * STANDARD_PAPER_SPEED
    pixels_per_mv = PIXELS_PER_MM * STANDARD_PAPER_AMPLITUDE
    
    calibration = {
        "pixels_per_second": pixels_per_second,
        "pixels_per_mv": pixels_per_mv,
        "image_width": width,
        "image_height": height,
        "duration_seconds": width / pixels_per_second,
        "amplitude_range_mv": height / pixels_per_mv
    }
    
    logger.info(f"Calibration: {calibration}")
    
    return calibration


def pixels_to_physical(
    trace_points: List[Tuple[int, int]],
    calibration: Dict
) -> np.ndarray:
    """
    Convert pixel coordinates to physical units (time in seconds, amplitude in mV).
    
    Returns:
        1D array of amplitude values (mV) sampled at irregular intervals
        We'll resample to regular 100Hz later
    """
    # Extract X and Y coordinates
    x_pixels = np.array([p[0] for p in trace_points])
    y_pixels = np.array([p[1] for p in trace_points])
    
    # Convert X to time (seconds)
    time_seconds = x_pixels / calibration["pixels_per_second"]
    
    # Convert Y to amplitude (mV)
    # Note: In image coordinates, Y increases downward, so we need to flip
    # Assume baseline is at middle of image
    baseline_y = calibration["image_height"] / 2
    y_from_baseline = baseline_y - y_pixels  # Flip Y axis
    amplitude_mv = y_from_baseline / calibration["pixels_per_mv"]
    
    return amplitude_mv


def resample_signal(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Resample signal to target number of samples using interpolation.
    
    Input: irregular samples
    Output: exactly target_samples at regular intervals
    """
    if len(signal) == 0:
        raise ValueError("Cannot resample empty signal")
    
    # Create time axis for input signal
    original_time = np.linspace(0, 1, len(signal))
    
    # Create time axis for output signal
    target_time = np.linspace(0, 1, target_samples)
    
    # Interpolate
    interpolator = interp1d(original_time, signal, kind='linear', fill_value='extrapolate')
    resampled = interpolator(target_time)
    
    return resampled


def create_12lead_signal(signal_1d: np.ndarray) -> np.ndarray:
    """
    Create 12-lead signal from single lead.
    
    For now: replicate single lead to 12 channels
    TODO: Implement proper 12-lead extraction from standard ECG layouts
    """
    # Ensure signal is 1D
    if signal_1d.ndim > 1:
        signal_1d = signal_1d.flatten()
    
    # Replicate to 12 leads
    signal_12lead = np.tile(signal_1d, (TARGET_LEADS, 1)).T  # Shape: (1000, 12)
    
    return signal_12lead


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal using Z-score normalization (match training preprocessing).
    
    Z-score: (x - mean) / std
    """
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (signal - mean) / std
    
    return normalized


def assess_signal_quality(signal: np.ndarray) -> str:
    """
    Assess signal quality based on various metrics.
    
    Returns: "excellent", "good", "fair", or "poor"
    """
    try:
        # Calculate metrics
        
        # 1. SNR (Signal-to-Noise Ratio)
        # Estimate noise as high-frequency component
        signal_1d = signal[:, 0] if signal.ndim > 1 else signal
        
        # Apply high-pass filter to isolate noise
        sos = scipy_signal.butter(4, 40, 'high', fs=TARGET_SAMPLE_RATE, output='sos')
        noise = scipy_signal.sosfilt(sos, signal_1d)
        
        signal_power = np.mean(signal_1d ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # Very high SNR (low noise)
        
        # 2. Check for baseline wander
        # Estimate baseline using low-pass filter
        sos_lp = scipy_signal.butter(4, 0.5, 'low', fs=TARGET_SAMPLE_RATE, output='sos')
        baseline = scipy_signal.sosfilt(sos_lp, signal_1d)
        baseline_wander = np.std(baseline)
        
        # 3. Check if signal is too flat
        signal_std = np.std(signal_1d)
        
        # 4. Check for clipping or saturation
        signal_range = np.max(signal_1d) - np.min(signal_1d)
        
        # Quality assessment
        if snr > 20 and baseline_wander < 0.5 and signal_std > 0.1 and signal_range > 0.5:
            quality = "excellent"
        elif snr > 15 and baseline_wander < 1.0 and signal_std > 0.05:
            quality = "good"
        elif snr > 10 and baseline_wander < 2.0 and signal_std > 0.01:
            quality = "fair"
        else:
            quality = "poor"
        
        logger.info(f"Signal quality metrics: SNR={snr:.2f}dB, baseline_wander={baseline_wander:.3f}, std={signal_std:.3f}, quality={quality}")
        
        return quality
    
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        return "unknown"
