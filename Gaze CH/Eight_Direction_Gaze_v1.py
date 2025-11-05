"""
Accurate 8-Direction Gaze Detection with Calibration Support
Uses hybrid zone-based detection with empirically tuned thresholds
Supports: CENTER, LEFT, RIGHT, UP, DOWN, LEFT_UP, LEFT_DOWN, RIGHT_UP, RIGHT_DOWN
"""

import cv2
import numpy as np
import onnxruntime as ort
import uniface
import collections
from typing import Tuple, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass, field
import time
import threading
import queue
from pynput import mouse
import json
import os

# ==================== SETTINGS ====================
@dataclass
class Settings:
    """Centralized configuration for accurate 8-direction gaze detection"""
    
    # Video/Camera Settings
    VIDEO_SOURCE: int = 0
    MODEL_PATH: str = "mobileone_s0_gaze.onnx"
    OUTPUT_VIDEO: Optional[str] = None
    
    # Performance Settings
    FRAME_SKIP: int = 2
    FACE_DETECTION_SCALE: float = 0.5
    USE_THREADING: bool = True
    
    # Empirically Tuned Thresholds for Each Direction
    # These values are based on typical gaze patterns and can be calibrated
    # CORRECTED: positive yaw = looking up, negative yaw = looking down
    # IMPROVED: Wider ranges for diagonals with overlap for easier detection
    THRESHOLDS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'LEFT': {'pitch_min': -35, 'pitch_max': -10, 'yaw_min': -12, 'yaw_max': 12},
        'RIGHT': {'pitch_min': 10, 'pitch_max': 35, 'yaw_min': -12, 'yaw_max': 12},
        'UP': {'pitch_min': -12, 'pitch_max': 12, 'yaw_min': 10, 'yaw_max': 35},    # FIXED: positive yaw for up
        'DOWN': {'pitch_min': -12, 'pitch_max': 12, 'yaw_min': -35, 'yaw_max': -10}, # FIXED: negative yaw for down
        # Diagonals with more generous thresholds for easier activation
        'LEFT_UP': {'pitch_min': -35, 'pitch_max': -5, 'yaw_min': 5, 'yaw_max': 35},     # Wider range
        'RIGHT_UP': {'pitch_min': 5, 'pitch_max': 35, 'yaw_min': 5, 'yaw_max': 35},      # Wider range
        'LEFT_DOWN': {'pitch_min': -35, 'pitch_max': -5, 'yaw_min': -35, 'yaw_max': -5}, # Wider range
        'RIGHT_DOWN': {'pitch_min': 5, 'pitch_max': 35, 'yaw_min': -35, 'yaw_max': -5},  # Wider range
        'CENTER': {'pitch_min': -10, 'pitch_max': 10, 'yaw_min': -10, 'yaw_max': 10}
    })
    
    # Zone-based detection parameters
    ZONE_OVERLAP: float = 0.2  # 20% overlap between zones
    ZONE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'distance': 0.4,    # Weight for distance from zone center
        'magnitude': 0.3,   # Weight for gaze magnitude
        'consistency': 0.3  # Weight for temporal consistency
    })
    
    # Priority system for resolving conflicts
    # Balanced priority - diagonals get fair chance
    DIRECTION_PRIORITY: List[str] = field(default_factory=lambda: [
        'CENTER',     # Highest priority
        'LEFT_UP', 'RIGHT_UP', 'LEFT_DOWN', 'RIGHT_DOWN',  # Diagonals (when both axes active)
        'UP', 'DOWN', 'LEFT', 'RIGHT'  # Cardinal directions (single axis dominant)
    ])
    
    # Calibration Settings
    ENABLE_CALIBRATION: bool = False
    CALIBRATION_SAMPLES: int = 30  # Samples per direction for calibration
    CALIBRATION_FILE: str = "gaze_calibration.json"
    AUTO_CALIBRATE: bool = False  # Auto-calibrate based on usage patterns
    
    # Temporal Smoothing
    SMOOTHING_WINDOW: int = 7  # Increased for better accuracy
    CONFIDENCE_THRESHOLD: float = 0.5  # Lowered for more responsive detection
    MIN_CONSISTENT_FRAMES: int = 3
    USE_KALMAN_FILTER: bool = True  # Use Kalman filter for smoother tracking
    
    # Dead Zone and Activation
    CENTER_DEAD_ZONE: float = 6.0  # Smaller dead zone for center
    MIN_GAZE_MAGNITUDE: float = 8.0  # Minimum magnitude to activate any direction
    DIAGONAL_MAGNITUDE_BOOST: float = 1.0  # Removed penalty - diagonals same as cardinals
    
    # Mouse Control Settings
    ENABLE_MOUSE_CONTROL: bool = True
    MOUSE_SPEED_BASE: int = 12
    MOUSE_SPEED_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        'LEFT': 1.0, 'RIGHT': 1.0, 'UP': 1.0, 'DOWN': 1.0,
        'LEFT_UP': 0.85, 'RIGHT_UP': 0.85, 'LEFT_DOWN': 0.85, 'RIGHT_DOWN': 0.85
    })
    MOUSE_ACCELERATION: float = 1.5
    MOUSE_MAX_SPEED: int = 40
    MOUSE_SMOOTH_FACTOR: float = 0.35
    
    # Movement Delay Settings (NEW)
    MOVEMENT_DELAY: float = 0.3  # Delay in seconds between direction changes
    MOVEMENT_HOLD_TIME: float = 0.15  # Minimum time to hold a direction before switching
    ALLOW_DIAGONAL_TRANSITION: bool = True  # Allow direct diagonal transitions
    REQUIRE_CENTER_BETWEEN: bool = False  # Require CENTER state between direction changes
    
    # Display Settings
    SHOW_VISUALIZATION: bool = True
    SHOW_DEBUG_INFO: bool = False
    SHOW_CALIBRATION_GUIDE: bool = True
    SHOW_CONFIDENCE_MAP: bool = False
    SHOW_FPS: bool = True
    SHOW_ANGLE_VALUES: bool = False  # Display real-time angle values
    LOG_CHANGES: bool = False
    
    # Advanced Settings
    RESET_AFTER_NO_FACE: int = 10
    USE_FACE_ORIENTATION: bool = True  # Consider face angle in detection
    BLINK_DETECTION: bool = False  # Detect blinks for additional control


# ==================== DIRECTION ENUM ====================
class GazeDirection(Enum):
    """8 primary gaze directions + center"""
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    LEFT_UP = "LEFT_UP"
    LEFT_DOWN = "LEFT_DOWN"
    RIGHT_UP = "RIGHT_UP"
    RIGHT_DOWN = "RIGHT_DOWN"


# ==================== KALMAN FILTER ====================
class KalmanFilter2D:
    """2D Kalman filter for smooth angle tracking"""
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.state = np.zeros(4)  # [x, y, dx, dy]
        self.P = np.eye(4) * 100  # Covariance matrix
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_variance
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_variance
        
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement"""
        if not self.initialized:
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return measurement
        
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return np.array([self.state[0], self.state[1]])
    
    def reset(self):
        """Reset filter state"""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False


# ==================== CALIBRATION MANAGER ====================
class CalibrationManager:
    """Manages calibration for personalized gaze detection"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calibration_data = {}
        self.current_samples = {}
        self.is_calibrating = False
        self.current_direction = None
        self.samples_collected = 0
        
        # Load existing calibration if available
        if os.path.exists(settings.CALIBRATION_FILE):
            self.load_calibration()
    
    def start_calibration(self, direction: GazeDirection):
        """Start calibration for a specific direction"""
        self.is_calibrating = True
        self.current_direction = direction
        self.current_samples = {'pitch': [], 'yaw': []}
        self.samples_collected = 0
        print(f"\nStarting calibration for {direction.value}")
        print(f"Please look {direction.value} and maintain gaze...")
    
    def add_sample(self, pitch: float, yaw: float):
        """Add a calibration sample"""
        if not self.is_calibrating:
            return False
        
        self.current_samples['pitch'].append(pitch)
        self.current_samples['yaw'].append(yaw)
        self.samples_collected += 1
        
        if self.samples_collected >= self.settings.CALIBRATION_SAMPLES:
            self.finish_current_calibration()
            return True
        
        return False
    
    def finish_current_calibration(self):
        """Finish calibration for current direction"""
        if not self.current_samples['pitch']:
            return
        
        # Calculate statistics
        pitch_mean = np.mean(self.current_samples['pitch'])
        pitch_std = np.std(self.current_samples['pitch'])
        yaw_mean = np.mean(self.current_samples['yaw'])
        yaw_std = np.std(self.current_samples['yaw'])
        
        # Store calibration data
        self.calibration_data[self.current_direction.value] = {
            'pitch_center': pitch_mean,
            'pitch_range': pitch_std * 2,  # 95% confidence interval
            'yaw_center': yaw_mean,
            'yaw_range': yaw_std * 2,
            'samples': self.samples_collected
        }
        
        print(f"Calibration complete for {self.current_direction.value}")
        print(f"  Pitch: {pitch_mean:.1f}° ± {pitch_std:.1f}°")
        print(f"  Yaw: {yaw_mean:.1f}° ± {yaw_std:.1f}°")
        
        self.is_calibrating = False
        self.current_direction = None
        
    def save_calibration(self):
        """Save calibration to file"""
        with open(self.settings.CALIBRATION_FILE, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"Calibration saved to {self.settings.CALIBRATION_FILE}")
    
    def load_calibration(self):
        """Load calibration from file"""
        try:
            with open(self.settings.CALIBRATION_FILE, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration loaded from {self.settings.CALIBRATION_FILE}")
            return True
        except Exception as e:
            print(f"Could not load calibration: {e}")
            return False
    
    def get_calibrated_thresholds(self):
        """Get calibrated thresholds based on collected data"""
        if not self.calibration_data:
            return None
        
        thresholds = {}
        for direction, data in self.calibration_data.items():
            thresholds[direction] = {
                'pitch_min': data['pitch_center'] - data['pitch_range'],
                'pitch_max': data['pitch_center'] + data['pitch_range'],
                'yaw_min': data['yaw_center'] - data['yaw_range'],
                'yaw_max': data['yaw_center'] + data['yaw_range']
            }
        
        return thresholds


# ==================== ACCURATE GAZE DETECTOR ====================
class AccurateGazeDetector:
    """Highly accurate 8-direction gaze detection using hybrid approach"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calibration = CalibrationManager(settings)
        
        # Kalman filters for smooth tracking
        self.kalman = KalmanFilter2D() if settings.USE_KALMAN_FILTER else None
        
        # Direction confidence scores
        self.direction_scores = {d: 0.0 for d in GazeDirection}
        
        # Historical data for pattern analysis
        self.angle_history = collections.deque(maxlen=20)
        self.magnitude_history = collections.deque(maxlen=20)
        
        # Zone centers for zone-based detection
        self.zone_centers = self._calculate_zone_centers()
        
        # Update thresholds if calibration exists
        self._update_thresholds_from_calibration()
    
    def _calculate_zone_centers(self) -> Dict[GazeDirection, Tuple[float, float]]:
        """Calculate ideal center points for each direction zone"""
        # CORRECTED: positive yaw = up, negative yaw = down
        return {
            GazeDirection.CENTER: (0, 0),
            GazeDirection.LEFT: (-20, 0),
            GazeDirection.RIGHT: (20, 0),
            GazeDirection.UP: (0, 20),      # FIXED: positive yaw for up
            GazeDirection.DOWN: (0, -20),   # FIXED: negative yaw for down
            GazeDirection.LEFT_UP: (-16, 16),     # Adjusted for better detection
            GazeDirection.RIGHT_UP: (16, 16),     # Adjusted for better detection
            GazeDirection.LEFT_DOWN: (-16, -16),  # Adjusted for better detection
            GazeDirection.RIGHT_DOWN: (16, -16),  # Adjusted for better detection
        }
    
    def _update_thresholds_from_calibration(self):
        """Update detection thresholds from calibration data"""
        calibrated = self.calibration.get_calibrated_thresholds()
        if calibrated:
            self.settings.THRESHOLDS.update(calibrated)
            print("Using calibrated thresholds")
    
    def detect_direction(self, pitch_deg: float, yaw_deg: float) -> Tuple[GazeDirection, float]:
        """
        Detect gaze direction with confidence score using hybrid approach.
        
        Args:
            pitch_deg: Pitch angle (horizontal: negative=left, positive=right)
            yaw_deg: Yaw angle (vertical: positive=up, negative=down)
            
        Returns:
            Tuple of (direction, confidence)
        """
        # Apply Kalman filtering if enabled
        if self.kalman:
            filtered = self.kalman.update(np.array([pitch_deg, yaw_deg]))
            pitch_deg, yaw_deg = filtered[0], filtered[1]
        
        # Store history
        self.angle_history.append((pitch_deg, yaw_deg))
        magnitude = np.sqrt(pitch_deg**2 + yaw_deg**2)
        self.magnitude_history.append(magnitude)
        
        # Add calibration sample if calibrating
        if self.calibration.is_calibrating:
            self.calibration.add_sample(pitch_deg, yaw_deg)
        
        # Use hybrid detection combining threshold and zone methods
        return self._hybrid_detection(pitch_deg, yaw_deg)
    
    def _threshold_based_detection(self, pitch: float, yaw: float) -> Tuple[GazeDirection, float]:
        """Threshold-based detection with improved diagonal detection"""
        magnitude = np.sqrt(pitch**2 + yaw**2)
        
        # Check center first
        if magnitude < self.settings.CENTER_DEAD_ZONE:
            return GazeDirection.CENTER, 1.0
        
        # Check minimum activation
        if magnitude < self.settings.MIN_GAZE_MAGNITUDE:
            return GazeDirection.CENTER, 0.7
        
        # Calculate the ratio of horizontal vs vertical movement
        # This helps determine if we should prioritize diagonal detection
        horizontal_strength = abs(pitch)
        vertical_strength = abs(yaw)
        
        # If both axes have significant movement, check diagonals first
        prefer_diagonal = (min(horizontal_strength, vertical_strength) / 
                          max(horizontal_strength, vertical_strength, 0.001)) > 0.5
        
        # Calculate scores for each direction
        scores = {}
        
        # Check all directions
        for direction in GazeDirection:
            if direction == GazeDirection.CENTER:
                continue
            
            thresholds = self.settings.THRESHOLDS[direction.value]
            
            # Check if angles fall within thresholds
            pitch_match = thresholds['pitch_min'] <= pitch <= thresholds['pitch_max']
            yaw_match = thresholds['yaw_min'] <= yaw <= thresholds['yaw_max']
            
            if pitch_match and yaw_match:
                # Calculate how centered the gaze is within the threshold range
                pitch_center = (thresholds['pitch_min'] + thresholds['pitch_max']) / 2
                yaw_center = (thresholds['yaw_min'] + thresholds['yaw_max']) / 2
                
                pitch_range = thresholds['pitch_max'] - thresholds['pitch_min']
                yaw_range = thresholds['yaw_max'] - thresholds['yaw_min']
                
                # Calculate individual axis scores
                pitch_score = 1.0 - abs(pitch - pitch_center) / (pitch_range / 2) if pitch_range > 0 else 0
                yaw_score = 1.0 - abs(yaw - yaw_center) / (yaw_range / 2) if yaw_range > 0 else 0
                
                # For diagonals, use multiplicative scoring to ensure both axes are active
                is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP, 
                                           GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
                
                if is_diagonal:
                    # Diagonal score based on both axes being active
                    base_score = pitch_score * yaw_score * 1.5  # Boost diagonal scores
                    
                    # Additional boost if both axes are strong
                    if prefer_diagonal:
                        base_score *= 1.3
                    
                    scores[direction] = base_score
                else:
                    # Cardinal direction scoring
                    base_score = (pitch_score + yaw_score) / 2
                    
                    # Reduce cardinal score if diagonal is preferred
                    if prefer_diagonal:
                        base_score *= 0.7
                    
                    scores[direction] = base_score
        
        if not scores:
            return GazeDirection.CENTER, 0.5
        
        # Get best direction
        best_direction = max(scores.items(), key=lambda x: x[1])
        
        # Handle conflicts between overlapping zones
        if len(scores) > 1:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # If diagonal and cardinal are close, prefer based on movement pattern
            if sorted_scores[0][1] - sorted_scores[1][1] < 0.15:
                first_is_diagonal = sorted_scores[0][0] in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                                            GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
                second_is_diagonal = sorted_scores[1][0] in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                                             GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
                
                # If choosing between diagonal and cardinal, use movement ratio
                if first_is_diagonal != second_is_diagonal:
                    if prefer_diagonal and first_is_diagonal:
                        return sorted_scores[0][0], sorted_scores[0][1]
                    elif not prefer_diagonal and not first_is_diagonal:
                        return sorted_scores[0][0], sorted_scores[0][1]
                    else:
                        return sorted_scores[1][0], sorted_scores[1][1]
        
        return best_direction[0], best_direction[1]
    
    def _zone_based_detection(self, pitch: float, yaw: float) -> Tuple[GazeDirection, float]:
        """Zone-based detection using distance and weights"""
        magnitude = np.sqrt(pitch**2 + yaw**2)
        
        # Check center
        if magnitude < self.settings.CENTER_DEAD_ZONE:
            return GazeDirection.CENTER, 1.0
        
        # Calculate weighted scores for each zone
        scores = {}
        
        for direction, center in self.zone_centers.items():
            if direction == GazeDirection.CENTER:
                continue
            
            # Calculate distance to zone center
            distance = np.sqrt((pitch - center[0])**2 + (yaw - center[1])**2)
            
            # Calculate individual components
            distance_score = np.exp(-distance / 10)  # Exponential decay
            magnitude_score = min(magnitude / 20, 1.0)  # Normalize magnitude
            
            # Temporal consistency score
            consistency_score = 0.5  # Default
            if len(self.angle_history) > 5:
                recent_angles = list(self.angle_history)[-5:]
                distances = [np.sqrt((p - center[0])**2 + (y - center[1])**2) 
                           for p, y in recent_angles]
                consistency_score = 1.0 - np.std(distances) / 10
            
            # Weighted combination
            weights = self.settings.ZONE_WEIGHTS
            scores[direction] = (
                weights['distance'] * distance_score +
                weights['magnitude'] * magnitude_score +
                weights['consistency'] * consistency_score
            )
            
            # Apply diagonal adjustment
            if direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                           GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]:
                if magnitude < self.settings.MIN_GAZE_MAGNITUDE * self.settings.DIAGONAL_MAGNITUDE_BOOST:
                    scores[direction] *= 0.8
        
        if not scores:
            return GazeDirection.CENTER, 0.5
        
        best_direction = max(scores.items(), key=lambda x: x[1])
        return best_direction[0], min(best_direction[1], 1.0)
    
    def _hybrid_detection(self, pitch: float, yaw: float) -> Tuple[GazeDirection, float]:
        """Hybrid detection combining threshold and zone methods"""
        # Get results from both methods
        thresh_dir, thresh_conf = self._threshold_based_detection(pitch, yaw)
        zone_dir, zone_conf = self._zone_based_detection(pitch, yaw)
        
        # If both agree, high confidence
        if thresh_dir == zone_dir:
            combined_conf = (thresh_conf + zone_conf) / 2
            return thresh_dir, min(combined_conf * 1.2, 1.0)
        
        # If they disagree, use weighted combination
        if thresh_conf > zone_conf * 1.3:
            return thresh_dir, thresh_conf * 0.9
        elif zone_conf > thresh_conf * 1.3:
            return zone_dir, zone_conf * 0.9
        else:
            # Very close, use priority system
            for priority_dir in self.settings.DIRECTION_PRIORITY:
                if thresh_dir.value == priority_dir:
                    return thresh_dir, thresh_conf * 0.8
                if zone_dir.value == priority_dir:
                    return zone_dir, zone_conf * 0.8
            
            return thresh_dir, thresh_conf * 0.7
    
    def reset(self):
        """Reset detector state"""
        if self.kalman:
            self.kalman.reset()
        self.angle_history.clear()
        self.magnitude_history.clear()
        self.direction_scores = {d: 0.0 for d in GazeDirection}


# ==================== MOUSE CONTROLLER ====================
class PreciseMouseController:
    """Precise mouse control with per-direction speed settings and movement delay"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mouse = mouse.Controller()
        self.is_active = settings.ENABLE_MOUSE_CONTROL
        
        # Movement state
        self.current_velocity = np.array([0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0])
        self.movement_duration = 0
        self.last_direction = GazeDirection.CENTER
        
        # Movement delay tracking (NEW)
        self.last_direction_change_time = time.time()
        self.current_direction_start_time = time.time()
        self.movement_cooldown = False
        self.cooldown_end_time = 0
        
        # Direction vectors with calibrated speeds
        self.direction_vectors = self._create_direction_vectors()
        
        # Threading
        self.movement_queue = queue.Queue()
        self.movement_thread = None
        self.running = False
        
        if settings.USE_THREADING:
            self.start_movement_thread()
    
    def _create_direction_vectors(self) -> Dict[GazeDirection, np.ndarray]:
        """Create movement vectors for each direction"""
        vectors = {}
        base = self.settings.MOUSE_SPEED_BASE
        
        for direction in GazeDirection:
            mult = self.settings.MOUSE_SPEED_MULTIPLIERS.get(direction.value, 1.0)
            
            if direction == GazeDirection.CENTER:
                vectors[direction] = np.array([0, 0])
            elif direction == GazeDirection.LEFT:
                vectors[direction] = np.array([-base * mult, 0])
            elif direction == GazeDirection.RIGHT:
                vectors[direction] = np.array([base * mult, 0])
            elif direction == GazeDirection.UP:
                vectors[direction] = np.array([0, -base * mult])
            elif direction == GazeDirection.DOWN:
                vectors[direction] = np.array([0, base * mult])
            elif direction == GazeDirection.LEFT_UP:
                vectors[direction] = np.array([-base * mult * 0.707, -base * mult * 0.707])
            elif direction == GazeDirection.RIGHT_UP:
                vectors[direction] = np.array([base * mult * 0.707, -base * mult * 0.707])
            elif direction == GazeDirection.LEFT_DOWN:
                vectors[direction] = np.array([-base * mult * 0.707, base * mult * 0.707])
            elif direction == GazeDirection.RIGHT_DOWN:
                vectors[direction] = np.array([base * mult * 0.707, base * mult * 0.707])
        
        return vectors
    
    def start_movement_thread(self):
        """Start background thread for smooth movement"""
        self.running = True
        self.movement_thread = threading.Thread(target=self._movement_worker, daemon=True)
        self.movement_thread.start()
    
    def _movement_worker(self):
        """Background worker for smooth mouse movement"""
        while self.running:
            try:
                command = self.movement_queue.get(timeout=0.01)
                if command:
                    direction, confidence = command
                    self._update_velocity(direction, confidence)
                    
            except queue.Empty:
                pass
            
            # Always apply current velocity
            self._apply_velocity()
            
            # Decay velocity when no input
            if np.linalg.norm(self.target_velocity) < 0.1:
                self.current_velocity *= 0.92
                if np.linalg.norm(self.current_velocity) < 0.5:
                    self.current_velocity = np.array([0.0, 0.0])
            
            time.sleep(0.01)  # 100 Hz update
    
    def move_mouse(self, direction: GazeDirection, confidence: float = 1.0):
        """Queue mouse movement command"""
        if not self.is_active:
            return
            
        if self.settings.USE_THREADING:
            self.movement_queue.put((direction, confidence))
        else:
            self._update_velocity(direction, confidence)
            self._apply_velocity()
    
    def _update_velocity(self, direction: GazeDirection, confidence: float):
        """Update target velocity based on direction and confidence with movement delay"""
        current_time = time.time()
        
        # Check if we're in cooldown
        if self.movement_cooldown:
            if current_time < self.cooldown_end_time:
                # Still in cooldown, reduce velocity
                self.target_velocity = np.array([0.0, 0.0])
                return
            else:
                # Cooldown ended
                self.movement_cooldown = False
        
        # Check for direction change
        if direction != self.last_direction:
            # Check if we've held the current direction long enough
            time_in_direction = current_time - self.current_direction_start_time
            
            # Check if we need to go through CENTER
            if self.settings.REQUIRE_CENTER_BETWEEN:
                if self.last_direction != GazeDirection.CENTER and direction != GazeDirection.CENTER:
                    # Need to go through center first
                    self.target_velocity = np.array([0.0, 0.0])
                    return
            
            # Check minimum hold time
            if time_in_direction < self.settings.MOVEMENT_HOLD_TIME:
                # Haven't held current direction long enough, ignore change
                return
            
            # Check diagonal transition rules
            if not self.settings.ALLOW_DIAGONAL_TRANSITION:
                is_last_diagonal = self.last_direction in [
                    GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                    GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN
                ]
                is_new_diagonal = direction in [
                    GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                    GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN
                ]
                
                if is_last_diagonal and is_new_diagonal:
                    # Diagonal to diagonal transition not allowed
                    self.target_velocity = np.array([0.0, 0.0])
                    return
            
            # Apply movement delay
            time_since_last_change = current_time - self.last_direction_change_time
            if time_since_last_change < self.settings.MOVEMENT_DELAY:
                # Too soon for another direction change
                self.movement_cooldown = True
                self.cooldown_end_time = self.last_direction_change_time + self.settings.MOVEMENT_DELAY
                self.target_velocity = np.array([0.0, 0.0])
                return
            
            # Direction change is allowed
            self.last_direction_change_time = current_time
            self.current_direction_start_time = current_time
            self.movement_duration = 0
            self.last_direction = direction
        else:
            # Same direction, increment duration
            self.movement_duration += 1
        
        # Calculate acceleration
        acceleration = min(1 + (self.movement_duration * 0.05), self.settings.MOUSE_ACCELERATION)
        
        # Get base velocity for direction
        base_velocity = self.direction_vectors[direction]
        
        # Apply confidence and acceleration
        self.target_velocity = base_velocity * confidence * acceleration
        
        # Smooth transition
        smooth = self.settings.MOUSE_SMOOTH_FACTOR
        self.current_velocity = smooth * self.current_velocity + (1 - smooth) * self.target_velocity
        
        # Limit maximum speed
        speed = np.linalg.norm(self.current_velocity)
        if speed > self.settings.MOUSE_MAX_SPEED:
            self.current_velocity = self.current_velocity / speed * self.settings.MOUSE_MAX_SPEED
    
    def _apply_velocity(self):
        """Apply current velocity to mouse position"""
        if np.linalg.norm(self.current_velocity) > 0.1:
            try:
                current_pos = self.mouse.position
                new_pos = (
                    current_pos[0] + self.current_velocity[0],
                    current_pos[1] + self.current_velocity[1]
                )
                self.mouse.position = new_pos
            except Exception as e:
                pass  # Ignore boundary errors
    
    def toggle_control(self):
        """Toggle mouse control on/off"""
        self.is_active = not self.is_active
        return self.is_active
    
    def stop(self):
        """Stop mouse controller"""
        self.running = False
        if self.movement_thread:
            self.movement_thread.join(timeout=1.0)


# ==================== TEMPORAL FILTER ====================
class AdvancedTemporalFilter:
    """Advanced temporal filtering with pattern recognition"""
    
    def __init__(self, window_size: int, confidence_threshold: float, min_consistent: int):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_consistent = min_consistent
        
        # History buffers
        self.direction_history = collections.deque(maxlen=window_size)
        self.confidence_history = collections.deque(maxlen=window_size)
        self.angle_history = collections.deque(maxlen=window_size)
        
        # State tracking
        self.current_direction = GazeDirection.CENTER
        self.consistent_count = 0
        self.last_change_time = time.time()
        
        # Pattern detection
        self.movement_patterns = []
        
    def update(self, direction: GazeDirection, confidence: float, 
               pitch: float, yaw: float) -> Tuple[GazeDirection, float]:
        """Update filter and return filtered direction"""
        # Add to history
        self.direction_history.append(direction)
        self.confidence_history.append(confidence)
        self.angle_history.append((pitch, yaw))
        
        # Track consistency
        if direction == self.current_direction:
            self.consistent_count += 1
        else:
            self.consistent_count = 1
            self.last_change_time = time.time()
        
        # Weighted voting with recency bias
        direction_weights = {}
        for d in GazeDirection:
            direction_weights[d] = 0.0
        
        # Apply exponential decay weights
        for i, (hist_dir, hist_conf) in enumerate(zip(self.direction_history, 
                                                      self.confidence_history)):
            weight = np.exp((i - len(self.direction_history)) / 3) * hist_conf
            direction_weights[hist_dir] += weight
        
        # Find best direction
        best_direction = max(direction_weights.items(), key=lambda x: x[1])
        total_weight = sum(direction_weights.values())
        filtered_confidence = best_direction[1] / total_weight if total_weight > 0 else 0
        
        # Boost confidence for consistent detection
        if self.consistent_count >= self.min_consistent:
            filtered_confidence = min(1.0, filtered_confidence * 1.3)
        
        # Apply threshold
        if filtered_confidence >= self.confidence_threshold:
            if best_direction[0] != self.current_direction:
                self.current_direction = best_direction[0]
                
                # Detect patterns
                if len(self.movement_patterns) < 100:
                    self.movement_patterns.append({
                        'from': self.current_direction,
                        'to': best_direction[0],
                        'time': time.time() - self.last_change_time
                    })
            
            return best_direction[0], filtered_confidence
        else:
            return self.current_direction, filtered_confidence
    
    def get_smoothed_angles(self) -> Tuple[float, float]:
        """Get smoothed angles"""
        if not self.angle_history:
            return 0.0, 0.0
        
        weights = np.exp(np.linspace(-2, 0, len(self.angle_history)))
        weights /= weights.sum()
        
        angles = np.array(self.angle_history)
        pitch = np.sum(angles[:, 0] * weights)
        yaw = np.sum(angles[:, 1] * weights)
        
        return pitch, yaw
    
    def reset(self):
        """Reset filter state"""
        self.direction_history.clear()
        self.confidence_history.clear()
        self.angle_history.clear()
        self.current_direction = GazeDirection.CENTER
        self.consistent_count = 0
        self.last_change_time = time.time()


# ==================== GAZE ESTIMATION MODEL ====================
class GazeEstimationONNX:
    """ONNX model for gaze estimation"""
    
    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        self.input_size = (448, 448)
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image"""
        img = cv2.resize(face_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32, copy=False) * (1.0/255.0)
        img -= self.input_mean
        img /= self.input_std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)
        
    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        """Estimate gaze angles"""
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        pitch_deg = self._compute_angle(outputs[0][0])
        yaw_deg = self._compute_angle(outputs[1][0])
        return pitch_deg, yaw_deg
        
    def _compute_angle(self, predictions: np.ndarray) -> float:
        """Compute angle from predictions"""
        exp_preds = np.exp(predictions - np.max(predictions))
        predictions = exp_preds / np.sum(exp_preds)
        angle_deg = np.sum(predictions * self.idx_tensor) * self._binwidth
        angle_deg -= self._angle_offset
        return angle_deg


# ==================== VISUALIZER ====================
class AccurateVisualizer:
    """Enhanced visualization with debug information"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.colors = {
            'center': (0, 255, 0),
            'cardinal': (255, 165, 0),
            'diagonal': (255, 105, 180),
            'face': (255, 255, 0),
            'mouse_on': (0, 255, 0),
            'mouse_off': (0, 0, 255),
            'calibrating': (0, 255, 255),
            'debug': (150, 150, 150)
        }
    
    def draw_overlay(self, frame: np.ndarray, bbox: np.ndarray,
                    direction: GazeDirection, confidence: float,
                    pitch: float, yaw: float, fps: float,
                    mouse_enabled: bool, calibrating: bool = False):
        """Draw complete overlay"""
        height, width = frame.shape[:2]
        
        # Draw face box
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            color = self.colors['calibrating'] if calibrating else self.colors['face']
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw direction indicator
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            self._draw_direction_indicator(frame, cx, cy, direction, confidence)
            
            # Draw debug info if enabled
            if self.settings.SHOW_DEBUG_INFO:
                self._draw_debug_info(frame, pitch, yaw, confidence)
        
        # Draw info panel
        self._draw_info_panel(frame, direction, confidence, pitch, yaw, fps, 
                            mouse_enabled, calibrating)
        
        # Draw instructions
        instructions = "Q:Quit | M:Mouse | C:Calibrate | R:Reset | +/-:Speed | [/]:Delay"
        if self.settings.SHOW_DEBUG_INFO:
            instructions += " | D:Debug"
        cv2.putText(frame, instructions, (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_direction_indicator(self, frame: np.ndarray, cx: int, cy: int,
                                 direction: GazeDirection, confidence: float):
        """Draw directional arrow"""
        if direction == GazeDirection.CENTER:
            cv2.circle(frame, (cx, cy), 5, self.colors['center'], -1)
            return
        
        # Determine arrow properties
        arrow_length = int(50 * confidence)
        # CORRECTED angles: positive yaw = up (screen up = negative y)
        angles = {
            GazeDirection.LEFT: 180,
            GazeDirection.RIGHT: 0,
            GazeDirection.UP: 90,      # Points upward on screen
            GazeDirection.DOWN: 270,   # Points downward on screen  
            GazeDirection.LEFT_UP: 135,    # Upper-left
            GazeDirection.RIGHT_UP: 45,    # Upper-right
            GazeDirection.LEFT_DOWN: 225,  # Lower-left
            GazeDirection.RIGHT_DOWN: 315  # Lower-right
        }
        
        angle = angles.get(direction, 0)
        angle_rad = np.radians(angle)
        
        # In screen coordinates: positive X = right, positive Y = down
        # So we need to invert Y for proper display
        end_x = cx + int(arrow_length * np.cos(angle_rad))
        end_y = cy - int(arrow_length * np.sin(angle_rad))  # Negative because screen Y is inverted
        
        is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                   GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
        color = self.colors['diagonal'] if is_diagonal else self.colors['cardinal']
        
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
    
    def _draw_info_panel(self, frame: np.ndarray, direction: GazeDirection,
                        confidence: float, pitch: float, yaw: float,
                        fps: float, mouse_enabled: bool, calibrating: bool):
        """Draw information panel"""
        panel_height = 140 if self.settings.SHOW_DEBUG_INFO else 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Direction text
        is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                   GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
        
        if calibrating:
            color = self.colors['calibrating']
        elif direction == GazeDirection.CENTER:
            color = self.colors['center']
        elif is_diagonal:
            color = self.colors['diagonal']
        else:
            color = self.colors['cardinal']
        
        status = "CALIBRATING" if calibrating else direction.value
        cv2.putText(frame, f"Direction: {status}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence bar
        conf_width = int(100 * confidence)
        cv2.rectangle(frame, (20, 45), (20 + conf_width, 55), color, -1)
        cv2.rectangle(frame, (20, 45), (120, 55), (100, 100, 100), 1)
        cv2.putText(frame, f"{confidence:.0%}", (130, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Mouse status
        mouse_color = self.colors['mouse_on'] if mouse_enabled else self.colors['mouse_off']
        cv2.putText(frame, f"Mouse: {'ON' if mouse_enabled else 'OFF'}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2)
        
        # Delay status
        cv2.putText(frame, f"Delay: {self.settings.MOVEMENT_DELAY:.1f}s", (120, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # FPS and angles
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {pitch:+.1f}° Yaw: {yaw:+.1f}°", (120, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_debug_info(self, frame: np.ndarray, pitch: float, yaw: float, confidence: float):
        """Draw debug visualization"""
        # Draw coordinate system
        origin_x, origin_y = 100, frame.shape[0] - 100
        cv2.circle(frame, (origin_x, origin_y), 50, self.colors['debug'], 1)
        
        # Draw axis labels
        cv2.putText(frame, "L", (origin_x - 60, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['debug'], 1)
        cv2.putText(frame, "R", (origin_x + 55, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['debug'], 1)
        cv2.putText(frame, "U", (origin_x - 5, origin_y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['debug'], 1)
        cv2.putText(frame, "D", (origin_x - 5, origin_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['debug'], 1)
        
        # Draw current gaze position
        # CORRECTED: positive yaw = up = negative screen Y
        gaze_x = origin_x + int(pitch * 2)
        gaze_y = origin_y - int(yaw * 2)  # Negative because positive yaw = up on screen
        cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
        
        # Draw threshold zones
        for direction, thresholds in self.settings.THRESHOLDS.items():
            if direction == 'CENTER':
                # Draw center zone as a circle
                radius = int(self.settings.CENTER_DEAD_ZONE * 2)
                cv2.circle(frame, (origin_x, origin_y), radius, (100, 255, 100), 1)
                continue
            
            x1 = origin_x + int(thresholds['pitch_min'] * 2)
            x2 = origin_x + int(thresholds['pitch_max'] * 2)
            # CORRECTED: invert yaw for display (positive yaw = up = negative screen Y)
            y1 = origin_y - int(thresholds['yaw_max'] * 2)
            y2 = origin_y - int(thresholds['yaw_min'] * 2)
            
            # Color based on direction type
            is_diagonal = direction in ['LEFT_UP', 'RIGHT_UP', 'LEFT_DOWN', 'RIGHT_DOWN']
            zone_color = (150, 100, 150) if is_diagonal else (100, 150, 100)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 1)
            
            # Add direction label
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = direction[:2] if len(direction) <= 5 else direction[:1] + direction[-1:]
            cv2.putText(frame, label, (cx-8, cy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Add angle values
        cv2.putText(frame, f"P:{pitch:+.1f}", (origin_x + 60, origin_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Y:{yaw:+.1f}", (origin_x + 60, origin_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ==================== MAIN APPLICATION ====================
class AccurateGazeApp:
    """Main application with accurate 8-direction detection"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = AccurateGazeDetector(settings)
        self.filter = AdvancedTemporalFilter(
            settings.SMOOTHING_WINDOW,
            settings.CONFIDENCE_THRESHOLD,
            settings.MIN_CONSISTENT_FRAMES
        )
        self.visualizer = AccurateVisualizer(settings)
        self.mouse_controller = PreciseMouseController(settings)
        
        print("Loading model...")
        self.model = GazeEstimationONNX(settings.MODEL_PATH)
        
        print("Loading face detector...")
        self.face_detector = uniface.RetinaFace()
        
        # Statistics
        self.stats = {d: 0 for d in GazeDirection}
        self.frame_count = 0
        self.last_direction = GazeDirection.CENTER
        
        # FPS tracking
        self.fps_history = collections.deque(maxlen=30)
        self.last_time = time.time()
        
        # Face tracking
        self.last_bbox = None
        
        # Calibration state
        self.calibration_sequence = list(GazeDirection)
        self.calibration_index = 0
        
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(self.settings.VIDEO_SOURCE)
        if not cap.isOpened():
            raise IOError(f"Failed to open video source")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\n" + "="*60)
        print("ACCURATE 8-DIRECTION GAZE DETECTION (HYBRID)")
        print("="*60)
        print(f"Resolution: {width}x{height}")
        print(f"Kalman Filter: {'ON' if self.settings.USE_KALMAN_FILTER else 'OFF'}")
        print(f"Movement Delay: {self.settings.MOVEMENT_DELAY:.1f}s")
        print("\nControls:")
        print("  Q - Quit | M - Toggle Mouse | R - Reset")
        print("  C - Start Calibration | D - Toggle Debug")
        print("  A - Show Angle Values")
        print("  +/- - Adjust Speed | [/] - Adjust Delay")
        print("="*60 + "\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.fps_history.append(fps)
                self.last_time = current_time
                avg_fps = np.mean(self.fps_history)
                
                # Process frame
                if self.frame_count % self.settings.FRAME_SKIP == 0:
                    self.process_frame(frame, avg_fps)
                elif self.last_bbox is not None:
                    # Show last state on skipped frames
                    direction, confidence = self.last_direction, 0.5
                    pitch, yaw = self.filter.get_smoothed_angles()
                    self.visualizer.draw_overlay(
                        frame, self.last_bbox, direction, confidence,
                        pitch, yaw, avg_fps, self.mouse_controller.is_active,
                        self.detector.calibration.is_calibrating
                    )
                
                cv2.imshow("Accurate Gaze Control", frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mouse_controller.toggle_control()
                    print(f"Mouse control: {'ON' if self.mouse_controller.is_active else 'OFF'}")
                elif key == ord('r'):
                    self.detector.reset()
                    self.filter.reset()
                    print("Reset complete")
                elif key == ord('c'):
                    self.start_calibration()
                elif key == ord('d'):
                    self.settings.SHOW_DEBUG_INFO = not self.settings.SHOW_DEBUG_INFO
                elif key == ord('+'):
                    self.settings.MOUSE_SPEED_BASE = min(30, self.settings.MOUSE_SPEED_BASE + 2)
                    self.mouse_controller.direction_vectors = self.mouse_controller._create_direction_vectors()
                    print(f"Speed: {self.settings.MOUSE_SPEED_BASE}")
                elif key == ord('-'):
                    self.settings.MOUSE_SPEED_BASE = max(5, self.settings.MOUSE_SPEED_BASE - 2)
                    self.mouse_controller.direction_vectors = self.mouse_controller._create_direction_vectors()
                    print(f"Speed: {self.settings.MOUSE_SPEED_BASE}")
                elif key == ord('['):
                    self.settings.MOVEMENT_DELAY = max(0.0, self.settings.MOVEMENT_DELAY - 0.1)
                    print(f"Movement Delay: {self.settings.MOVEMENT_DELAY:.1f}s")
                elif key == ord(']'):
                    self.settings.MOVEMENT_DELAY = min(2.0, self.settings.MOVEMENT_DELAY + 0.1)
                    print(f"Movement Delay: {self.settings.MOVEMENT_DELAY:.1f}s")
                elif key == ord('h'):
                    self.settings.MOVEMENT_HOLD_TIME = max(0.0, self.settings.MOVEMENT_HOLD_TIME - 0.05)
                    print(f"Hold Time: {self.settings.MOVEMENT_HOLD_TIME:.2f}s")
                elif key == ord('j'):
                    self.settings.MOVEMENT_HOLD_TIME = min(1.0, self.settings.MOVEMENT_HOLD_TIME + 0.05)
                    print(f"Hold Time: {self.settings.MOVEMENT_HOLD_TIME:.2f}s")
                elif key == ord('a'):
                    self.settings.SHOW_ANGLE_VALUES = not self.settings.SHOW_ANGLE_VALUES
                    print(f"Angle display: {'ON' if self.settings.SHOW_ANGLE_VALUES else 'OFF'}")
                    
        finally:
            self.mouse_controller.stop()
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            self.print_statistics()
    
    def process_frame(self, frame: np.ndarray, fps: float):
        """Process single frame"""
        # Detect face
        if self.settings.FACE_DETECTION_SCALE < 1.0:
            small = cv2.resize(frame, None, 
                             fx=self.settings.FACE_DETECTION_SCALE,
                             fy=self.settings.FACE_DETECTION_SCALE)
            bboxes, _ = self.face_detector.detect(small)
            if len(bboxes) > 0:
                bboxes = bboxes / self.settings.FACE_DETECTION_SCALE
        else:
            bboxes, _ = self.face_detector.detect(frame)
        
        if len(bboxes) == 0:
            return
        
        bbox = bboxes[0]
        self.last_bbox = bbox
        
        # Extract face
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        face = frame[y_min:y_max, x_min:x_max]
        if face.size == 0:
            return
        
        # Estimate gaze
        pitch, yaw = self.model.estimate(face)
        
        # Detect direction
        direction, confidence = self.detector.detect_direction(pitch, yaw)
        
        # Apply temporal filter
        filtered_dir, filtered_conf = self.filter.update(direction, confidence, pitch, yaw)
        
        # Update stats
        self.stats[filtered_dir] += 1
        self.last_direction = filtered_dir
        
        # Display angle values if enabled
        if self.settings.SHOW_ANGLE_VALUES and self.frame_count % 10 == 0:
            print(f"Pitch: {pitch:+6.1f}° | Yaw: {yaw:+6.1f}° | Dir: {filtered_dir.value:10s} | Conf: {filtered_conf:.2f}")
        
        # Control mouse
        if self.mouse_controller.is_active and filtered_conf > self.settings.CONFIDENCE_THRESHOLD:
            self.mouse_controller.move_mouse(filtered_dir, filtered_conf)
        
        # Visualize
        self.visualizer.draw_overlay(
            frame, bbox, filtered_dir, filtered_conf,
            pitch, yaw, fps, self.mouse_controller.is_active,
            self.detector.calibration.is_calibrating
        )
    
    def start_calibration(self):
        """Start calibration sequence"""
        print("\n" + "="*40)
        print("CALIBRATION MODE")
        print("Follow the on-screen instructions")
        print("="*40 + "\n")
        
        self.calibration_index = 0
        direction = self.calibration_sequence[self.calibration_index]
        self.detector.calibration.start_calibration(direction)
    
    def print_statistics(self):
        """Print session statistics"""
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        total = sum(self.stats.values())
        if total > 0:
            for direction, count in sorted(self.stats.items(), 
                                         key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                bar = '█' * int(pct / 2)
                print(f"{direction.value:12s}: {pct:5.1f}% {bar}")
        print("="*60)


# ==================== MAIN ====================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Accurate 8-Direction Gaze Detection with Hybrid Approach")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--model", default="mobileone_s0_gaze.onnx")
    parser.add_argument("--no-mouse", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--kalman", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3, help="Movement delay in seconds")
    parser.add_argument("--hold-time", type=float, default=0.15, help="Minimum hold time before direction change")
    parser.add_argument("--center-required", action="store_true", help="Require center between direction changes")
    
    args = parser.parse_args()
    
    settings = Settings(
        VIDEO_SOURCE=args.source,
        MODEL_PATH=args.model,
        ENABLE_MOUSE_CONTROL=not args.no_mouse,
        ENABLE_CALIBRATION=args.calibrate,
        SHOW_DEBUG_INFO=args.debug,
        USE_KALMAN_FILTER=args.kalman,
        MOVEMENT_DELAY=args.delay,
        MOVEMENT_HOLD_TIME=args.hold_time,
        REQUIRE_CENTER_BETWEEN=args.center_required
    )
    
    print("\n" + "="*60)
    print("ACCURATE 8-DIRECTION GAZE DETECTION SYSTEM")
    print("="*60)
    print("Features:")
    print("  • Hybrid detection combining threshold and zone methods")
    print("  • Kalman filtering for smooth tracking")
    print("  • Calibration support for personalization")
    print("  • Advanced temporal filtering")
    print("  • Movement delay system for controlled navigation")
    print("  • Precise mouse control with per-direction speeds")
    print("\nDelay Settings:")
    print(f"  • Movement Delay: {settings.MOVEMENT_DELAY:.1f}s")
    print(f"  • Hold Time: {settings.MOVEMENT_HOLD_TIME:.2f}s")
    print(f"  • Center Required: {'Yes' if settings.REQUIRE_CENTER_BETWEEN else 'No'}")
    print("="*60 + "\n")
    
    app = AccurateGazeApp(settings)
    app.run()


if __name__ == "__main__":
    main()
