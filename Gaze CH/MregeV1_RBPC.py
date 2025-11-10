"""
Accurate 8-Direction Gaze Detection for Raspberry Pi Zero 2W
Optimized for limited resources (512MB RAM, low CPU)
Compatible with PiCamera V2.1

CHANGES FROM ORIGINAL:
- Replaced OpenCV VideoCapture with libcamera (via picamera2) for better performance
- Reduced resolution to 480p for faster processing
- Disabled Kalman filtering by default (uses too much memory)
- Simplified temporal filtering
- Reduced ONNX model input size
- Added frame skipping to improve FPS
- Disabled threading for mouse control
- Optimized memory allocation
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
import queue
from pynput import mouse
from pynput.mouse import Controller, Button
import json
import os
import mediapipe as mp
from collections import deque
import pygetwindow as gw
import platform
import pyautogui

# ============================================================================
# RASPBERRY PI SPECIFIC IMPORTS
# ============================================================================
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("WARNING: picamera2 not available, will use OpenCV fallback")

# ============================================================================
# CONSTANTS - OPTIMIZED FOR PI ZERO
# ============================================================================
EAR_CLOSE_THRESHOLD = 0.27
EAR_OPEN_THRESHOLD = 0.30
BLINK_CONSEC_FRAMES = 2
ACTION_COOLDOWN_MS = 300
SCROLL_AMOUNT = 250
num_of_clicks = 1

# PI ZERO SPECIFIC SETTINGS
PI_CAMERA_WIDTH = 640      # Reduced from typical 1280 for performance
PI_CAMERA_HEIGHT = 480     # Reduced from typical 720
PI_CAMERA_FPS = 30         # Conservative FPS for stability
FRAME_SKIP = 2             # Process every 2nd frame to reduce CPU load
MODEL_INPUT_SIZE = (320, 320)  # Reduced from 448x448 for faster inference


def now_ms():
    """Get current time in milliseconds"""
    return int(time.time() * 1000)


# ============================================================================
# EYE ASPECT RATIO CALCULATION
# ============================================================================
def eye_aspect_ratio(landmarks, eye_indices) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Args:
        landmarks: Face landmarks from MediaPipe
        eye_indices: List of eye landmark indices
    
    Returns:
        EAR value (float)
    """
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance 1
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance 2
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)


# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================================
# VIRTUAL KEYBOARD CONTROL
# ============================================================================
def is_virtual_keyboard_open():
    """Check if virtual keyboard is open (Windows only)"""
    try:
        if platform.system() == "Windows":
            import psutil
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in ['TabTip.exe', 'TextInputHost.exe']:
                    return True
        return False
    except:
        return False


def toggle_virtual_keyboard():
    """Toggle Windows virtual keyboard"""
    try:
        if platform.system() == "Windows":
            pyautogui.hotkey('ctrl', 'win', 'o')
            time.sleep(0.3)
            is_open = is_virtual_keyboard_open()
            status = "opened" if is_open else "toggled"
            print(f"✓ Touch keyboard {status}.")
            return True
        else:
            print("Virtual keyboard not supported on this OS.")
    except Exception as e:
        print(f"Error toggling keyboard: {e}")
    return False


# ============================================================================
# BLINK DETECTION CLASS
# ============================================================================
class BlinkDetection:
    """Detects single blinks, double blinks, and per-eye blinks"""
    
    def __init__(self):
        # Single blink counters
        self.left_blink_frames = 0
        self.right_blink_frames = 0
        self.both_blink_frames = 0

        # Double blink timing
        self.last_both_blink_ms = 0
        self.both_blink_count = 0
        self.double_blink_interval_ms = 500
        self.pending_single_click = False
        self.pending_click_time = 0

        # Action cooldown
        self.last_action_ms = 0
        self.action_cooldown_ms = ACTION_COOLDOWN_MS

        # Keyboard state
        self.keyboard_open = False

    def reset_blink_counters(self):
        """Reset all blink frame counters"""
        self.left_blink_frames = 0
        self.right_blink_frames = 0
        self.both_blink_frames = 0
    
    def can_trigger_action(self, now_ms):
        """Check if enough time has passed since last action"""
        return (now_ms - self.last_action_ms) >= self.action_cooldown_ms
    
    def update_blink_state(self, left_closed, right_closed, left_open, right_open, left_ear, right_ear):
        """
        Update blink detection state based on eye closure
        
        Returns:
            'both', 'left', 'right', 'both_complete', or None
        """
        if left_closed and right_closed:
            self.both_blink_frames += 1
            self.left_blink_frames = 0
            self.right_blink_frames = 0
            return 'both'
        elif left_closed and right_open:
            self.left_blink_frames += 1
            self.right_blink_frames = 0
            self.both_blink_frames = 0
            return 'left'
        elif right_closed and left_open:
            self.right_blink_frames += 1
            self.left_blink_frames = 0
            self.both_blink_frames = 0
            return 'right'
        else:
            # Eyes open - check if we just completed a blink
            had_both_blink = self.both_blink_frames >= BLINK_CONSEC_FRAMES
            self.reset_blink_counters()
            if had_both_blink:
                return 'both_complete'
            return None

    def process_double_blink(self, now_ms):
        """
        Process double blink detection with timeout logic
        
        Returns:
            'wait', 'double_blink', or 'single_click'
        """
        time_since_last_blink = now_ms - self.last_both_blink_ms

        if self.both_blink_count == 0:
            self.both_blink_count = 1
            self.last_both_blink_ms = now_ms
            self.pending_single_click = True
            self.pending_click_time = now_ms
            return 'wait'
        
        elif time_since_last_blink < self.double_blink_interval_ms:
            self.both_blink_count = 0
            self.last_both_blink_ms = 0
            self.pending_single_click = False
            return 'double_blink'
        
        else:
            self.both_blink_count = 1
            self.last_both_blink_ms = now_ms
            self.pending_single_click = True
            self.pending_click_time = now_ms
            return 'single_click'
    
    def check_pending_click(self, now_ms):
        """Check if a pending single click should be executed"""
        if self.pending_single_click:
            time_since_pending = now_ms - self.pending_click_time
            if time_since_pending >= self.double_blink_interval_ms:
                self.pending_single_click = False
                return True
        return False
    
    def handle_blink_actions(self, blink_type, landmarks, mouse_controller, now_ms):
        """
        Handle blink-based actions (mouse clicks, keyboard toggle)
        
        Returns:
            Action string or None
        """
        # Check if we have a pending single click that timed out
        if self.check_pending_click(now_ms):
            if self.can_trigger_action(now_ms):
                mouse_controller.click(Button.left, num_of_clicks)
                self.last_action_ms = now_ms
                return "left_click" 
        
        if not self.can_trigger_action(now_ms):
            return None
        
        # BOTH EYES BLINK COMPLETE
        if blink_type == 'both_complete':
            blink_result = self.process_double_blink(now_ms)
            
            if blink_result == 'double_blink':
                toggle_virtual_keyboard()
                self.keyboard_open = is_virtual_keyboard_open()
                self.last_action_ms = now_ms
                print(f"[DEBUG] Double blink detected! Keyboard state: {self.keyboard_open}")
                return "keyboard_toggle"
            
            elif blink_result == 'single_click':
                return None
            
            elif blink_result == 'wait':
                print(f"[DEBUG] First blink detected, waiting for second...")
                return None
        
        # RIGHT EYE BLINK 
        elif blink_type == 'right' and self.right_blink_frames >= BLINK_CONSEC_FRAMES:
            mouse_controller.click(Button.right, 1)
            self.last_action_ms = now_ms
            self.reset_blink_counters()
            return "right_click"
        
        # LEFT EYE BLINK
        elif blink_type == 'left' and self.left_blink_frames >= BLINK_CONSEC_FRAMES:
            toggle_virtual_keyboard()
            self.keyboard_open = is_virtual_keyboard_open()
            self.last_action_ms = now_ms
            print(f"[DEBUG] Left eye blink detected! Keyboard state: {self.keyboard_open}")
            return "keyboard_toggle"
        
        return None


# ============================================================================
# EYEBROW DETECTION FOR SCROLL MODE
# ============================================================================
def eyebrow_eye_distance(landmarks, eyebrow_idxs, eye_idxs):
    """
    Calculate vertical distance between eyebrow and eye
    
    Returns:
        Positive value when eyebrow is raised
    """
    eye_cx = np.mean([landmarks[i][0] for i in eye_idxs])
    eye_cy = np.mean([landmarks[i][1] for i in eye_idxs])
    brow_cx = np.mean([landmarks[i][0] for i in eyebrow_idxs])
    brow_cy = np.mean([landmarks[i][1] for i in eyebrow_idxs])
    return eye_cy - brow_cy


# ============================================================================
# SCROLL MODE DETECTOR
# ============================================================================
class ScrollModeDetector:
    """Detects eyebrow raises for scroll mode activation"""
    
    def __init__(self):
        self.prev_left_dist = None
        self.prev_right_dist = None
        self.scroll_mode = False
        self.last_toggle_ms = 0
        self.cooldown_ms = 800
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)
        self.abs_thresh = 4.0
        self.rel_thresh_ratio = 0.15
        self.raise_frames = 0

        
    def update(self, landmarks, now_ms, eyes_open=True):
        """
        Update scroll mode detection
        
        Returns:
            Boolean indicating if scroll mode is active
        """
        # Skip detection if eyes are closed
        if not eyes_open:
            self.raise_frames = 0
            return self.scroll_mode
        
        # Calculate eyebrow-eye distance for both sides
        left_dist = eyebrow_eye_distance(landmarks, [70, 63, 105], LEFT_EYE)
        right_dist = eyebrow_eye_distance(landmarks, [300, 293, 334], RIGHT_EYE)

        # Initialize baseline on first frame
        if self.prev_left_dist is None:
            self.prev_left_dist, self.prev_right_dist = left_dist, right_dist
            self.left_history.extend([left_dist] * 5)
            self.right_history.extend([right_dist] * 5)
            return self.scroll_mode

        self.left_history.append(left_dist)
        self.right_history.append(right_dist)
        smooth_left = np.mean(self.left_history)
        smooth_right = np.mean(self.right_history)
        
        # Calculate change in eyebrow position
        change = ((left_dist - self.prev_left_dist) +
                  (right_dist - self.prev_right_dist)) / 2
        
        # Calculate relative change (normalized by baseline)
        baseline = (self.prev_left_dist + self.prev_right_dist) / 2
        rel_change = change / baseline if baseline != 0 else 0

        # Detect sustained eyebrow raise
        if change > self.abs_thresh or rel_change > self.rel_thresh_ratio:
            self.raise_frames += 1
        else:
            self.raise_frames = 0

        # Toggle scroll mode after threshold
        if self.raise_frames >= 1 and (now_ms - self.last_toggle_ms) > self.cooldown_ms:
            self.scroll_mode = not self.scroll_mode
            self.last_toggle_ms = now_ms
            self.raise_frames = 0
            print(f"Scroll mode toggled: {'ON' if self.scroll_mode else 'OFF'}")

        # Update stored distances
        self.prev_left_dist, self.prev_right_dist = smooth_left, smooth_right
        return self.scroll_mode


# ============================================================================
# SETTINGS - OPTIMIZED FOR PI ZERO
# ============================================================================
@dataclass
class Settings:
    """Configuration settings - optimized for Raspberry Pi Zero 2W"""
    
    # Video/Camera Settings - PI ZERO SPECIFIC
    VIDEO_SOURCE: int = 0
    MODEL_PATH: str = "mobileone_s0_gaze.onnx"
    OUTPUT_VIDEO: Optional[str] = None
    USE_PICAMERA: bool = True  # NEW: Use libcamera instead of OpenCV
    
    # Performance Settings - REDUCED FOR PI ZERO
    FACE_DETECTION_SCALE: float = 0.5
    USE_THREADING: bool = False  # DISABLED: Threading causes slowdown on Pi Zero
    
    # Gaze Detection Thresholds
    THRESHOLDS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'LEFT': {'pitch_min': -35, 'pitch_max': -10, 'yaw_min': -12, 'yaw_max': 12},
        'RIGHT': {'pitch_min': 10, 'pitch_max': 35, 'yaw_min': -12, 'yaw_max': 12},
        'UP': {'pitch_min': 12, 'pitch_max': -12, 'yaw_min': 10, 'yaw_max': 35},
        'DOWN': {'pitch_min': 12, 'pitch_max': -12, 'yaw_min': -35, 'yaw_max': -10},
        'LEFT_UP': {'pitch_min': -35, 'pitch_max': -5, 'yaw_min': 5, 'yaw_max': 35},
        'RIGHT_UP': {'pitch_min': 5, 'pitch_max': 35, 'yaw_min': 5, 'yaw_max': 35},
        'LEFT_DOWN': {'pitch_min': -35, 'pitch_max': -5, 'yaw_min': -35, 'yaw_max': -5},
        'RIGHT_DOWN': {'pitch_min': 5, 'pitch_max': 35, 'yaw_min': -35, 'yaw_max': -5},
        'CENTER': {'pitch_min': -10, 'pitch_max': 10, 'yaw_min': -10, 'yaw_max': 10}
    })
    
    # Zone-based detection parameters
    ZONE_OVERLAP: float = 0.2
    ZONE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'distance': 0.4,
        'magnitude': 0.3,
        'consistency': 0.3
    })
    
    # Priority system
    DIRECTION_PRIORITY: List[str] = field(default_factory=lambda: [
        'CENTER',
        'LEFT_UP', 'RIGHT_UP', 'LEFT_DOWN', 'RIGHT_DOWN',
        'UP', 'DOWN', 'LEFT', 'RIGHT'
    ])
    
    # Calibration Settings
    ENABLE_CALIBRATION: bool = False
    CALIBRATION_SAMPLES: int = 30
    CALIBRATION_FILE: str = "gaze_calibration.json"
    AUTO_CALIBRATE: bool = False
    
    # Temporal Smoothing - REDUCED FOR PI ZERO
    SMOOTHING_WINDOW: int = 5  # Reduced from 7
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_CONSISTENT_FRAMES: int = 3
    USE_KALMAN_FILTER: bool = False  # DISABLED: Memory intensive
    
    # Dead Zone and Activation
    CENTER_DEAD_ZONE: float = 6.0
    MIN_GAZE_MAGNITUDE: float = 8.0
    DIAGONAL_MAGNITUDE_BOOST: float = 1.0
    
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
    
    # Movement Delay Settings
    MOVEMENT_DELAY: float = 0.3
    MOVEMENT_HOLD_TIME: float = 0.15
    ALLOW_DIAGONAL_TRANSITION: bool = True
    REQUIRE_CENTER_BETWEEN: bool = False
    
    # Display Settings - MINIMAL FOR PI ZERO
    SHOW_VISUALIZATION: bool = True
    SHOW_DEBUG_INFO: bool = False
    SHOW_CALIBRATION_GUIDE: bool = True
    SHOW_CONFIDENCE_MAP: bool = False
    SHOW_FPS: bool = True
    SHOW_ANGLE_VALUES: bool = False
    LOG_CHANGES: bool = False
    
    # Advanced Settings
    RESET_AFTER_NO_FACE: int = 10
    USE_FACE_ORIENTATION: bool = True
    BLINK_DETECTION: bool = True  # ENABLED: Blink detection is lightweight


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


# ============================================================================
# SIMPLE TEMPORAL FILTER - REDUCED FOR PI ZERO
# ============================================================================
class SimpleTemporalFilter:
    """Simplified temporal filter for resource-constrained devices"""
    
    def __init__(self, window_size: int, confidence_threshold: float, min_consistent: int):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_consistent = min_consistent
        
        # History buffers - REDUCED SIZE for Pi Zero
        self.direction_history = collections.deque(maxlen=window_size)
        self.confidence_history = collections.deque(maxlen=window_size)
        self.angle_history = collections.deque(maxlen=window_size)
        
        # State tracking
        self.current_direction = GazeDirection.CENTER
        self.consistent_count = 0
        self.last_change_time = time.time()

    def update(self, direction: GazeDirection, confidence: float, 
               pitch: float, yaw: float) -> Tuple[GazeDirection, float]:
        """
        Update filter and return smoothed direction
        """
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
        
        # Simple voting with recency bias
        direction_weights = {d: 0.0 for d in GazeDirection}
        
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
            
            return best_direction[0], filtered_confidence
        else:
            return self.current_direction, filtered_confidence
    
    def get_smoothed_angles(self) -> Tuple[float, float]:
        """Get smoothed angles from history"""
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


# ============================================================================
# GAZE DETECTOR - SIMPLIFIED FOR PI ZERO
# ============================================================================
class GazeDetector:
    """Simplified gaze direction detection (threshold-based only)"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.angle_history = collections.deque(maxlen=20)
        self.magnitude_history = collections.deque(maxlen=20)

    def detect_direction(self, pitch_deg: float, yaw_deg: float) -> Tuple[GazeDirection, float]:
        """
        Detect gaze direction from pitch and yaw angles
        
        Args:
            pitch_deg: Horizontal angle (negative=left, positive=right)
            yaw_deg: Vertical angle (positive=up, negative=down)
        
        Returns:
            Tuple of (direction, confidence)
        """
        # Store history
        self.angle_history.append((pitch_deg, yaw_deg))
        magnitude = np.sqrt(pitch_deg**2 + yaw_deg**2)
        self.magnitude_history.append(magnitude)
        
        # Check center first
        if magnitude < self.settings.CENTER_DEAD_ZONE:
            return GazeDirection.CENTER, 1.0
        
        # Check minimum activation
        if magnitude < self.settings.MIN_GAZE_MAGNITUDE:
            return GazeDirection.CENTER, 0.7
        
        # Calculate scores for each direction
        scores = {}
        
        for direction in GazeDirection:
            if direction == GazeDirection.CENTER:
                continue
            
            thresholds = self.settings.THRESHOLDS[direction.value]
            
            # Check if angles fall within thresholds
            pitch_match = thresholds['pitch_min'] <= pitch_deg <= thresholds['pitch_max']
            yaw_match = thresholds['yaw_min'] <= yaw_deg <= thresholds['yaw_max']
            
            if pitch_match and yaw_match:
                # Calculate centering score
                pitch_center = (thresholds['pitch_min'] + thresholds['pitch_max']) / 2
                yaw_center = (thresholds['yaw_min'] + thresholds['yaw_max']) / 2
                
                pitch_range = thresholds['pitch_max'] - thresholds['pitch_min']
                yaw_range = thresholds['yaw_max'] - thresholds['yaw_min']
                
                pitch_score = 1.0 - abs(pitch_deg - pitch_center) / (pitch_range / 2) if pitch_range > 0 else 0
                yaw_score = 1.0 - abs(yaw_deg - yaw_center) / (yaw_range / 2) if yaw_range > 0 else 0
                
                # Combine scores
                is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP, 
                                           GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
                
                if is_diagonal:
                    scores[direction] = pitch_score * yaw_score * 1.5
                else:
                    scores[direction] = (pitch_score + yaw_score) / 2
        
        if not scores:
            return GazeDirection.CENTER, 0.5
        
        best_direction = max(scores.items(), key=lambda x: x[1])
        return best_direction[0], best_direction[1]
    
    def reset(self):
        """Reset detector state"""
        self.angle_history.clear()
        self.magnitude_history.clear()


# ============================================================================
# MOUSE CONTROLLER - NO THREADING FOR PI ZERO
# ============================================================================
class MouseController:
    """Simple mouse control without threading"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mouse = mouse.Controller()
        self.is_active = settings.ENABLE_MOUSE_CONTROL
        
        # Movement state
        self.current_velocity = np.array([0.0, 0.0])
        self.last_direction = GazeDirection.CENTER
        self.last_direction_change_time = time.time()
        self.current_direction_start_time = time.time()
        
        # Direction vectors
        self.direction_vectors = self._create_direction_vectors()
    
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
    
    def move_mouse(self, direction: GazeDirection, confidence: float = 1.0):
        """
        Move mouse based on gaze direction (NO THREADING for Pi Zero)
        """
        if not self.is_active or direction == GazeDirection.CENTER:
            return
        
        # Get velocity for direction
        velocity = self.direction_vectors[direction] * confidence
        
        # Apply smoothing
        smooth = self.settings.MOUSE_SMOOTH_FACTOR
        self.current_velocity = smooth * self.current_velocity + (1 - smooth) * velocity
        
        # Limit max speed
        speed = np.linalg.norm(self.current_velocity)
        if speed > self.settings.MOUSE_MAX_SPEED:
            self.current_velocity = self.current_velocity / speed * self.settings.MOUSE_MAX_SPEED
        
        # Apply movement
        if np.linalg.norm(self.current_velocity) > 0.1:
            try:
                current_pos = self.mouse.position
                new_pos = (
                    current_pos[0] + self.current_velocity[0],
                    current_pos[1] + self.current_velocity[1]
                )
                self.mouse.position = new_pos
            except:
                pass
    
    def toggle_control(self):
        """Toggle mouse control on/off"""
        self.is_active = not self.is_active
        return self.is_active


# ============================================================================
# ONNX GAZE ESTIMATION - REDUCED INPUT SIZE
# ============================================================================
class GazeEstimationONNX:
    """ONNX model for gaze estimation - optimized for Pi Zero"""
    
    def __init__(self, model_path: str):
        """Initialize ONNX model with Pi Zero optimizations"""
        options = ort.SessionOptions()
        # PI ZERO SPECIFIC: Reduced thread count
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        
        # Angle binning
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        # INPUT SIZE REDUCED FOR PI ZERO
        self.input_size = MODEL_INPUT_SIZE  # 320x320 instead of 448x448
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for model inference"""
        img = cv2.resize(face_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32, copy=False) * (1.0/255.0)
        img -= self.input_mean
        img /= self.input_std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)
        
    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        """Estimate gaze angles from face image"""
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        pitch_deg = self._compute_angle(outputs[0][0])
        yaw_deg = self._compute_angle(outputs[1][0])
        return pitch_deg, yaw_deg
        
    def _compute_angle(self, predictions: np.ndarray) -> float:
        """Compute angle from model predictions"""
        exp_preds = np.exp(predictions - np.max(predictions))
        predictions = exp_preds / np.sum(exp_preds)
        angle_deg = np.sum(predictions * self.idx_tensor) * self._binwidth
        angle_deg -= self._angle_offset
        return angle_deg


# ============================================================================
# SIMPLE VISUALIZER - MINIMAL FOR PI ZERO
# ============================================================================
class SimpleVisualizer:
    """Minimal visualization for Pi Zero (reduced overhead)"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.colors = {
            'center': (0, 255, 0),
            'cardinal': (255, 165, 0),
            'diagonal': (255, 105, 180),
            'face': (255, 255, 0),
        }
    
    def draw_overlay(self, frame: np.ndarray, bbox: np.ndarray,
                    direction: GazeDirection, confidence: float,
                    pitch: float, yaw: float, fps: float):
        """Draw minimal overlay (blink info only)"""
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.colors['face'], 2)
        
        # Draw info panel
        cv2.putText(frame, f"Dir: {direction.value} | Conf: {confidence:.0%}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ============================================================================
# PICAMERA2 WRAPPER - FOR RASPBERRY PI ZERO
# ============================================================================
class PiCameraStream:
    """Wrapper for PiCamera V2.1 using libcamera (picamera2)"""
    
    def __init__(self, width=PI_CAMERA_WIDTH, height=PI_CAMERA_HEIGHT, fps=PI_CAMERA_FPS):
        """
        Initialize PiCamera stream
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        if not PI_CAMERA_AVAILABLE:
            raise RuntimeError("picamera2 not available. Install: pip install picamera2")
        
        self.picam2 = Picamera2()
        
        # Configure camera with Pi Zero optimizations
        config = self.picam2.create_preview_configuration(
            main={"format": 'BGR888', "size": (width, height)},
            controls={"FrameRate": fps}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print(f"PiCamera started: {width}x{height} @ {fps}fps")
    
    def read(self):
        """
        Read frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        try:
            frame = self.picam2.capture_array()
            return True, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        self.picam2.stop()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class GazeApp:
    """Main application for gaze detection on Pi Zero"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize detectors
        self.gaze_detector = GazeDetector(settings)
        self.filter = SimpleTemporalFilter(
            settings.SMOOTHING_WINDOW,
            settings.CONFIDENCE_THRESHOLD,
            settings.MIN_CONSISTENT_FRAMES
        )
        self.visualizer = SimpleVisualizer(settings)
        self.mouse_controller = MouseController(settings)
        
        # Initialize blink/scroll detection
        self.blink_detector = BlinkDetection()
        self.scroll_detector = ScrollModeDetector()
        self.scroll_mode_active = False
        
        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load ONNX model
        print("Loading gaze model...")
        self.model = GazeEstimationONNX(settings.MODEL_PATH)
        
        # Load face detector
        print("Loading face detector...")
        self.face_detector = uniface.RetinaFace()
        
        # Statistics
        self.stats = {d: 0 for d in GazeDirection}
        self.frame_count = 0
        self.last_direction = GazeDirection.CENTER
        self.last_bbox = None
        
        # FPS tracking
        self.fps_history = collections.deque(maxlen=30)
        self.last_time = time.time()
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        if self.settings.USE_PICAMERA and PI_CAMERA_AVAILABLE:
            print("Using PiCamera (libcamera)...")
            cap = PiCameraStream(PI_CAMERA_WIDTH, PI_CAMERA_HEIGHT, PI_CAMERA_FPS)
        else:
            print("Using OpenCV camera fallback...")
            cap = cv2.VideoCapture(self.settings.VIDEO_SOURCE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("GAZE DETECTION FOR RASPBERRY PI ZERO 2W")
        print("="*60)
        print(f"Camera: {'PiCamera V2.1' if self.settings.USE_PICAMERA else 'USB/Default'}")
        print(f"Resolution: {PI_CAMERA_WIDTH}x{PI_CAMERA_HEIGHT}")
        print(f"Frame Skip: {FRAME_SKIP} (process every {FRAME_SKIP} frames)")
        print(f"Model Input: {MODEL_INPUT_SIZE}")
        print(f"Mouse Control: {'ON' if self.settings.ENABLE_MOUSE_CONTROL else 'OFF'}")
        print("\nControls:")
        print("  Q - Quit | M - Toggle Mouse | R - Reset | D - Debug")
        print("="*60 + "\n")
        
        try:
            while True:
                # Capture frame
                if self.settings.USE_PICAMERA and PI_CAMERA_AVAILABLE:
                    ret, frame = cap.read()
                else:
                    ret, frame = cap.read()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Frame skipping for better FPS
                if self.frame_count % FRAME_SKIP != 0:
                    # Display last detection
                    if self.last_bbox is not None:
                        direction, confidence = self.last_direction, 0.5
                        pitch, yaw = self.filter.get_smoothed_angles()
                        
                        self.visualizer.draw_overlay(
                            frame, self.last_bbox, direction, confidence,
                            pitch, yaw, np.mean(self.fps_history) if self.fps_history else 0
                        )
                    
                    cv2.imshow("Gaze Detection - Pi Zero", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.fps_history.append(fps)
                self.last_time = current_time
                avg_fps = np.mean(self.fps_history)
                
                # Process frame
                self.process_frame(frame, avg_fps)
                
                # Display
                cv2.imshow("Gaze Detection - Pi Zero", frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mouse_controller.toggle_control()
                    print(f"Mouse: {'ON' if self.mouse_controller.is_active else 'OFF'}")
                elif key == ord('r'):
                    self.gaze_detector.reset()
                    self.filter.reset()
                    print("Reset complete")
                elif key == ord('d'):
                    self.settings.SHOW_DEBUG_INFO = not self.settings.SHOW_DEBUG_INFO
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_statistics()
    
    def process_frame(self, frame: np.ndarray, fps: float):
        """Process single frame for gaze detection"""
        
        # Initialize variables
        left_closed = right_closed = False
        left_ear = right_ear = 0.0
        landmarks = None
        
        # MediaPipe face mesh for blink detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Calculate eye aspect ratios
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            
            # Determine eye states
            left_closed = left_ear < EAR_CLOSE_THRESHOLD
            right_closed = right_ear < EAR_CLOSE_THRESHOLD
            left_open = left_ear > EAR_OPEN_THRESHOLD
            right_open = right_ear > EAR_OPEN_THRESHOLD
            
            eyes_open = not (left_closed and right_closed)
            
            # Update blink state
            blink_type = self.blink_detector.update_blink_state(
                left_closed, right_closed, left_open, right_open, left_ear, right_ear
            )
            
            # Process blink actions
            now = now_ms()
            action = self.blink_detector.handle_blink_actions(
                blink_type, landmarks, self.mouse_controller.mouse, now
            )
            
            # Update scroll mode
            self.scroll_mode_active = self.scroll_detector.update(landmarks, now, eyes_open=eyes_open)
            
            # Visual feedback
            if action:
                print(f"Action: {action}")
        
        # Face detection
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
        
        # Extract face region
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
        direction, confidence = self.gaze_detector.detect_direction(pitch, yaw)
        
        # Apply temporal filtering
        filtered_dir, filtered_conf = self.filter.update(direction, confidence, pitch, yaw)
        
        # Update statistics
        self.stats[filtered_dir] += 1
        self.last_direction = filtered_dir
        
        # Move mouse
        if (self.mouse_controller.is_active and 
            filtered_conf > self.settings.CONFIDENCE_THRESHOLD):
            self.mouse_controller.move_mouse(filtered_dir, filtered_conf)
        
        # Draw overlay
        self.visualizer.draw_overlay(
            frame, bbox, filtered_dir, filtered_conf,
            pitch, yaw, fps
        )
        
        # Draw blink info
        if landmarks is not None:
            for idx in LEFT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
            for idx in RIGHT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (0, 0, 255), -1)
        
        cv2.putText(frame, f"L-EAR: {left_ear:.2f} | R-EAR: {right_ear:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
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


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gaze Detection for Pi Zero 2W")
    parser.add_argument("--source", type=int, default=0, help="Camera source")
    parser.add_argument("--model", default="mobileone_s0_gaze.onnx", help="ONNX model path")
    parser.add_argument("--no-mouse", action="store_true", help="Disable mouse control")
    parser.add_argument("--no-picamera", action="store_true", help="Use OpenCV instead of libcamera")
    parser.add_argument("--debug", action="store_true", help="Enable debug info")
    
    args = parser.parse_args()
    
    settings = Settings(
        VIDEO_SOURCE=args.source,
        MODEL_PATH=args.model,
        ENABLE_MOUSE_CONTROL=not args.no_mouse,
        USE_PICAMERA=not args.no_picamera,
        SHOW_DEBUG_INFO=args.debug
    )
    
    app = GazeApp(settings)
    app.run()


if __name__ == "__main__":
    main()
