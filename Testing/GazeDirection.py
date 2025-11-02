"""
Optimized Gaze Direction Detection with Mouse Control
Fixed Up/Down directions and improved FPS performance
Mouse control via pynput library
"""

import cv2
import numpy as np
import onnxruntime as ort
import uniface
import collections
from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import time
import threading
import queue
from pynput import mouse

# ==================== SETTINGS ====================
@dataclass
class Settings:
    """Centralized configuration for gaze detection and mouse control"""
    
    # Video/Camera Settings
    VIDEO_SOURCE: int = 0  # 0 for webcam, or path to video file
    MODEL_PATH: str = "mobileone_s0_gaze.onnx"  # Path to ONNX model
    OUTPUT_VIDEO: Optional[str] = None  # Path to save output video (None to skip)
    
    # Performance Settings
    FRAME_SKIP: int = 2  # Process every Nth frame for better FPS (1 = no skip)
    FACE_DETECTION_SCALE: float = 0.5  # Scale factor for face detection (0.5 = half resolution)
    USE_THREADING: bool = True  # Use threading for mouse control
    
    # Direction Detection Thresholds (in degrees)
    HORIZONTAL_THRESHOLD: float = 12.0  # Degrees to trigger LEFT/RIGHT
    VERTICAL_THRESHOLD: float = 10.0    # Degrees to trigger UP/DOWN
    
    # Dead Zone Settings (in degrees)
    DEAD_ZONE_HORIZONTAL: float = 8.0   # Dead zone for left/right
    DEAD_ZONE_VERTICAL: float = 6.0     # Dead zone for up/down
    
    # Temporal Smoothing (reduced for better FPS)
    SMOOTHING_WINDOW: int = 5          # Reduced from 12 for faster response
    CONFIDENCE_THRESHOLD: float = 0.6  # Minimum confidence to accept direction
    MIN_CONSISTENT_FRAMES: int = 2     # Minimum frames for direction confirmation
    
    # Mouse Control Settings
    ENABLE_MOUSE_CONTROL: bool = True   # Enable/disable mouse movement
    MOUSE_SPEED_HORIZONTAL: int = 15    # Pixels per frame for horizontal movement
    MOUSE_SPEED_VERTICAL: int = 15      # Pixels per frame for vertical movement
    MOUSE_ACCELERATION: float = 1.5      # Acceleration factor for continuous movement
    MOUSE_MAX_SPEED: int = 50           # Maximum mouse speed
    MOUSE_SMOOTH_FACTOR: float = 0.3    # Smoothing factor (0-1, higher = smoother)
    
    # Display Settings (can disable for better FPS)
    SHOW_VISUALIZATION: bool = True
    SHOW_ZONES: bool = False  # Disabled by default for better FPS
    SHOW_ANGLES: bool = True
    SHOW_ANGLE_BARS: bool = True
    SHOW_FPS: bool = True
    LOG_CHANGES: bool = False  # Disabled by default for better FPS
    
    # Advanced Settings
    RESET_AFTER_NO_FACE: int = 10


# ==================== DIRECTION ENUM ====================
class GazeDirection(Enum):
    """4 primary gaze directions + center"""
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


# ==================== MOUSE CONTROLLER ====================
class MouseController:
    """Handles mouse movement based on gaze direction"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mouse = mouse.Controller()
        self.is_active = settings.ENABLE_MOUSE_CONTROL
        
        # Movement state
        self.current_speed_x = 0
        self.current_speed_y = 0
        self.movement_duration = 0
        self.last_direction = GazeDirection.CENTER
        
        # Threading for smooth movement
        self.movement_queue = queue.Queue()
        self.movement_thread = None
        self.running = False
        
        if settings.USE_THREADING:
            self.start_movement_thread()
    
    def start_movement_thread(self):
        """Start background thread for smooth mouse movement"""
        self.running = True
        self.movement_thread = threading.Thread(target=self._movement_worker, daemon=True)
        self.movement_thread.start()
    
    def _movement_worker(self):
        """Background worker for smooth mouse movement"""
        while self.running:
            try:
                # Get movement command with timeout
                direction = self.movement_queue.get(timeout=0.01)
                
                if direction is not None:
                    self._execute_movement(direction)
                    
            except queue.Empty:
                # Continue moving if we have momentum
                if self.current_speed_x != 0 or self.current_speed_y != 0:
                    self._apply_movement()
                    # Gradually slow down when no input
                    self.current_speed_x *= 0.9
                    self.current_speed_y *= 0.9
                    
                    if abs(self.current_speed_x) < 1:
                        self.current_speed_x = 0
                    if abs(self.current_speed_y) < 1:
                        self.current_speed_y = 0
                        
            time.sleep(0.01)  # 100 Hz update rate
    
    def move_mouse(self, direction: GazeDirection, confidence: float = 1.0):
        """Move mouse based on gaze direction"""
        if not self.is_active:
            return
            
        if self.settings.USE_THREADING:
            # Queue movement for smooth execution
            self.movement_queue.put(direction)
        else:
            # Direct movement
            self._execute_movement(direction)
    
    def _execute_movement(self, direction: GazeDirection):
        """Execute mouse movement for given direction"""
        # Track consecutive movements for acceleration
        if direction == self.last_direction and direction != GazeDirection.CENTER:
            self.movement_duration += 1
        else:
            self.movement_duration = 0
            self.last_direction = direction
        
        # Calculate acceleration
        acceleration = min(1 + (self.movement_duration * 0.1), self.settings.MOUSE_ACCELERATION)
        
        # Calculate target speeds
        target_speed_x = 0
        target_speed_y = 0
        
        if direction == GazeDirection.LEFT:
            target_speed_x = -self.settings.MOUSE_SPEED_HORIZONTAL * acceleration
        elif direction == GazeDirection.RIGHT:
            target_speed_x = self.settings.MOUSE_SPEED_HORIZONTAL * acceleration
        elif direction == GazeDirection.UP:
            target_speed_y = -self.settings.MOUSE_SPEED_VERTICAL * acceleration
        elif direction == GazeDirection.DOWN:
            target_speed_y = self.settings.MOUSE_SPEED_VERTICAL * acceleration
        
        # Apply smoothing
        smooth = self.settings.MOUSE_SMOOTH_FACTOR
        self.current_speed_x = smooth * self.current_speed_x + (1 - smooth) * target_speed_x
        self.current_speed_y = smooth * self.current_speed_y + (1 - smooth) * target_speed_y
        
        # Limit maximum speed
        self.current_speed_x = np.clip(self.current_speed_x, 
                                       -self.settings.MOUSE_MAX_SPEED, 
                                       self.settings.MOUSE_MAX_SPEED)
        self.current_speed_y = np.clip(self.current_speed_y, 
                                       -self.settings.MOUSE_MAX_SPEED, 
                                       self.settings.MOUSE_MAX_SPEED)
        
        # Apply movement
        self._apply_movement()
    
    def _apply_movement(self):
        """Apply the current speed to mouse position"""
        if abs(self.current_speed_x) > 0.5 or abs(self.current_speed_y) > 0.5:
            try:
                current_x, current_y = self.mouse.position
                new_x = current_x + self.current_speed_x
                new_y = current_y + self.current_speed_y
                self.mouse.position = (new_x, new_y)
            except Exception as e:
                print(f"Mouse movement error: {e}")
    
    def toggle_control(self):
        """Toggle mouse control on/off"""
        self.is_active = not self.is_active
        status = "enabled" if self.is_active else "disabled"
        print(f"\nMouse control {status}")
        return self.is_active
    
    def stop(self):
        """Stop mouse controller"""
        self.running = False
        if self.movement_thread:
            self.movement_thread.join(timeout=1.0)


# ==================== GAZE DETECTOR ====================
class OptimizedGazeDetector:
    """Optimized gaze detection for better FPS"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Simple moving average for smoothing (faster than Kalman)
        self.pitch_history = collections.deque(maxlen=3)
        self.yaw_history = collections.deque(maxlen=3)
        
    def detect_direction(self, pitch_deg: float, yaw_deg: float) -> GazeDirection:
        """
        Determine gaze direction from pitch and yaw angles.
        FIXED: Corrected up/down mapping
        
        Args:
            pitch_deg: Pitch angle in degrees (negative = left, positive = right)
            yaw_deg: Yaw angle in degrees (positive = up, negative = down) - FIXED
            
        Returns:
            GazeDirection enum (CENTER, LEFT, RIGHT, UP, or DOWN)
        """
        # Quick smoothing with moving average
        self.pitch_history.append(pitch_deg)
        self.yaw_history.append(yaw_deg)
        
        if len(self.pitch_history) > 1:
            pitch_deg = np.mean(self.pitch_history)
            yaw_deg = np.mean(self.yaw_history)
        
        abs_pitch = abs(pitch_deg)
        abs_yaw = abs(yaw_deg)
        
        # Check if within dead zone (looking center)
        if (abs_pitch < self.settings.DEAD_ZONE_HORIZONTAL and 
            abs_yaw < self.settings.DEAD_ZONE_VERTICAL):
            return GazeDirection.CENTER
        
        # Calculate relative strengths
        pitch_strength = max(0, abs_pitch - self.settings.DEAD_ZONE_HORIZONTAL)
        yaw_strength = max(0, abs_yaw - self.settings.DEAD_ZONE_VERTICAL)
        
        # Determine dominant direction
        if pitch_strength > yaw_strength * 1.2:  # Horizontal dominance
            if abs_pitch > self.settings.HORIZONTAL_THRESHOLD:
                return GazeDirection.LEFT if pitch_deg < 0 else GazeDirection.RIGHT
        elif yaw_strength > pitch_strength * 1.2:  # Vertical dominance
            if abs_yaw > self.settings.VERTICAL_THRESHOLD:
                # FIXED: Reversed the up/down mapping
                return GazeDirection.DOWN if yaw_deg < 0 else GazeDirection.UP
        else:
            # When similar, choose based on threshold
            if abs_pitch > self.settings.HORIZONTAL_THRESHOLD:
                return GazeDirection.LEFT if pitch_deg < 0 else GazeDirection.RIGHT
            elif abs_yaw > self.settings.VERTICAL_THRESHOLD:
                # FIXED: Reversed the up/down mapping
                return GazeDirection.DOWN if yaw_deg < 0 else GazeDirection.UP
                
        return GazeDirection.CENTER
    
    def reset(self):
        """Reset detector state"""
        self.pitch_history.clear()
        self.yaw_history.clear()


# ==================== TEMPORAL FILTER ====================
class FastTemporalFilter:
    """Lightweight temporal filtering for better FPS"""
    
    def __init__(self, window_size: int, confidence_threshold: float, min_consistent: int):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_consistent = min_consistent
        
        self.direction_history = collections.deque(maxlen=window_size)
        self.pitch_history = collections.deque(maxlen=window_size)
        self.yaw_history = collections.deque(maxlen=window_size)
        
        self.current_direction = GazeDirection.CENTER
        self.consistent_count = 0
        
    def update(self, direction: GazeDirection, pitch_deg: float, yaw_deg: float):
        """Add new observation to history"""
        self.direction_history.append(direction)
        self.pitch_history.append(pitch_deg)
        self.yaw_history.append(yaw_deg)
        
        if direction == self.current_direction:
            self.consistent_count += 1
        else:
            self.consistent_count = 1
            self.current_direction = direction
            
    def get_filtered_direction(self) -> Tuple[GazeDirection, float]:
        """Get filtered direction with confidence"""
        if not self.direction_history:
            return GazeDirection.CENTER, 0.0
            
        # Simple majority voting (faster than weighted)
        counter = collections.Counter(self.direction_history)
        most_common = counter.most_common(1)[0]
        confidence = most_common[1] / len(self.direction_history)
        
        # Apply consistency check
        if self.consistent_count >= self.min_consistent:
            confidence = min(1.0, confidence * 1.2)
            
        if confidence >= self.confidence_threshold:
            return most_common[0], confidence
        else:
            return self.direction_history[-1], confidence
            
    def get_smoothed_angles(self) -> Tuple[float, float]:
        """Get smoothed angles using simple average"""
        if not self.pitch_history:
            return 0.0, 0.0
        return np.mean(self.pitch_history), np.mean(self.yaw_history)
        
    def reset(self):
        """Clear all history"""
        self.direction_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.current_direction = GazeDirection.CENTER
        self.consistent_count = 0


# ==================== GAZE ESTIMATION MODEL ====================
class GazeEstimationONNX:
    """Optimized ONNX model for gaze estimation"""
    
    def __init__(self, model_path: str):
        # Configure ONNX Runtime for better performance
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # Limit threads for better overall performance
        options.inter_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        
        # Model configuration
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        # Preprocessing parameters
        self.input_size = (448, 448)
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for face image"""
        # Use faster interpolation
        img = cv2.resize(face_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize in-place
        img = img.astype(np.float32, copy=False) * (1.0/255.0)
        img -= self.input_mean
        img /= self.input_std
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        return np.expand_dims(img, axis=0).astype(np.float32)
        
    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        """Estimate pitch and yaw angles from face image"""
        # Preprocess image
        input_tensor = self.preprocess(face_image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Parse outputs
        pitch_pred = outputs[0][0]
        yaw_pred = outputs[1][0]
        
        # Convert predictions to angles
        pitch_deg = self._compute_angle(pitch_pred)
        yaw_deg = self._compute_angle(yaw_pred)
        
        return pitch_deg, yaw_deg
        
    def _compute_angle(self, predictions: np.ndarray) -> float:
        """Optimized angle computation"""
        # Apply softmax
        exp_preds = np.exp(predictions - np.max(predictions))  # Numerical stability
        predictions = exp_preds / np.sum(exp_preds)
        
        # Compute expected value
        angle_deg = np.sum(predictions * self.idx_tensor) * self._binwidth
        angle_deg -= self._angle_offset
        
        return angle_deg


# ==================== SIMPLE VISUALIZER ====================
class SimpleVisualizer:
    """Simplified visualization for better FPS"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.colors = {
            'center': (0, 255, 0),
            'direction': (255, 165, 0),
            'face': (255, 255, 0),
            'mouse_on': (0, 255, 0),
            'mouse_off': (0, 0, 255)
        }
        
    def draw_minimal_overlay(self, frame: np.ndarray, bbox: np.ndarray, 
                            direction: GazeDirection, confidence: float,
                            pitch_deg: float, yaw_deg: float, fps: float,
                            mouse_enabled: bool):
        """Draw minimal overlay for better performance"""
        # Draw face box
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                         self.colors['face'], 2)
            
            # Draw simple direction arrow
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            arrow_length = 50
            
            if direction == GazeDirection.LEFT:
                end_x, end_y = center_x - arrow_length, center_y
            elif direction == GazeDirection.RIGHT:
                end_x, end_y = center_x + arrow_length, center_y
            elif direction == GazeDirection.UP:
                end_x, end_y = center_x, center_y - arrow_length
            elif direction == GazeDirection.DOWN:
                end_x, end_y = center_x, center_y + arrow_length
            else:
                end_x, end_y = center_x, center_y
                
            if direction != GazeDirection.CENTER:
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                               (0, 0, 255), 3, cv2.LINE_AA)
        
        # Create compact info panel
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Direction text
        color = self.colors['center'] if direction == GazeDirection.CENTER else self.colors['direction']
        cv2.putText(frame, f"Direction: {direction.value}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Mouse control status
        mouse_color = self.colors['mouse_on'] if mouse_enabled else self.colors['mouse_off']
        mouse_text = "Mouse: ON" if mouse_enabled else "Mouse: OFF"
        cv2.putText(frame, mouse_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2, cv2.LINE_AA)
        
        # FPS
        if self.settings.SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Angles (compact)
        if self.settings.SHOW_ANGLES:
            cv2.putText(frame, f"H:{pitch_deg:+.0f}° V:{yaw_deg:+.0f}°", (150, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'm' to toggle mouse, 'r' to reset", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ==================== MAIN APPLICATION ====================
class OptimizedGazeApp:
    """Optimized main application with mouse control"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = OptimizedGazeDetector(settings)
        self.filter = FastTemporalFilter(settings.SMOOTHING_WINDOW, 
                                        settings.CONFIDENCE_THRESHOLD,
                                        settings.MIN_CONSISTENT_FRAMES)
        self.visualizer = SimpleVisualizer(settings)
        self.mouse_controller = MouseController(settings)
        
        print("Loading model...")
        self.model = GazeEstimationONNX(settings.MODEL_PATH)
        
        print("Loading face detector...")
        self.face_detector = uniface.RetinaFace()
        
        # Statistics
        self.frame_count = 0
        self.process_count = 0
        self.direction_changes = 0
        self.last_direction = None
        self.no_face_count = 0
        
        # FPS tracking
        self.fps_history = collections.deque(maxlen=30)
        self.last_time = time.time()
        
        # Face tracking for optimization
        self.last_bbox = None
        self.face_lost_frames = 0
        
    def run(self):
        """Main application loop"""
        # Open video source
        cap = cv2.VideoCapture(self.settings.VIDEO_SOURCE)
        if not cap.isOpened():
            raise IOError(f"Failed to open video source: {self.settings.VIDEO_SOURCE}")
        
        # Set camera properties for better FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\n" + "="*60)
        print("OPTIMIZED GAZE DETECTION WITH MOUSE CONTROL")
        print("="*60)
        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        print(f"Frame Skip: Processing every {self.settings.FRAME_SKIP} frame(s)")
        print(f"Face Detection Scale: {self.settings.FACE_DETECTION_SCALE}")
        print(f"Mouse Control: {'Enabled' if self.settings.ENABLE_MOUSE_CONTROL else 'Disabled'}")
        print(f"Mouse Speed: H={self.settings.MOUSE_SPEED_HORIZONTAL}, V={self.settings.MOUSE_SPEED_VERTICAL}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'm' - Toggle mouse control")
        print("  'r' - Reset filters")
        print("  '+' - Increase mouse speed")
        print("  '-' - Decrease mouse speed")
        print("="*60 + "\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps_current = 1.0 / (current_time - self.last_time) if current_time != self.last_time else 0
                self.fps_history.append(fps_current)
                self.last_time = current_time
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                
                # Process frame (with frame skipping for better FPS)
                if self.frame_count % self.settings.FRAME_SKIP == 0:
                    self.process_frame(frame, avg_fps)
                    self.process_count += 1
                else:
                    # Still show visualization on skipped frames
                    if self.last_bbox is not None and self.settings.SHOW_VISUALIZATION:
                        direction, confidence = self.filter.get_filtered_direction()
                        pitch, yaw = self.filter.get_smoothed_angles()
                        self.visualizer.draw_minimal_overlay(
                            frame, self.last_bbox, direction, confidence,
                            pitch, yaw, avg_fps, self.mouse_controller.is_active
                        )
                
                cv2.imshow("Gaze Mouse Control", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mouse_controller.toggle_control()
                elif key == ord('r'):
                    self.detector.reset()
                    self.filter.reset()
                    print("\n[RESET] Filters have been reset\n")
                elif key == ord('+'):
                    self.settings.MOUSE_SPEED_HORIZONTAL = min(50, self.settings.MOUSE_SPEED_HORIZONTAL + 5)
                    self.settings.MOUSE_SPEED_VERTICAL = min(50, self.settings.MOUSE_SPEED_VERTICAL + 5)
                    print(f"\nMouse speed increased: H={self.settings.MOUSE_SPEED_HORIZONTAL}, V={self.settings.MOUSE_SPEED_VERTICAL}")
                elif key == ord('-'):
                    self.settings.MOUSE_SPEED_HORIZONTAL = max(5, self.settings.MOUSE_SPEED_HORIZONTAL - 5)
                    self.settings.MOUSE_SPEED_VERTICAL = max(5, self.settings.MOUSE_SPEED_VERTICAL - 5)
                    print(f"\nMouse speed decreased: H={self.settings.MOUSE_SPEED_HORIZONTAL}, V={self.settings.MOUSE_SPEED_VERTICAL}")
                    
        finally:
            self.mouse_controller.stop()
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print(f"Session Complete:")
            print(f"  Frames: {self.frame_count} total, {self.process_count} processed")
            print(f"  Average FPS: {np.mean(self.fps_history):.1f}" if self.fps_history else "N/A")
            print(f"  Direction changes: {self.direction_changes}")
            print("="*60)
            
    def process_frame(self, frame: np.ndarray, fps: float):
        """Process a single frame with optimizations"""
        # Detect faces (with scaling for better performance)
        if self.settings.FACE_DETECTION_SCALE < 1.0:
            small_frame = cv2.resize(frame, None, 
                                    fx=self.settings.FACE_DETECTION_SCALE,
                                    fy=self.settings.FACE_DETECTION_SCALE,
                                    interpolation=cv2.INTER_LINEAR)
            bboxes, _ = self.face_detector.detect(small_frame)
            # Scale bboxes back to original size
            if len(bboxes) > 0:
                bboxes = bboxes / self.settings.FACE_DETECTION_SCALE
        else:
            bboxes, _ = self.face_detector.detect(frame)
        
        if len(bboxes) == 0:
            self.handle_no_face(frame)
            return
            
        self.no_face_count = 0
        self.face_lost_frames = 0
        
        # Process largest face
        bbox = bboxes[0]
        self.last_bbox = bbox
        
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        
        # Ensure valid crop
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        face_crop = frame[y_min:y_max, x_min:x_max]
        
        if face_crop.size == 0:
            return
            
        # Estimate gaze
        pitch_deg, yaw_deg = self.model.estimate(face_crop)
        
        # Detect direction (with fixed up/down)
        direction = self.detector.detect_direction(pitch_deg, yaw_deg)
        
        # Apply temporal filtering
        self.filter.update(direction, pitch_deg, yaw_deg)
        filtered_direction, confidence = self.filter.get_filtered_direction()
        smooth_pitch, smooth_yaw = self.filter.get_smoothed_angles()
        
        # Track direction changes
        if (filtered_direction != self.last_direction and 
            confidence > self.settings.CONFIDENCE_THRESHOLD):
            self.direction_changes += 1
            self.last_direction = filtered_direction
            
            if self.settings.LOG_CHANGES:
                print(f"[Frame {self.frame_count:05d}] Direction: {filtered_direction.value:8s} "
                      f"(H: {smooth_pitch:+6.1f}°, V: {smooth_yaw:+6.1f}°)")
        
        # Control mouse
        if self.mouse_controller.is_active and confidence > self.settings.CONFIDENCE_THRESHOLD:
            self.mouse_controller.move_mouse(filtered_direction, confidence)
        
        # Visualize
        if self.settings.SHOW_VISUALIZATION:
            self.visualizer.draw_minimal_overlay(
                frame, bbox, filtered_direction, confidence,
                smooth_pitch, smooth_yaw, fps, self.mouse_controller.is_active
            )
            
    def handle_no_face(self, frame: np.ndarray):
        """Handle frames with no detected face"""
        self.no_face_count += 1
        self.face_lost_frames += 1
        
        # Use last known bbox for a few frames (optimization)
        if self.face_lost_frames < 5 and self.last_bbox is not None:
            # Continue with last known face position
            return
        
        self.last_bbox = None
        
        if self.no_face_count > self.settings.RESET_AFTER_NO_FACE:
            self.detector.reset()
            self.filter.reset()
            
        # Stop mouse movement when no face
        if self.mouse_controller.is_active:
            self.mouse_controller.move_mouse(GazeDirection.CENTER)
        
        # Show "No Face" message
        cv2.putText(frame, "No Face Detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


# ==================== MAIN ENTRY POINT ====================
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Gaze Detection with Mouse Control")
    parser.add_argument("--source", type=int, default=0, help="Video source")
    parser.add_argument("--model", type=str, default="mobileone_s0_gaze.onnx", help="Model path")
    parser.add_argument("--no-mouse", action="store_true", help="Disable mouse control")
    parser.add_argument("--mouse-speed", type=int, default=15, help="Mouse movement speed")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--detection-scale", type=float, default=0.5, help="Face detection scale (0-1)")
    
    args = parser.parse_args()
    
    # Create settings
    settings = Settings(
        VIDEO_SOURCE=args.source,
        MODEL_PATH=args.model,
        ENABLE_MOUSE_CONTROL=not args.no_mouse,
        MOUSE_SPEED_HORIZONTAL=args.mouse_speed,
        MOUSE_SPEED_VERTICAL=args.mouse_speed,
        FRAME_SKIP=args.frame_skip,
        FACE_DETECTION_SCALE=args.detection_scale
    )
    
    # Check if pynput is installed
    try:
        import pynput
    except ImportError:
        print("Error: pynput is not installed. Please install it using:")
        print("pip install pynput")
        return
    
    # Run application
    app = OptimizedGazeApp(settings)
    app.run()


if __name__ == "__main__":
    main()