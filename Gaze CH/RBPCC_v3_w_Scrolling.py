import cv2
import logging
import argparse 
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import math
import mediapipe as mp
import pyautogui
import uniface
from typing import Tuple
import onnxruntime as ort
import time 
from collections import deque

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============= CONFIGURATION CONSTANTS =============
# Blink Detection
EAR_CLOSE_THRESHOLD = 0.12  # Even lower for glasses wearers
EAR_OPEN_THRESHOLD = 0.18   # Lower gap for glasses
BLINK_CONSEC_FRAMES = 2     # Reduced back to 2 for easier detection
ACTION_COOLDOWN_MS = 600    # Slightly longer cooldown

# Auto-recalibration
AUTO_RECALIBRATE_SECONDS = 10  # Check for recalibration every 10 seconds
RECALIBRATION_THRESHOLD = 15  # Pixels - if resting position drifts this much    

# Head Position Control
HEAD_SENSITIVITY = 2.5       # How sensitive to head movement
HEAD_THRESHOLD = 30          # Pixels of movement needed to trigger (left/right/up)
HEAD_THRESHOLD_DOWN = 20    # Lower threshold for down movement (more sensitive)
MOVEMENT_SPEED = 15          # Cursor movement speed
SMOOTH_FACTOR = 0.3          # Smoothing for cursor movement
DIAGONAL_ENABLED = True      # Enable diagonal movement

# Scrolling
SCROLL_SPEED = 15            # Increased from 5 to 15 for faster scrolling
SCROLL_THRESHOLD = 40        # Head movement needed for scrolling

# Performance
FRAME_SKIP = 1              
CLEAR_CONSOLE_FREQ = 20    

# MediaPipe Eye Landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]  
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  

# ============= HEAD POSITION CONTROLLER =============
class HeadPositionController:
    """
    Control cursor using head position relative to calibrated center.
    Supports 4-directional and 8-directional (diagonal) movement.
    """
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.calibrated_center = None
        self.head_history = deque(maxlen=5)
        self.last_direction = "CENTER"
        self.movement_active = False
        
        # Auto-recalibration tracking
        self.resting_positions = deque(maxlen=30)  # Track positions when not moving
        self.last_recalibration_time = time.time()
        self.frames_since_movement = 0
        
    def calibrate(self, face_center_x, face_center_y):
        """Calibrate neutral head position."""
        self.calibrated_center = (face_center_x, face_center_y)
        self.resting_positions.clear()
        self.last_recalibration_time = time.time()
        print(f"Head position calibrated at ({face_center_x}, {face_center_y})")
        
    def auto_recalibrate_check(self, face_center_x, face_center_y, is_moving):
        """Check if auto-recalibration is needed based on resting position drift."""
        current_time = time.time()
        
        # Only track resting positions when not actively moving
        if not is_moving:
            self.frames_since_movement += 1
            if self.frames_since_movement > 10:  # Been still for 10 frames
                self.resting_positions.append((face_center_x, face_center_y))
        else:
            self.frames_since_movement = 0
            self.resting_positions.clear()
        
        # Check for recalibration every AUTO_RECALIBRATE_SECONDS
        if (current_time - self.last_recalibration_time > AUTO_RECALIBRATE_SECONDS and
            len(self.resting_positions) >= 20):
            
            # Calculate average resting position
            avg_x = sum(pos[0] for pos in self.resting_positions) / len(self.resting_positions)
            avg_y = sum(pos[1] for pos in self.resting_positions) / len(self.resting_positions)
            
            # Check if resting position has drifted
            if self.calibrated_center:
                drift = math.sqrt((avg_x - self.calibrated_center[0])**2 + 
                                (avg_y - self.calibrated_center[1])**2)
                
                if drift > RECALIBRATION_THRESHOLD:
                    print(f"Auto-recalibrating (drift: {drift:.1f} pixels)")
                    self.calibrate(int(avg_x), int(avg_y))
                    return True
        
        return False
        
    def get_direction(self, face_center_x, face_center_y):
        """Get movement direction based on head offset from center (supports diagonal)."""
        if self.calibrated_center is None:
            self.calibrate(face_center_x, face_center_y)
            return "CENTER", 0, 0
        
        # Calculate offset from calibrated center
        offset_x = face_center_x - self.calibrated_center[0]
        offset_y = face_center_y - self.calibrated_center[1]
        
        # Add to history for smoothing
        self.head_history.append((offset_x, offset_y))
        
        # Average the history
        if len(self.head_history) >= 3:
            avg_x = sum(h[0] for h in self.head_history) / len(self.head_history)
            avg_y = sum(h[1] for h in self.head_history) / len(self.head_history)
        else:
            avg_x, avg_y = offset_x, offset_y
        
        # Determine direction
        direction = "CENTER"
        move_x, move_y = 0, 0
        
        # Different threshold for down movement (more sensitive)
        y_threshold_up = HEAD_THRESHOLD
        y_threshold_down = HEAD_THRESHOLD_DOWN
        
        # Check horizontal movement
        horizontal_active = False
        if avg_x > HEAD_THRESHOLD:
            move_x = -MOVEMENT_SPEED  # INVERTED: head right = cursor left
            horizontal_active = True
        elif avg_x < -HEAD_THRESHOLD:
            move_x = MOVEMENT_SPEED  # INVERTED: head left = cursor right
            horizontal_active = True
        
        # Check vertical movement
        vertical_active = False
        if avg_y < -y_threshold_up:
            move_y = -MOVEMENT_SPEED  # Head up = cursor up
            vertical_active = True
        elif avg_y > y_threshold_down:
            move_y = MOVEMENT_SPEED  # Head down = cursor down
            vertical_active = True
        
        # Determine direction string for display
        if DIAGONAL_ENABLED and horizontal_active and vertical_active:
            # Diagonal movement
            if move_x < 0 and move_y < 0:
                direction = "UP-LEFT"
            elif move_x > 0 and move_y < 0:
                direction = "UP-RIGHT"
            elif move_x < 0 and move_y > 0:
                direction = "DOWN-LEFT"
            elif move_x > 0 and move_y > 0:
                direction = "DOWN-RIGHT"
        elif horizontal_active:
            direction = "LEFT" if move_x < 0 else "RIGHT"
        elif vertical_active:
            direction = "UP" if move_y < 0 else "DOWN"
        
        # Auto-recalibration check
        is_moving = (direction != "CENTER")
        self.auto_recalibrate_check(face_center_x, face_center_y, is_moving)
        
        return direction, move_x, move_y
    
    def update_cursor(self, face_center_x, face_center_y):
        """Update cursor position based on head movement."""
        direction, move_x, move_y = self.get_direction(face_center_x, face_center_y)
        
        if direction != "CENTER":
            # Move cursor (supports diagonal movement now)
            pyautogui.moveRel(move_x, move_y, duration=0)
            
            if direction != self.last_direction:
                print(f"Moving: {direction}")
                self.last_direction = direction
        else:
            if self.last_direction != "CENTER":
                print("Centered - Stop")
                self.last_direction = "CENTER"
        
        return direction
    
    def update_scroll(self, face_center_x, face_center_y):
        """Handle scrolling based on head position."""
        if self.calibrated_center is None:
            self.calibrate(face_center_x, face_center_y)
            return
        
        # Calculate offset
        offset_x = face_center_x - self.calibrated_center[0]
        offset_y = face_center_y - self.calibrated_center[1]
        
        # Auto-recalibration check for scroll mode too
        is_moving = (abs(offset_x) > SCROLL_THRESHOLD or abs(offset_y) > SCROLL_THRESHOLD)
        self.auto_recalibrate_check(face_center_x, face_center_y, is_moving)
        
        # Vertical scrolling (works on all platforms)
        if abs(offset_y) > SCROLL_THRESHOLD:
            if offset_y < -SCROLL_THRESHOLD:  # Head up = scroll up
                pyautogui.scroll(SCROLL_SPEED)
                print("Scrolling UP")
            elif offset_y > SCROLL_THRESHOLD:  # Head down = scroll down
                pyautogui.scroll(-SCROLL_SPEED)
                print("Scrolling DOWN")
        
        # Horizontal movement simulated with arrow keys (since hscroll doesn't work)
        if abs(offset_x) > SCROLL_THRESHOLD:
            if offset_x < -SCROLL_THRESHOLD:  # Head left = scroll right
                pyautogui.press('right')  # Use arrow key instead
                print("Scrolling RIGHT (arrow key)")
            elif offset_x > SCROLL_THRESHOLD:  # Head right = scroll left
                pyautogui.press('left')  # Use arrow key instead
                print("Scrolling LEFT (arrow key)")
    
    def draw_indicator(self, frame, face_center_x, face_center_y, mode="cursor"):
        """Draw visual indicators on frame."""
        h, w = frame.shape[:2]
        
        if self.calibrated_center:
            # Draw calibrated center (blue dot)
            cv2.circle(frame, (int(self.calibrated_center[0]), int(self.calibrated_center[1])), 
                      5, (255, 0, 0), -1)
            
            # Draw current face center
            color = (0, 255, 255) if mode == "scroll" else (0, 255, 0)
            cv2.circle(frame, (int(face_center_x), int(face_center_y)), 
                      5, color, -1)
            
            # Draw connection line
            cv2.line(frame, 
                    (int(self.calibrated_center[0]), int(self.calibrated_center[1])),
                    (int(face_center_x), int(face_center_y)), 
                    color, 2)
            
            # Draw threshold zones
            if mode == "cursor":
                h_threshold = HEAD_THRESHOLD
                v_threshold_up = HEAD_THRESHOLD
                v_threshold_down = HEAD_THRESHOLD_DOWN  # Different for down
            else:
                h_threshold = SCROLL_THRESHOLD
                v_threshold_up = SCROLL_THRESHOLD
                v_threshold_down = SCROLL_THRESHOLD
                
            # Horizontal threshold lines (inverted)
            cv2.line(frame, (int(self.calibrated_center[0] - h_threshold), 0),
                    (int(self.calibrated_center[0] - h_threshold), h), (100, 100, 100), 1)
            cv2.line(frame, (int(self.calibrated_center[0] + h_threshold), 0),
                    (int(self.calibrated_center[0] + h_threshold), h), (100, 100, 100), 1)
            
            # Vertical threshold lines (different for up/down in cursor mode)
            cv2.line(frame, (0, int(self.calibrated_center[1] - v_threshold_up)),
                    (w, int(self.calibrated_center[1] - v_threshold_up)), (100, 100, 100), 1)
            cv2.line(frame, (0, int(self.calibrated_center[1] + v_threshold_down)),
                    (w, int(self.calibrated_center[1] + v_threshold_down)), (100, 100, 100), 1)
            
            # Add direction labels (inverted for horizontal)
            cv2.putText(frame, "R", (int(self.calibrated_center[0] - h_threshold - 20), 
                       int(self.calibrated_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "L", (int(self.calibrated_center[0] + h_threshold + 10), 
                       int(self.calibrated_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "U", (int(self.calibrated_center[0]), 
                       int(self.calibrated_center[1] - v_threshold_up - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "D", (int(self.calibrated_center[0]), 
                       int(self.calibrated_center[1] + v_threshold_down + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show auto-recalibration status
            time_since_recal = time.time() - self.last_recalibration_time
            if time_since_recal < 2.0:
                cv2.putText(frame, "RECALIBRATED", (w - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ============= GLOBAL STATE =============
class TrackerState:
    def __init__(self):
        self.scroll_mode = False
        self.last_blink_ms = 0
        self.left_blink_frames = 0
        self.right_blink_frames = 0
        self.both_blink_frames = 0
        self.double_click_frames = 0
        self.frame_count = 0
        self.last_face_position = None  # Track face position for movement detection
        self.head_movement_detected = False
        
state = TrackerState()

# ============= UTILITY FUNCTIONS =============
def now_ms():
    return int(time.time() * 1000)

def clear_console():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ============= DRAWING FUNCTIONS =============
def draw_bbox(frame, bbox, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

def draw_status(frame, head_controller):
    """Draw status information."""
    mode_text = "SCROLL MODE" if state.scroll_mode else "CURSOR MODE"
    color = (0, 255, 255) if state.scroll_mode else (0, 255, 0)
    
    # Main status box
    cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 120), color, 2)
    
    cv2.putText(frame, f"Mode: {mode_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Instructions
    cv2.putText(frame, "Controls:", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Move head: 8-directional (diagonals enabled)", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "L-Blink: Toggle Mode | R-Blink: Right Click | Both: Left Click", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Calibration status
    if head_controller.calibrated_center:
        cv2.putText(frame, "Auto-Recal ON", (350, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Calibrating...", (360, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ============= GAZE ESTIMATION MODEL =============
class GazeEstimationTorch:
    def __init__(self, model_path: str):
        self.input_size = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.input_mean, std=self.input_std)
        ])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image).unsqueeze(0).numpy()
        return tensor

    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch, yaw = outputs
        
        pitch = torch.softmax(torch.tensor(pitch), dim=1).numpy()
        yaw = torch.softmax(torch.tensor(yaw), dim=1).numpy()
        
        pitch = np.sum(pitch * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw * self.idx_tensor) * self._binwidth - self._angle_offset
        
        return np.radians(pitch), np.radians(yaw)

# ============= MAIN CONTROL LOGIC =============
def process_blinks(landmarks, face_center_x, face_center_y):
    """Process blink detection for clicking - optimized for glasses wearers."""
    # Check for head movement to avoid false blink detection
    head_moving = False
    if state.last_face_position is not None:
        movement = abs(face_center_x - state.last_face_position[0]) + \
                  abs(face_center_y - state.last_face_position[1])
        if movement > 20:  # Increased threshold for glasses wearers
            head_moving = True
    
    state.last_face_position = (face_center_x, face_center_y)
    
    # Don't process blinks during significant head movement
    if head_moving:
        state.left_blink_frames = 0
        state.right_blink_frames = 0
        state.both_blink_frames = 0
        return None
    
    # Calculate eye aspect ratios
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    
    # Debug output for glasses wearers (every 30 frames)
    if state.frame_count % 30 == 0:
        print(f"EAR Debug - L: {left_ear:.3f}, R: {right_ear:.3f} (Threshold: {EAR_CLOSE_THRESHOLD})")
    
    # Detect closed eyes with glasses-friendly thresholds
    left_closed = left_ear < EAR_CLOSE_THRESHOLD
    right_closed = right_ear < EAR_CLOSE_THRESHOLD
    
    # Alternative: If both EAR values are similar and low, it's likely both eyes closed
    both_eyes_similar = abs(left_ear - right_ear) < 0.03
    both_likely_closed = both_eyes_similar and (left_ear + right_ear) / 2 < EAR_CLOSE_THRESHOLD
    
    # Update blink counters
    if (left_closed and right_closed) or both_likely_closed:
        state.both_blink_frames += 1
        state.left_blink_frames = 0
        state.right_blink_frames = 0
        
        # Check for double click
        if state.both_blink_frames == BLINK_CONSEC_FRAMES:
            current_time = now_ms()
            if state.double_click_frames > 0 and (current_time - state.last_blink_ms) < 500:
                print("DOUBLE CLICK")
                pyautogui.doubleClick()
                state.double_click_frames = 0
                state.last_blink_ms = current_time
                return "DOUBLE_CLICK"
            else:
                state.double_click_frames += 1
    elif left_closed and not right_closed:
        state.left_blink_frames += 1
        state.right_blink_frames = 0
        state.both_blink_frames = 0
    elif right_closed and not left_closed:
        state.right_blink_frames += 1
        state.left_blink_frames = 0
        state.both_blink_frames = 0
    else:
        state.left_blink_frames = 0
        state.right_blink_frames = 0
        state.both_blink_frames = 0
    
    # Process actions with cooldown
    now = now_ms()
    if now - state.last_blink_ms >= ACTION_COOLDOWN_MS:
        # Both eyes - Left click
        if state.both_blink_frames >= BLINK_CONSEC_FRAMES:
            print("LEFT CLICK")
            pyautogui.leftClick()
            state.last_blink_ms = now
            state.both_blink_frames = 0
            return "LEFT_CLICK"
        
        # Right eye - Right click
        elif state.right_blink_frames >= BLINK_CONSEC_FRAMES:
            print("RIGHT CLICK")
            pyautogui.rightClick()
            state.last_blink_ms = now
            state.right_blink_frames = 0
            return "RIGHT_CLICK"
        
        # Left eye - Toggle mode
        elif state.left_blink_frames >= BLINK_CONSEC_FRAMES:
            state.scroll_mode = not state.scroll_mode
            mode = "SCROLL" if state.scroll_mode else "CURSOR"
            print(f"MODE: {mode}")
            state.last_blink_ms = now
            state.left_blink_frames = 0
            return f"MODE_{mode}"
    
    return None

# ============= MAIN FUNCTION =============
def parse_args():
    parser = argparse.ArgumentParser(description="Eye Tracking with Head Position Control")
    parser.add_argument("--source", type=str, required=True, help="Camera index")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path")
    parser.add_argument("--output", type=str, default=None, help="Output video")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing head position control system...")
    engine = GazeEstimationTorch(args.model)
    detector = uniface.RetinaFace()
    head_controller = HeadPositionController()
    
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                               cap.get(cv2.CAP_PROP_FPS) or 30, (width, height))
    
    print("\n=== HEAD POSITION CONTROL ===")
    print("SETUP:")
    print("  1. Position your head comfortably in center of camera")
    print("  2. System will auto-calibrate to this position")
    print("  3. Move head to control cursor/scrolling")
    print("  4. AUTO-RECALIBRATION: System will automatically adjust if you shift position")
    print("\nCONTROLS:")
    print("  - Move head: 8-DIRECTIONAL (including diagonals!)")
    print("  - Left Eye Blink: Toggle Cursor/Scroll Mode")
    print("  - Right Eye Blink: Right Click")
    print("  - Both Eyes Blink: Left Click")
    print("  - Double Blink (both): Double Click")
    print("\nKEYBOARD SHORTCUTS:")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to force recalibrate center position")
    print("  - Press 's' to toggle scroll mode")
    print("  - Press 'v' to toggle visual indicators")
    print("  - Press '1' for LEFT CLICK")
    print("  - Press '2' for RIGHT CLICK")
    print("  - Press '3' for DOUBLE CLICK")
    print("  - Press 'd' to see debug info")
    print("========================\n")
    
    show_indicators = True
    show_debug = False  # Debug mode for glasses wearers
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            state.frame_count += 1
            
            if state.frame_count % CLEAR_CONSOLE_FREQ == 0:
                clear_console()
            
            # Face detection
            bboxes, _ = detector.detect(frame)
            
            if len(bboxes) > 0:
                bbox = bboxes[0]
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                face_img = frame[y_min:y_max, x_min:x_max]
                
                if face_img.size == 0:
                    continue
                
                # Calculate face center (for head position)
                face_center_x = (x_min + x_max) // 2
                face_center_y = (y_min + y_max) // 2
                
                # Still run gaze estimation (for future use if needed)
                pitch, yaw = engine.estimate(face_img)
                
                # Draw bounding box
                color = (0, 255, 255) if state.scroll_mode else (0, 255, 0)
                draw_bbox(frame, bbox, color)
                
                # MediaPipe for blink detection
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    landmarks = [(int(pt.x * w), int(pt.y * h)) 
                               for pt in face_landmarks.landmark]
                    
                    # Process blinks (now with head movement detection)
                    action = process_blinks(landmarks, face_center_x, face_center_y)
                    
                    # Handle mode-specific actions using HEAD POSITION
                    if state.scroll_mode:
                        # Scroll mode - use head position for scrolling
                        head_controller.update_scroll(face_center_x, face_center_y)
                    else:
                        # Cursor mode - use head position for cursor movement
                        head_controller.update_cursor(face_center_x, face_center_y)
                    
                    # Draw visual indicators
                    if show_indicators:
                        mode = "scroll" if state.scroll_mode else "cursor"
                        head_controller.draw_indicator(frame, face_center_x, face_center_y, mode)
            
            # Draw status
            draw_status(frame, head_controller)
            
            if writer:
                writer.write(frame)
            
            cv2.imshow("Head Position Control", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if len(bboxes) > 0:
                    head_controller.calibrated_center = None
                    print("Manual recalibration...")
            elif key == ord('s'):
                state.scroll_mode = not state.scroll_mode
                print(f"Mode: {'SCROLL' if state.scroll_mode else 'CURSOR'}")
            elif key == ord('v'):
                show_indicators = not show_indicators
                print(f"Visual indicators: {'ON' if show_indicators else 'OFF'}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug mode: {'ON - Check console for EAR values' if show_debug else 'OFF'}")
            # Keyboard shortcuts for clicking (for glasses wearers)
            elif key == ord('1'):
                print("KEYBOARD: LEFT CLICK")
                pyautogui.leftClick()
            elif key == ord('2'):
                print("KEYBOARD: RIGHT CLICK")
                pyautogui.rightClick()
            elif key == ord('3'):
                print("KEYBOARD: DOUBLE CLICK")
                pyautogui.doubleClick()
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print("Head tracking stopped.")

if __name__ == "__main__":
    main()
