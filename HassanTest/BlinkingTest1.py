"""
Enhanced Blink Detection System with Mode-Based Interaction
Model 1: Normal Mode (clicks + keyboard) and Scroll Mode

Interaction Model:
-----------------
NORMAL MODE (Default):
  - Left Eye Blink (100-800ms)    → Left Click
  - Right Eye Blink (100-800ms)   → Right Click
  - Both Eyes Closed (<1000ms)    → Open Virtual Keyboard
  - Both Eyes Closed (>2000ms)    → Toggle to SCROLL MODE

SCROLL MODE:
  - Gaze UP/DOWN/LEFT/RIGHT       → Scroll in that direction
  - Both Eyes Closed (any)        → Return to NORMAL MODE
"""

import cv2
import numpy as np
import time
from enum import Enum
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Callable
from pynput.mouse import Button, Controller as MouseController


# ==================== ENUMS & CONSTANTS ====================
class EyeState(Enum):
    """Eye detection states"""
    BOTH_OPEN = "BOTH_OPEN"
    LEFT_ONLY = "LEFT_ONLY"      # Right eye closed (left blink detected)
    RIGHT_ONLY = "RIGHT_ONLY"    # Left eye closed (right blink detected)
    BOTH_CLOSED = "BOTH_CLOSED"
    NO_FACE = "NO_FACE"


class BlinkType(Enum):
    """Types of blink events"""
    LEFT_BLINK = "LEFT_BLINK"
    RIGHT_BLINK = "RIGHT_BLINK"
    BOTH_BLINK = "BOTH_BLINK"
    NONE = "NONE"


class SystemMode(Enum):
    """System operation modes"""
    NORMAL = "NORMAL"
    SCROLL = "SCROLL"


class ActionType(Enum):
    """User action types"""
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    OPEN_KEYBOARD = "OPEN_KEYBOARD"
    SCROLL_MODE_ON = "SCROLL_MODE_ON"
    SCROLL_MODE_OFF = "SCROLL_MODE_OFF"
    NONE = "NONE"


# ==================== SETTINGS ====================
@dataclass
class BlinkDetectionSettings:
    """Configuration for blink detection"""

    # Haar Cascade Parameters
    FACE_SCALE_FACTOR: float = 1.1
    FACE_MIN_NEIGHBORS: int = 5
    FACE_MIN_SIZE: Tuple[int, int] = (30, 30)

    EYE_SCALE_FACTOR: float = 1.03
    EYE_MIN_NEIGHBORS: int = 5  # Reduced for better sensitivity
    EYE_MIN_SIZE: Tuple[int, int] = (20, 20)

    # Blink Duration Thresholds (in seconds)
    MIN_BLINK_DURATION: float = 0.1      # 100ms - minimum to count as intentional
    MAX_BLINK_DURATION: float = 0.8      # 800ms - maximum for single blink
    KEYBOARD_THRESHOLD: float = 1.0       # 1000ms - both eyes closed for keyboard
    SCROLL_MODE_THRESHOLD: float = 2.0    # 2000ms - both eyes closed for scroll mode

    # Temporal Filtering
    STATE_CONFIRMATION_FRAMES: int = 1    # Frames needed to confirm state change (more responsive)
    DEBOUNCE_TIME: float = 0.3           # 300ms debounce between actions

    # Eye Position Tolerance
    EYE_POSITION_TOLERANCE: int = 20      # Pixels tolerance for left/right eye tracking

    # Display Settings
    SHOW_DEBUG_INFO: bool = True
    SHOW_EYE_REGIONS: bool = True
    SHOW_STATE_HISTORY: bool = True


# ==================== BLINK DETECTOR ====================
class BlinkDetector:
    """
    Advanced blink detector with temporal filtering and mode management
    """

    def __init__(self, settings: BlinkDetectionSettings):
        self.settings = settings

        # State tracking
        self.current_state = EyeState.NO_FACE
        self.previous_state = EyeState.NO_FACE
        self.state_start_time = time.time()
        self.state_duration = 0.0

        # Eye tracking
        self.left_eye_pos = None
        self.right_eye_pos = None
        self.last_known_left_pos = None
        self.last_known_right_pos = None

        # State history for temporal filtering
        self.state_history = deque(maxlen=settings.STATE_CONFIRMATION_FRAMES * 2)

        # Mode management
        self.current_mode = SystemMode.NORMAL

        # Action tracking
        self.last_action_time = 0.0
        self.last_action = ActionType.NONE

        # Statistics
        self.blink_count = {"left": 0, "right": 0, "both": 0}
        self.action_count = {action: 0 for action in ActionType}

        # Event callbacks
        self.action_callbacks = {}

    def register_action_callback(self, action_type: ActionType, callback: Callable):
        """Register a callback for a specific action type"""
        self.action_callbacks[action_type] = callback

    def update(self, eyes_detected: list, face_region: Tuple[int, int, int, int]) -> Optional[ActionType]:
        """
        Update detector with new frame data

        Args:
            eyes_detected: List of (x, y, w, h) tuples for detected eyes
            face_region: (x, y, w, h) of the face region

        Returns:
            ActionType if an action should be triggered, None otherwise
        """
        current_time = time.time()

        # Determine current eye state
        new_state = self._determine_eye_state(eyes_detected, face_region)

        # Add to state history
        self.state_history.append(new_state)

        # Check if state has changed with confirmation
        confirmed_state = self._get_confirmed_state()

        if confirmed_state != self.current_state:
            # State changed
            action = self._handle_state_change(confirmed_state, current_time)
            self.previous_state = self.current_state
            self.current_state = confirmed_state
            self.state_start_time = current_time
            self.state_duration = 0.0

            return action
        else:
            # State unchanged, update duration
            self.state_duration = current_time - self.state_start_time

            # Check for long-press actions (both eyes closed)
            if self.current_state == EyeState.BOTH_CLOSED:
                return self._check_long_press_actions(current_time)

        return None

    def _determine_eye_state(self, eyes: list, face_region: Tuple[int, int, int, int]) -> EyeState:
        """Determine the current eye state from detected eyes"""
        num_eyes = len(eyes)

        if num_eyes == 0:
            return EyeState.BOTH_CLOSED
        elif num_eyes == 1:
            # Determine which eye is open
            # Eye coordinates are relative to the face ROI
            eye_x = eyes[0][0]
            _, _, face_w, _ = face_region
            face_center_x = face_w // 2  # Center of face ROI

            # If eye is on the left side of face center, it's the left eye
            # (from camera view, which appears on left side of screen = user's RIGHT eye)
            if eye_x < face_center_x:
                return EyeState.LEFT_ONLY  # Camera left = User's right
            else:
                return EyeState.RIGHT_ONLY  # Camera right = User's left
        else:  # 2 or more eyes
            # Sort eyes by x-coordinate (left to right in camera view)
            sorted_eyes = sorted(eyes[:2], key=lambda e: e[0])
            self.left_eye_pos = (sorted_eyes[0][0], sorted_eyes[0][1])
            self.right_eye_pos = (sorted_eyes[1][0], sorted_eyes[1][1])

            # Update last known positions
            self.last_known_left_pos = self.left_eye_pos
            self.last_known_right_pos = self.right_eye_pos

            return EyeState.BOTH_OPEN

    def _get_confirmed_state(self) -> EyeState:
        """Get confirmed state using temporal filtering"""
        if len(self.state_history) < self.settings.STATE_CONFIRMATION_FRAMES:
            return self.current_state

        # Get last N states
        recent_states = list(self.state_history)[-self.settings.STATE_CONFIRMATION_FRAMES:]

        # Check if all recent states agree
        if all(s == recent_states[0] for s in recent_states):
            return recent_states[0]
        else:
            return self.current_state

    def _handle_state_change(self, new_state: EyeState, current_time: float) -> Optional[ActionType]:
        """Handle state transition and detect blink events"""

        # Ignore state changes during debounce period
        if current_time - self.last_action_time < self.settings.DEBOUNCE_TIME:
            return None

        # Detect blink completion (transition from closed to open)
        if new_state == EyeState.BOTH_OPEN and self.current_state != EyeState.BOTH_OPEN:
            blink_duration = self.state_duration

            # Check if blink duration is valid
            if self.settings.MIN_BLINK_DURATION <= blink_duration <= self.settings.MAX_BLINK_DURATION:
                return self._process_blink_event(self.current_state, blink_duration, current_time)

        return None

    def _process_blink_event(self, closed_state: EyeState, duration: float, current_time: float) -> Optional[ActionType]:
        """Process a completed blink event"""

        action = None

        if self.current_mode == SystemMode.NORMAL:
            # Normal mode: map blinks to clicks and keyboard
            if closed_state == EyeState.LEFT_ONLY:
                # Right eye was closed (from user's perspective)
                action = ActionType.RIGHT_CLICK
                self.blink_count["right"] += 1

            elif closed_state == EyeState.RIGHT_ONLY:
                # Left eye was closed (from user's perspective)
                action = ActionType.LEFT_CLICK
                self.blink_count["left"] += 1

            # Both eyes blink handled in long-press

        elif self.current_mode == SystemMode.SCROLL:
            # In scroll mode, any blink exits scroll mode
            if closed_state in [EyeState.LEFT_ONLY, EyeState.RIGHT_ONLY, EyeState.BOTH_CLOSED]:
                action = ActionType.SCROLL_MODE_OFF
                self.current_mode = SystemMode.NORMAL

        if action:
            self._trigger_action(action, current_time)

        return action

    def _check_long_press_actions(self, current_time: float) -> Optional[ActionType]:
        """Check for long-press actions (both eyes closed)"""

        # Avoid triggering multiple times for same long press
        if current_time - self.last_action_time < self.settings.DEBOUNCE_TIME:
            return None

        action = None

        if self.current_mode == SystemMode.NORMAL:
            # Check for keyboard trigger
            if (self.state_duration >= self.settings.KEYBOARD_THRESHOLD and
                self.state_duration < self.settings.SCROLL_MODE_THRESHOLD):

                # Only trigger once when threshold is crossed
                if self.last_action != ActionType.OPEN_KEYBOARD:
                    action = ActionType.OPEN_KEYBOARD
                    self.blink_count["both"] += 1

            # Check for scroll mode trigger
            elif self.state_duration >= self.settings.SCROLL_MODE_THRESHOLD:
                if self.last_action != ActionType.SCROLL_MODE_ON:
                    action = ActionType.SCROLL_MODE_ON
                    self.current_mode = SystemMode.SCROLL

        if action:
            self._trigger_action(action, current_time)

        return action

    def _trigger_action(self, action: ActionType, current_time: float):
        """Trigger an action and update tracking"""
        self.last_action = action
        self.last_action_time = current_time
        self.action_count[action] += 1

        # Call registered callback if exists
        if action in self.action_callbacks:
            self.action_callbacks[action]()

        # Print action for debugging
        print(f"\n[ACTION] {action.value} at {current_time:.2f}s (Mode: {self.current_mode.value})")

    def get_state_info(self) -> dict:
        """Get current state information for visualization"""
        return {
            "state": self.current_state,
            "mode": self.current_mode,
            "duration": self.state_duration,
            "left_eye": self.left_eye_pos,
            "right_eye": self.right_eye_pos,
            "blink_count": self.blink_count.copy(),
            "action_count": self.action_count.copy()
        }

    def reset(self):
        """Reset detector state"""
        self.current_state = EyeState.NO_FACE
        self.previous_state = EyeState.NO_FACE
        self.state_start_time = time.time()
        self.state_duration = 0.0
        self.state_history.clear()
        self.last_action = ActionType.NONE


# ==================== VISUALIZER ====================
class BlinkVisualizer:
    """Handles visualization of blink detection"""

    def __init__(self, settings: BlinkDetectionSettings):
        self.settings = settings

        # Colors
        self.colors = {
            "face": (255, 255, 0),      # Cyan for face
            "eye": (0, 255, 0),          # Green for eyes
            "left_eye": (0, 255, 255),   # Yellow for left eye
            "right_eye": (255, 0, 255),  # Magenta for right eye
            "normal_mode": (0, 255, 0),  # Green for normal mode
            "scroll_mode": (0, 165, 255) # Orange for scroll mode
        }

    def draw_debug_overlay(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                          eyes: list, state_info: dict):
        """Draw comprehensive debug overlay"""

        h, w = frame.shape[:2]

        # Draw face rectangle
        x, y, fw, fh = face_bbox
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), self.colors["face"], 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, self.colors["face"], 2, cv2.LINE_AA)

        # Draw eyes
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes[:2], key=lambda e: e[0])

            # Left eye (user's right)
            ex, ey, ew, eh = sorted_eyes[0]
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                         self.colors["left_eye"], 2)
            cv2.putText(frame, "L", (x + ex, y + ey - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["left_eye"], 2)

            # Right eye (user's left)
            ex, ey, ew, eh = sorted_eyes[1]
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                         self.colors["right_eye"], 2)
            cv2.putText(frame, "R", (x + ex, y + ey - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["right_eye"], 2)
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                         self.colors["eye"], 2)

        # Draw info panel
        self._draw_info_panel(frame, state_info)

    def _draw_info_panel(self, frame: np.ndarray, state_info: dict):
        """Draw information panel with state and statistics"""

        h, w = frame.shape[:2]
        panel_width = 280
        panel_height = 180
        panel_x = w - panel_width - 10
        panel_y = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_height = 20
        y_offset = panel_y + 20

        # Title
        cv2.putText(frame, "BLINK DETECTOR", (panel_x + 10, y_offset),
                   font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += line_height + 3

        # Mode
        mode = state_info["mode"]
        mode_color = self.colors["scroll_mode"] if mode == SystemMode.SCROLL else self.colors["normal_mode"]
        cv2.putText(frame, f"Mode: {mode.value}", (panel_x + 10, y_offset),
                   font, 0.6, mode_color, 2, cv2.LINE_AA)
        y_offset += line_height + 3

        # Eye State
        state = state_info["state"]
        state_text = f"State: {state.value}"
        cv2.putText(frame, state_text, (panel_x + 10, y_offset),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += line_height

        # State Duration
        duration = state_info["duration"]
        duration_text = f"Duration: {duration:.2f}s"
        cv2.putText(frame, duration_text, (panel_x + 10, y_offset),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += line_height + 3

        # Blink counts (compact)
        blink_count = state_info["blink_count"]
        cv2.putText(frame, f"Blinks: L:{blink_count['left']} R:{blink_count['right']} B:{blink_count['both']}",
                   (panel_x + 10, y_offset),
                   font, 0.4, (200, 200, 200), thickness, cv2.LINE_AA)
        y_offset += line_height + 3

        # Action mapping based on mode (compact)
        if mode == SystemMode.NORMAL:
            cv2.putText(frame, "L-Blink->LClick R-Blink->RClick", (panel_x + 10, y_offset),
                       font, 0.35, (180, 180, 180), thickness, cv2.LINE_AA)
            y_offset += 15
            cv2.putText(frame, "Both <1s->Keyboard >2s->Scroll", (panel_x + 10, y_offset),
                       font, 0.35, (180, 180, 180), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Gaze -> Scroll", (panel_x + 10, y_offset),
                       font, 0.4, (180, 180, 180), thickness, cv2.LINE_AA)
            y_offset += 15
            cv2.putText(frame, "Any Blink -> Exit Scroll", (panel_x + 10, y_offset),
                       font, 0.4, (180, 180, 180), thickness, cv2.LINE_AA)

        # Instructions at bottom
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset",
                   (10, h - 10), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ==================== MAIN APPLICATION ====================
class BlinkDetectionApp:
    """Main application for blink detection"""

    def __init__(self, settings: BlinkDetectionSettings):
        self.settings = settings

        # Load Haar Cascades
        print("Loading Haar Cascade classifiers...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Try to use the eye_tree_eyeglasses cascade which is better for open eyes
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

        # Fallback to regular eye cascade if not available
        if self.eye_cascade.empty():
            print("Using fallback eye cascade...")
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifiers")

        # Initialize detector and visualizer
        self.detector = BlinkDetector(settings)
        self.visualizer = BlinkVisualizer(settings)

        # Register action callbacks
        self._register_callbacks()

        # Statistics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

        print("Blink Detection System initialized successfully!")

    def _register_callbacks(self):
        """Register callbacks for actions"""

        # Initialize mouse controller
        self.mouse = MouseController()

        def on_left_click():
            print("  → Executing: LEFT CLICK")
            try:
                self.mouse.click(Button.left, 1)
            except Exception as e:
                print(f"    Error clicking: {e}")

        def on_right_click():
            print("  → Executing: RIGHT CLICK")
            try:
                self.mouse.click(Button.right, 1)
            except Exception as e:
                print(f"    Error clicking: {e}")

        def on_keyboard():
            print("  → Executing: OPEN VIRTUAL KEYBOARD")
            # TODO: Implement virtual keyboard opening
            # For now, just print
            print("    [Virtual Keyboard would open here]")

        def on_scroll_mode_on():
            print("  → Executing: SCROLL MODE ACTIVATED")
            # In scroll mode, scrolling will be controlled by gaze direction

        def on_scroll_mode_off():
            print("  → Executing: SCROLL MODE DEACTIVATED")

        self.detector.register_action_callback(ActionType.LEFT_CLICK, on_left_click)
        self.detector.register_action_callback(ActionType.RIGHT_CLICK, on_right_click)
        self.detector.register_action_callback(ActionType.OPEN_KEYBOARD, on_keyboard)
        self.detector.register_action_callback(ActionType.SCROLL_MODE_ON, on_scroll_mode_on)
        self.detector.register_action_callback(ActionType.SCROLL_MODE_OFF, on_scroll_mode_off)

    def run(self, video_source=0):
        """Run the blink detection application"""

        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        print("\n" + "="*60)
        print("BLINK DETECTION SYSTEM - MODEL 1")
        print("="*60)
        print("\nNORMAL MODE:")
        print("  Left Eye Blink (0.1-0.8s)  → Left Click")
        print("  Right Eye Blink (0.1-0.8s) → Right Click")
        print("  Both Eyes (<1s)            → Virtual Keyboard")
        print("  Both Eyes (>2s)            → Scroll Mode")
        print("\nSCROLL MODE:")
        print("  Gaze Direction             → Scroll")
        print("  Any Blink                  → Exit to Normal Mode")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset detector")
        print("="*60 + "\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time) if current_time != self.last_time else 0
                self.fps_history.append(fps)
                self.last_time = current_time
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0

                # Process frame
                self.process_frame(frame)

                # Show FPS
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Display frame
                cv2.imshow("Blink Detection - Model 1", frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.detector.reset()
                    print("\n[RESET] Detector has been reset\n")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Print statistics
            state_info = self.detector.get_state_info()
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)
            print(f"Total Frames: {self.frame_count}")
            print(f"Average FPS: {np.mean(self.fps_history):.1f}" if self.fps_history else "N/A")
            print(f"\nBlink Count:")
            print(f"  Left:  {state_info['blink_count']['left']}")
            print(f"  Right: {state_info['blink_count']['right']}")
            print(f"  Both:  {state_info['blink_count']['both']}")
            print(f"\nAction Count:")
            for action, count in state_info['action_count'].items():
                if count > 0:
                    print(f"  {action.value}: {count}")
            print("="*60)

    def process_frame(self, frame: np.ndarray):
        """Process a single frame for blink detection"""

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.settings.FACE_SCALE_FACTOR,
            minNeighbors=self.settings.FACE_MIN_NEIGHBORS,
            minSize=self.settings.FACE_MIN_SIZE
        )

        if len(faces) == 0:
            # No face detected
            self.detector.update([], (0, 0, 0, 0))
            cv2.putText(frame, "No Face Detected", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return

        # Process largest face
        face = max(faces, key=lambda f: f[2] * f[3])  # Largest by area
        x, y, w, h = face

        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within face
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=self.settings.EYE_SCALE_FACTOR,
            minNeighbors=self.settings.EYE_MIN_NEIGHBORS,
            minSize=self.settings.EYE_MIN_SIZE,
            maxSize=(w//3, h//3)
        )

        # Filter out overlapping detections and ensure eyes are horizontally separated
        filtered_eyes = []
        for eye in eyes:
            ex, ey, ew, eh = eye
            # Check if this eye is far enough from other detected eyes
            is_unique = True
            for other_eye in filtered_eyes:
                ox, oy, ow, oh = other_eye
                # Calculate horizontal distance between eye centers
                eye_center_x = ex + ew // 2
                other_center_x = ox + ow // 2
                distance = abs(eye_center_x - other_center_x)

                # If eyes are too close horizontally, it's probably the same eye
                if distance < w // 4:  # Eyes should be at least 1/4 face width apart
                    is_unique = False
                    break

            if is_unique:
                filtered_eyes.append(eye)

        eyes = filtered_eyes[:2]  # Keep only top 2 eyes

        # Debug: Show raw eye count and which eye if only one
        debug_text = f"Eyes Detected: {len(eyes)}"
        if len(eyes) == 1:
            eye_x = eyes[0][0]
            face_center = w // 2
            which_eye = "LEFT" if eye_x < face_center else "RIGHT"
            debug_text += f" ({which_eye})"
        cv2.putText(frame, debug_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Update detector
        action = self.detector.update(list(eyes), face)

        # Visualize
        if self.settings.SHOW_DEBUG_INFO:
            state_info = self.detector.get_state_info()
            self.visualizer.draw_debug_overlay(frame, face, eyes, state_info)


# ==================== MAIN ENTRY POINT ====================
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Blink Detection System - Model 1")
    parser.add_argument("--source", type=int, default=0, help="Video source (0 for webcam)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visualization")

    args = parser.parse_args()

    # Create settings
    settings = BlinkDetectionSettings()
    if args.no_debug:
        settings.SHOW_DEBUG_INFO = False

    # Run application
    app = BlinkDetectionApp(settings)
    app.run(video_source=args.source)


if __name__ == "__main__":
    main()
