"""
Unified Eye Tracking Control System
Integrates gaze-based mouse movement with blink-based click actions
"""

import cv2
import numpy as np
import time
import argparse
from threading import Thread, Lock
import queue

# Import from gaze detection module
from Eight_Direction_Gaze_v1 import (
    Settings as GazeSettings,
    AccurateGazeDetector,
    GazeEstimationONNX,
    AdvancedTemporalFilter,
    PreciseMouseController,
    GazeDirection
)

# Import from blink detection module
from blink_detection import (
    BlinkDetection,
    eye_aspect_ratio,
    ScrollModeDetector,
    LEFT_EYE,
    RIGHT_EYE,
    EAR_CLOSE_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    BLINK_CONSEC_FRAMES,
    now_ms
)

# Additional imports needed
import uniface
import mediapipe as mp
from pynput.mouse import Controller, Button


class UnifiedEyeTracker:
    """
    Unified eye tracking system combining gaze direction and blink detection
    """
    
    def __init__(self, gaze_settings: GazeSettings, camera_source: int = 0):
        print("\n" + "="*80)
        print("UNIFIED EYE TRACKING CONTROL SYSTEM")
        print("="*80)
        print("\nInitializing components...")
        
        self.gaze_settings = gaze_settings
        self.camera_source = camera_source
        
        # Initialize gaze detection components
        print("  • Loading gaze estimation model...")
        self.gaze_model = GazeEstimationONNX(gaze_settings.MODEL_PATH)
        
        print("  • Loading face detector...")
        self.face_detector = uniface.RetinaFace()
        
        print("  • Initializing gaze detector...")
        self.gaze_detector = AccurateGazeDetector(gaze_settings)
        
        print("  • Setting up temporal filter...")
        self.gaze_filter = AdvancedTemporalFilter(
            gaze_settings.SMOOTHING_WINDOW,
            gaze_settings.CONFIDENCE_THRESHOLD,
            gaze_settings.MIN_CONSISTENT_FRAMES
        )
        
        # Initialize blink detection components
        print("  • Loading MediaPipe Face Mesh...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("  • Initializing blink detector...")
        self.blink_detector = BlinkDetection()
        
        print("  • Initializing scroll mode detector...")
        self.scroll_detector = ScrollModeDetector()
        
        # Mouse controller (shared between gaze and blink)
        print("  • Setting up mouse controller...")
        self.mouse_controller = PreciseMouseController(gaze_settings)
        self.mouse = Controller()  # For blink actions
        
        # State management
        self.last_bbox = None
        self.frame_count = 0
        self.lock = Lock()
        
        # Performance tracking
        self.fps_times = []
        self.last_time = time.time()
        
        # Current state
        self.current_gaze_direction = GazeDirection.CENTER
        self.current_confidence = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        
        print("\n✓ All components initialized successfully!")
        print("="*80 + "\n")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.fps_times) > 30:
            self.fps_times.pop(0)
        
        if len(self.fps_times) > 0:
            avg_time = sum(self.fps_times) / len(self.fps_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0
        return 0.0
    
    def process_gaze(self, frame, bbox):
        """
        Process gaze detection on detected face
        Returns: (direction, confidence, pitch, yaw)
        """
        start_time = time.time()
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        face = frame[y_min:y_max, x_min:x_max]
        if face.size == 0:
            return None
        
        # Estimate gaze angles
        pitch, yaw = self.gaze_model.estimate(face)
        time_calculator(start_time, "Model inference")
        start_time = time.time()
        # Detect direction
        direction, confidence = self.gaze_detector.detect_direction(pitch, yaw)
        time_calculator(start_time, "Direction detection")
        # Apply temporal filter
        start_time = time.time()
        filtered_dir, filtered_conf = self.gaze_filter.update(
            direction, confidence, pitch, yaw
        )
        time_calculator(start_time, "Filter")
        return filtered_dir, filtered_conf, pitch, yaw
    
    def process_blink(self, frame, landmarks):
        """
        Process blink detection on face landmarks
        Returns: action string or None
        """
        h, w, _ = frame.shape
        landmark_points = [(int(pt.x * w), int(pt.y * h)) 
                          for pt in landmarks.landmark]
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(landmark_points, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmark_points, RIGHT_EYE)
        
        # Determine eye states
        left_closed = left_ear < EAR_CLOSE_THRESHOLD
        right_closed = right_ear < EAR_CLOSE_THRESHOLD
        left_open = left_ear > EAR_OPEN_THRESHOLD
        right_open = right_ear > EAR_OPEN_THRESHOLD
        
        eyes_open = not (left_closed and right_closed)
        
        # Update blink state
        blink_type = self.blink_detector.update_blink_state(
            left_closed, right_closed, left_open, right_open
        )
        
        # Update scroll mode
        current_time = now_ms()
        scroll_mode = self.scroll_detector.update(
            landmark_points, current_time, eyes_open=eyes_open
        )
        
        # Handle blink actions
        action = self.blink_detector.handle_blink_actions(
            blink_type, landmark_points, self.mouse, current_time
        )
        
        return action, scroll_mode, left_ear, right_ear
    
    def process_frame(self, frame):
        """
        Main frame processing - combines gaze and blink detection
        """
        self.frame_count += 1
        fps = self.calculate_fps()
        
        # Initialize results
        gaze_result = None
        blink_result = None
        scroll_mode = False
        left_ear = 0.0
        right_ear = 0.0
        start_time = time.time()
        # Process every N frames for performance
        if self.frame_count % self.gaze_settings.FRAME_SKIP == 0:
            # GAZE DETECTION (using RetinaFace)
            if self.gaze_settings.FACE_DETECTION_SCALE < 1.0:
                small = cv2.resize(
                    frame, None,
                    fx=self.gaze_settings.FACE_DETECTION_SCALE,
                    fy=self.gaze_settings.FACE_DETECTION_SCALE
                )
                bboxes, _ = self.face_detector.detect(small)
                if len(bboxes) > 0:
                    bboxes = bboxes / self.gaze_settings.FACE_DETECTION_SCALE
            else:
                bboxes, _ = self.face_detector.detect(frame)
            time_calculator(start_time, "Face detect")
            
            if len(bboxes) > 0:
                start_time = time.time()
                bbox = bboxes[0]
                self.last_bbox = bbox
                gaze_result = self.process_gaze(frame, bbox)
                
                if gaze_result:
                    start_time = time.time()
                    direction, confidence, pitch, yaw = gaze_result
                    
                    # Update state
                    with self.lock:
                        self.current_gaze_direction = direction
                        self.current_confidence = confidence
                        self.current_pitch = pitch
                        self.current_yaw = yaw
                    time_calculator(start_time, "Data extraction")
                    # Move mouse based on gaze
                    start_time = time.time()
                    if self.mouse_controller.is_active and \
                       confidence > self.gaze_settings.CONFIDENCE_THRESHOLD:
                        self.mouse_controller.move_mouse(direction, confidence)
                        time_calculator(start_time, "Move mouse")
        
        start_time = time.time()
        # BLINK DETECTION (using MediaPipe - runs every frame for responsiveness)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            blink_result = self.process_blink(frame, face_landmarks)
            
            if blink_result:
                action, scroll_mode, left_ear, right_ear = blink_result
        time_calculator(start_time, "Blinking")
        # Draw visualization
        start_time = time.time()
        self.draw_visualization(
            frame, fps, gaze_result, blink_result, scroll_mode
        )
        time_calculator(start_time, "visualization")
        return frame
    
    def draw_visualization(self, frame, fps, gaze_result, blink_result, scroll_mode):
        """Draw comprehensive visualization overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 35
        line_height = 30
        
        # === GAZE INFORMATION ===
        cv2.putText(frame, "GAZE TRACKING", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height
        
        with self.lock:
            direction = self.current_gaze_direction
            confidence = self.current_confidence
            pitch = self.current_pitch
            yaw = self.current_yaw
        
        # Direction with color coding
        is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                   GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
        dir_color = (255, 105, 180) if is_diagonal else (255, 165, 0)
        if direction == GazeDirection.CENTER:
            dir_color = (0, 255, 0)
        
        cv2.putText(frame, f"Direction: {direction.value}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dir_color, 2)
        y_pos += line_height
        
        # Confidence bar
        conf_width = int(200 * confidence)
        cv2.rectangle(frame, (20, y_pos-15), (20 + conf_width, y_pos-5),
                     dir_color, -1)
        cv2.rectangle(frame, (20, y_pos-15), (220, y_pos-5), (100, 100, 100), 1)
        cv2.putText(frame, f"{confidence:.0%}", (230, y_pos-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_height
        
        # Angles
        cv2.putText(frame, f"P:{pitch:+.1f}° Y:{yaw:+.1f}°", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_pos += line_height
        
        # === BLINK INFORMATION ===
        cv2.putText(frame, "BLINK DETECTION", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_pos += line_height
        
        if blink_result:
            action, scroll_mode_active, left_ear, right_ear = blink_result
            
            # EAR values
            cv2.putText(frame, f"L-EAR: {left_ear:.2f}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"R-EAR: {right_ear:.2f}", (200, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += line_height
            
            # Blink counters
            cv2.putText(frame, 
                       f"Both:{self.blink_detector.both_blink_frames} " +
                       f"L:{self.blink_detector.left_blink_frames} " +
                       f"R:{self.blink_detector.right_blink_frames}",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            y_pos += line_height
            
            # Scroll mode status
            scroll_color = (0, 255, 0) if scroll_mode_active else (100, 100, 100)
            cv2.putText(frame, f"Scroll: {'ON' if scroll_mode_active else 'OFF'}",
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       scroll_color, 2)
        
        y_pos += line_height + 10
        
        # === STATUS ===
        mouse_color = (0, 255, 0) if self.mouse_controller.is_active else (0, 0, 255)
        cv2.putText(frame, f"Mouse: {'ON' if self.mouse_controller.is_active else 'OFF'}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw face box if available
        if self.last_bbox is not None:
            x_min, y_min, x_max, y_max = map(int, self.last_bbox[:4])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                         (255, 255, 0), 2)
            
            # Draw direction arrow
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            self.draw_direction_arrow(frame, cx, cy, direction, confidence)
        
        # Instructions
        instructions = "Q:Quit | M:Mouse | R:Reset | +/-:Speed | D:Debug"
        cv2.putText(frame, instructions, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_direction_arrow(self, frame, cx, cy, direction, confidence):
        """Draw directional arrow indicating gaze"""
        if direction == GazeDirection.CENTER:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            return
        
        arrow_length = int(50 * confidence)
        angles = {
            GazeDirection.LEFT: 180,
            GazeDirection.RIGHT: 0,
            GazeDirection.UP: 90,
            GazeDirection.DOWN: 270,
            GazeDirection.LEFT_UP: 135,
            GazeDirection.RIGHT_UP: 45,
            GazeDirection.LEFT_DOWN: 225,
            GazeDirection.RIGHT_DOWN: 315
        }
        
        angle = angles.get(direction, 0)
        angle_rad = np.radians(angle)
        
        end_x = cx + int(arrow_length * np.cos(angle_rad))
        end_y = cy - int(arrow_length * np.sin(angle_rad))
        
        is_diagonal = direction in [GazeDirection.LEFT_UP, GazeDirection.RIGHT_UP,
                                   GazeDirection.LEFT_DOWN, GazeDirection.RIGHT_DOWN]
        color = (255, 105, 180) if is_diagonal else (255, 165, 0)
        
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
    
    def run(self):
        """Main application loop"""
        print("\nStarting unified eye tracking system...")
        print("\nFeatures Active:")
        print("  ✓ Gaze-based mouse movement (8 directions)")
        print("  ✓ Blink-based click actions")
        print("  ✓ Scroll mode with eyebrow raise")
        print("  ✓ Double-blink keyboard toggle")
        print("\nBlink Actions:")
        print("  • Double Both-Eye Blink → Toggle virtual keyboard")
        print("  • Single Both-Eye Blink → Left click")
        print("  • Right Eye Blink → Right click")
        print("  • Left Eye Blink → Toggle keyboard")
        print("\nControls:")
        print("  Q - Quit")
        print("  M - Toggle mouse control")
        print("  R - Reset system")
        print("  +/- - Adjust mouse speed")
        print("  D - Toggle debug mode")
        print("\n" + "="*80 + "\n")
        
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise IOError(f"Failed to open camera source {self.camera_source}")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x" +
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n")
        
        try:
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                time_calculator(start_time, "Cam capture")
                # Process frame
                start_time = time.time()
                processed_frame = self.process_frame(frame)
                time_calculator(start_time, "Frame processing Total")
                start_time = time.time()
                # Display
                cv2.imshow("Unified Eye Tracking Control", processed_frame)
                time_calculator(start_time, "Show image CV")



                # Handle keyboard input
                start_time = time.time()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Shutting down...")
                    break
                elif key == ord('m'):
                    self.mouse_controller.toggle_control()
                    status = "ON" if self.mouse_controller.is_active else "OFF"
                    print(f"Mouse control: {status}")
                elif key == ord('r'):
                    self.gaze_detector.reset()
                    self.gaze_filter.reset()
                    print("System reset")
                elif key == ord('+'):
                    self.gaze_settings.MOUSE_SPEED_BASE = min(
                        30, self.gaze_settings.MOUSE_SPEED_BASE + 2
                    )
                    self.mouse_controller.direction_vectors = \
                        self.mouse_controller._create_direction_vectors()
                    print(f"Speed: {self.gaze_settings.MOUSE_SPEED_BASE}")
                elif key == ord('-'):
                    self.gaze_settings.MOUSE_SPEED_BASE = max(
                        5, self.gaze_settings.MOUSE_SPEED_BASE - 2
                    )
                    self.mouse_controller.direction_vectors = \
                        self.mouse_controller._create_direction_vectors()
                    print(f"Speed: {self.gaze_settings.MOUSE_SPEED_BASE}")
                elif key == ord('d'):
                    self.gaze_settings.SHOW_DEBUG_INFO = \
                        not self.gaze_settings.SHOW_DEBUG_INFO
                    print(f"Debug: {'ON' if self.gaze_settings.SHOW_DEBUG_INFO else 'OFF'}")
                time_calculator(start_time, "Handle inputs")
        
        finally:
            self.mouse_controller.stop()
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Cleanup complete")
            print(f"Total frames processed: {self.frame_count}")


# Time function to calculate time of each step
def time_calculator(start_inf, label=""):
    return #remvoe the line if you want timings
    end_inf = time.time()
    calc_time = end_inf - start_inf
    print(f"{label} time: {calc_time*1000:.5f} ms")
def main():
    parser = argparse.ArgumentParser(
        description="Unified Eye Tracking Control System"
    )
    parser.add_argument("--source", type=int, default=0,
                       help="Camera source index")
    parser.add_argument("--model", type=str, default="mobileone_s0_gaze.onnx",
                       help="Path to gaze estimation model")
    parser.add_argument("--no-mouse", action="store_true",
                       help="Disable mouse control at start")
    parser.add_argument("--speed", type=int, default=12,
                       help="Base mouse speed (5-30)")
    parser.add_argument("--delay", type=float, default=0.3,
                       help="Movement delay in seconds")
    
    args = parser.parse_args()
    
    # Configure gaze settings
    gaze_settings = GazeSettings(
        VIDEO_SOURCE=args.source,
        MODEL_PATH=args.model,
        ENABLE_MOUSE_CONTROL=not args.no_mouse,
        MOUSE_SPEED_BASE=args.speed,
        MOVEMENT_DELAY=args.delay,
        SHOW_VISUALIZATION=True,
        SHOW_DEBUG_INFO=False,
        USE_KALMAN_FILTER=True,
        SHOW_FPS=True
    )
    
    # Create and run unified tracker
    tracker = UnifiedEyeTracker(gaze_settings, args.source)
    tracker.run()


if __name__ == "__main__":
    main()


# command to run: 
# python driver.py --model mobileone_s0_gaze.onnx --speed 20 --delay 0.3

#Show image CV time: 0.00000 ms
#Handle inputs time: 8.52394 ms (sometimes 1-2ms)
#Cam capture time: 8.45599 ms
#Face detect time: 84.22136 ms (avg is 60 tho)
#Model inference time: 26.95608 ms (avg is 30)
#Direction detection time: 0.99802 ms (rarly 10 ms)
#Filter time: 0.00000 ms
#Data extraction time: 0.00000 ms
#Move mouse time: 0.00000 ms (though i didn't move the mouse any time)
#Blinking time: 8.87418 ms ) (avg 6)
#visualization time: 0.99778 ms (sometimes 14ms but not common)
#Frame processing Total time: 122.04742 ms (avg is 100ms)
#total: 100+8+8+8+1=125 ms, worst case is 8.5+9+122+6+14=160 ms
#Frame skipping doubles it, so 12.5

