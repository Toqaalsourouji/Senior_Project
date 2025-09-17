#!/usr/bin/env python3
import cv2
import numpy as np
import time
import sys
import pyautogui
from typing import Tuple
import onnxruntime as ort
import argparse

# Configure pyautogui for Raspberry Pi
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove default pause between commands

def softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class FastGazeEstimation:
    """Optimized gaze estimation for Pi Zero."""
    
    def __init__(self, model_path: str):
        self.input_size = (448, 448)
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        # Load ONNX model
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] Model loaded: {model_path}")
    
    def preprocess_fast(self, image: np.ndarray) -> np.ndarray:
        """Fast preprocessing without unnecessary conversions."""
        # Resize directly (skip RGB conversion for speed)
        resized = cv2.resize(image, self.input_size)
        
        # Fast normalization
        normalized = resized.astype(np.float32) * (1.0/255.0)
        normalized = (normalized - self.input_mean) / self.input_std
        
        # Transpose and add batch
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0).astype(np.float32)
    
    def estimate(self, face_image: np.ndarray) -> Tuple[float, float, str]:
        """Estimate gaze angles and return direction."""
        if face_image.size == 0:
            return 0.0, 0.0, "NO_FACE"
        
        input_tensor = self.preprocess_fast(face_image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        pitch_probs = softmax(outputs[0])
        yaw_probs = softmax(outputs[1])
        
        pitch = np.sum(pitch_probs * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor) * self._binwidth - self._angle_offset
        
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Get direction
        direction = self.get_direction(pitch_rad, yaw_rad)
        
        return pitch_rad, yaw_rad, direction
    
    def get_direction(self, pitch_rad: float, yaw_rad: float, threshold: float = 0.2) -> str:
        """Get gaze direction as string."""
        vertical = ""
        horizontal = ""
        
        if pitch_rad > threshold:
            vertical = "UP"
        elif pitch_rad < -threshold:
            vertical = "DOWN"
        
        if yaw_rad > threshold:
            horizontal = "RIGHT"
        elif yaw_rad < -threshold:
            horizontal = "LEFT"
        
        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        elif vertical:
            return vertical
        elif horizontal:
            return horizontal
        else:
            return "CENTER"

class UltraFastFaceDetector:
    """Ultra-fast face detection optimized for Pi Zero."""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.last_face = None
        self.face_lost_frames = 0
        self.detection_scale = 0.25  # Very aggressive downscaling
    
    def detect_fast(self, frame):
        """Ultra-fast detection with aggressive optimization."""
        # Use last known position if recent
        if self.last_face is not None and self.face_lost_frames < 10:
            self.face_lost_frames += 1
            return [self.last_face]
        
        # Aggressive downscale
        small_frame = cv2.resize(frame, None, fx=self.detection_scale, fy=self.detection_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            scale_inv = 1.0 / self.detection_scale
            bbox = [
                int(x * scale_inv),
                int(y * scale_inv),
                int((x + w) * scale_inv),
                int((y + h) * scale_inv),
                1.0
            ]
            self.last_face = bbox
            self.face_lost_frames = 0
            return [bbox]
        
        return []

class MouseController:
    """Fixed mouse controller for Raspberry Pi."""
    
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        self.last_move_time = time.time()
        self.move_threshold = 0.05  # Minimum time between moves
        
        # Try to move mouse to center initially
        try:
            pyautogui.moveTo(self.center_x, self.center_y, duration=0)
            print(f"[INFO] Mouse initialized at center: {self.center_x}, {self.center_y}")
            print(f"[INFO] Screen size: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(f"[WARNING] Mouse initialization failed: {e}")
    
    def move_by_gaze(self, direction: str, pitch_rad: float, yaw_rad: float):
        """Move mouse based on gaze direction."""
        current_time = time.time()
        if current_time - self.last_move_time < self.move_threshold:
            return
        
        # Calculate movement speed based on angle
        base_speed = 20
        pitch_factor = abs(np.degrees(pitch_rad)) / 30.0  # Normalize to 0-1 range
        yaw_factor = abs(np.degrees(yaw_rad)) / 30.0
        
        dx = 0
        dy = 0
        
        if "LEFT" in direction:
            dx = -int(base_speed * min(yaw_factor, 1.0))
        elif "RIGHT" in direction:
            dx = int(base_speed * min(yaw_factor, 1.0))
        
        if "UP" in direction:
            dy = -int(base_speed * min(pitch_factor, 1.0))
        elif "DOWN" in direction:
            dy = int(base_speed * min(pitch_factor, 1.0))
        
        if dx != 0 or dy != 0:
            try:
                # Use moveRel for relative movement
                pyautogui.moveRel(dx, dy, duration=0)
                print(f"[MOUSE] Moving: {direction} (dx:{dx:+3d}, dy:{dy:+3d})")
                self.last_move_time = current_time
            except Exception as e:
                print(f"[ERROR] Mouse move failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Headless Gaze Tracker")
    parser.add_argument("--source", type=str, default="0", help="Camera index")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--no-mouse", action="store_true", help="Disable mouse control")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("[INFO] Starting Headless Gaze Tracker")
    print(f"[INFO] Frame skip: {args.frame_skip}")
    print(f"[INFO] Mouse control: {'Disabled' if args.no_mouse else 'Enabled'}")
    
    # Initialize components
    engine = FastGazeEstimation(args.model)
    detector = UltraFastFaceDetector()
    mouse = MouseController() if not args.no_mouse else None
    
    # Open camera with minimal resolution
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    # Set minimal resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for less lag
    
    print(f"[INFO] Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
    print("[INFO] Press Ctrl+C to quit")
    print("-" * 50)
    
    # Performance tracking
    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    last_direction = "CENTER"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            if time.time() - fps_start > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            # Skip frames for performance
            if frame_count % args.frame_skip != 0:
                continue
            
            # Detect face
            faces = detector.detect_fast(frame)
            
            if faces:
                bbox = faces[0]
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # Estimate gaze
                    pitch, yaw, direction = engine.estimate(face_img)
                    
                    # Print status only if changed or verbose
                    if direction != last_direction or args.verbose:
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        print(f"[GAZE] {direction:<15} Pitch:{pitch_deg:+6.1f}° Yaw:{yaw_deg:+6.1f}° FPS:{current_fps:2d}")
                        last_direction = direction
                    
                    # Move mouse
                    if mouse and direction != "CENTER":
                        mouse.move_by_gaze(direction, pitch, yaw)
            else:
                if args.verbose:
                    print(f"[INFO] No face detected - FPS: {current_fps}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()
