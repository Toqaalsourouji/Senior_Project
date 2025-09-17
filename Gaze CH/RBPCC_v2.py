import cv2
import logging
import argparse
import warnings
import numpy as np
import math
import time
import mediapipe as mp
import pyautogui
from typing import Tuple, Optional
import onnxruntime as ort

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
pyautogui.FAILSAFE = False

def softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Draws gaze direction on a frame given bounding box and gaze angles."""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))
    
    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)
    
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(frame, point1, point2, color=color, thickness=thickness,
                   line_type=cv2.LINE_AA, tipLength=0.25)

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    """Draw bounding box with corner emphasis."""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    
    width = x_max - x_min
    height = y_max - y_min
    corner_length = int(proportion * min(width, height))
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
    
    # Corner lines for emphasis
    corners = [
        [(x_min, y_min), (x_min + corner_length, y_min), (x_min, y_min + corner_length)],
        [(x_max, y_min), (x_max - corner_length, y_min), (x_max, y_min + corner_length)],
        [(x_min, y_max), (x_min, y_max - corner_length), (x_min + corner_length, y_max)],
        [(x_max, y_max), (x_max, y_max - corner_length), (x_max - corner_length, y_max)]
    ]
    
    for corner in corners:
        cv2.line(image, corner[0], corner[1], color, thickness)
        cv2.line(image, corner[0], corner[2], color, thickness)

def draw_bbox_gaze(frame: np.ndarray, bbox, pitch, yaw):
    """Draw both bounding box and gaze direction."""
    draw_bbox(frame, bbox)
    draw_gaze(frame, bbox, pitch, yaw)

def eye_aspect_ratio(landmarks, eye_indices) -> float:
    """Calculate eye aspect ratio for blink detection."""
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Tracking for Raspberry Pi")
    parser.add_argument("--source", type=str, required=True, help="Camera index or video path")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.onnx)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--no-mouse", action="store_true", help="Disable mouse control")
    parser.add_argument("--no-display", action="store_true", help="Run headless without display")
    parser.add_argument("--no-blink", action="store_true", help="Disable blink detection for performance")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame (default: 3)")
    return parser.parse_args()

class GazeEstimation:
    """Lightweight gaze estimation for Raspberry Pi."""
    
    def __init__(self, model_path: str):
        # Model configuration
        self.input_size = (448, 448)
        self.input_mean = np.array([0.485, 0.456, 0.406])
        self.input_std = np.array([0.229, 0.224, 0.225])
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        
        # Load ONNX model with providers for ARM
        providers = ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image using OpenCV (no PyTorch required)."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Convert to float and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        normalized = (normalized - self.input_mean) / self.input_std
        
        # Transpose from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(transposed, axis=0).astype(np.float32)
        
        return batch
    
    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        """Estimate gaze angles from face image."""
        if face_image.size == 0:
            return 0.0, 0.0
        
        # Preprocess the image
        input_tensor = self.preprocess(face_image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch_logits, yaw_logits = outputs
        
        # Apply softmax using numpy
        pitch_probs = softmax(pitch_logits)
        yaw_probs = softmax(yaw_logits)
        
        # Calculate angles
        pitch = np.sum(pitch_probs * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor) * self._binwidth - self._angle_offset
        
        return np.radians(pitch), np.radians(yaw)

class OptimizedFaceDetector:
    """Optimized face detector with tracking for better performance."""
    
    def __init__(self):
        # Use Haar Cascade for initial detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        
        # Initialize tracker
        self.tracker = None
        self.tracking = False
        self.detection_scale = 0.5  # Scale down for detection
        self.frames_since_detection = 0
        self.redetect_interval = 30  # Redetect every 30 frames
    
    def detect_faces(self, frame):
        """Detect faces in frame with downscaling for speed."""
        # Downscale for detection
        small_frame = cv2.resize(frame, None, fx=self.detection_scale, fy=self.detection_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,  # Faster scanning
            minNeighbors=4,   # Balance between speed and accuracy
            minSize=(40, 40)  # Skip very small faces
        )
        
        # Scale coordinates back up
        bboxes = []
        for (x, y, w, h) in faces:
            scale_inv = 1.0 / self.detection_scale
            x, y, w, h = int(x * scale_inv), int(y * scale_inv), int(w * scale_inv), int(h * scale_inv)
            bboxes.append([x, y, x + w, y + h, 1.0])
        
        return bboxes
    
    def detect_or_track(self, frame):
        """Use tracking when possible, redetect periodically."""
        self.frames_since_detection += 1
        
        # Try tracking first if we have a tracker
        if self.tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            
            if success and self.frames_since_detection < self.redetect_interval:
                # Successful tracking
                x, y, w, h = [int(v) for v in box]
                return [[x, y, x + w, y + h, 1.0]], None
            else:
                # Tracking failed or time to redetect
                self.tracking = False
                self.tracker = None
        
        # Perform detection
        bboxes = self.detect_faces(frame)
        
        if bboxes:
            # Initialize tracker with first detected face
            bbox = bboxes[0]
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
            
            # Use KCF tracker (good balance of speed and accuracy)
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, (x, y, w, h))
            self.tracking = True
            self.frames_since_detection = 0
        
        return bboxes, None

def project_to_2d(pitch_rad: float, yaw_rad: float, width: int, height: int,
                 face_center_x: int, face_center_y: int) -> Tuple[int, int]:
    """Project 3D gaze to 2D screen coordinates."""
    arrow_dx = -np.sin(pitch_rad) * np.cos(yaw_rad)
    arrow_dy = -np.sin(yaw_rad)
    screen_scale = min(width, height) * 0.8
    gaze_x = face_center_x + int(arrow_dx * screen_scale)
    gaze_y = face_center_y + int(arrow_dy * screen_scale)
    return (
        np.clip(gaze_x, 0, width - 1),
        np.clip(gaze_y, 0, height - 1)
    )

def main():
    args = parse_args()
    
    # Initialize components
    print("Initializing gaze estimation system...")
    print(f"Frame skip: Processing every {args.frame_skip} frames")
    print(f"Blink detection: {'Disabled' if args.no_blink else 'Enabled'}")
    
    try:
        engine = GazeEstimation(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    detector = OptimizedFaceDetector()
    
    # Initialize MediaPipe face mesh for blink detection (if enabled)
    face_mesh = None
    if not args.no_blink:
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Blink detection initialized")
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            print("Continuing without blink detection")
    
    # Open video capture
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                               cap.get(cv2.CAP_PROP_FPS) or 15, (width, height))
    
    # Blink detection variables
    blink_counter = 0
    BLINK_CONSEC_FRAMES = 1
    
    # Performance tracking
    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0
    
    print("Starting main loop. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every second
        if time.time() - fps_timer > 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        
        # Skip frames for performance
        if frame_count % args.frame_skip != 0:
            continue
        
        # Detect or track faces
        bboxes, _ = detector.detect_or_track(frame)
        
        for bbox in bboxes[:1]:  # Process only first face for performance
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                continue
            
            # Estimate gaze
            pitch, yaw = engine.estimate(face_img)
            
            # Visualization
            draw_bbox_gaze(frame, bbox, pitch, yaw)
            face_center_x = (x_min + x_max) // 2
            face_center_y = (y_min + y_max) // 2
            gaze_x, gaze_y = project_to_2d(pitch, yaw, frame.shape[1], frame.shape[0],
                                         face_center_x, face_center_y)
            cv2.circle(frame, (gaze_x, gaze_y), 8, (0, 0, 255), -1)
            
            # Mouse control (if enabled)
            if not args.no_mouse:
                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2
                deadzone_threshold = 0.4 * min(frame_width, frame_height) / 2
                speed = 8  # Optimized for Pi Zero
                dx = gaze_x - frame_center_x
                dy = gaze_y - frame_center_y
                
                if abs(dx) > deadzone_threshold or abs(dy) > deadzone_threshold:
                    dir_x = 1 if dx > 0 else -1 if dx < 0 else 0
                    dir_y = 1 if dy > 0 else -1 if dy < 0 else 0
                    
                    if abs(dx) > abs(dy):
                        pyautogui.moveRel(dir_x * speed, 0, duration=0.01)
                    else:
                        pyautogui.moveRel(0, dir_y * speed, duration=0.01)
                else:
                    cv2.putText(frame, "DZ", (frame.shape[1] - 40, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Blink detection (if enabled and MediaPipe is available)
            if face_mesh is not None and not args.no_blink:
                # Only process blink detection every few frames for performance
                if frame_count % (args.frame_skip * 2) == 0:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        h, w, _ = frame.shape
                        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                        
                        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        if avg_ear < 0.2:
                            blink_counter += 1
                            if blink_counter >= BLINK_CONSEC_FRAMES and not args.no_mouse:
                                print("BLINK - Click!")
                                pyautogui.click()
                                blink_counter = 0
                                # Visual feedback for blink
                                cv2.putText(frame, "BLINK!", (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            blink_counter = 0
        
        # Display frame (unless running headless)
        if not args.no_display:
            # Add performance info
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if detector.tracking:
                cv2.putText(frame, "Mode: Tracking", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "Mode: Detecting", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Gaze Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if writer:
            writer.write(frame)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()
