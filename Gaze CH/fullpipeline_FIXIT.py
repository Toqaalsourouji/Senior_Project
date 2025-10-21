import cv2
import logging
import argparse 
import warnings
import numpy as np
import math
import mediapipe as mp
from pynput.mouse import Controller, Button
import uniface
from typing import Tuple
import onnxruntime as ort
import time 
import tkinter as tk

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

#constants for blink detection (can me edited and epxermented with)
EAR_CLOSE_THRESHOLD = 0.20 # Eye Aspect Ratio threshold to indicate closed eyes
EAR_OPEN_THRESHOLD = 0.25  # Eye Aspect Ratio threshold to indicate open eyes
BLINK_CONSEC_FRAMES = 2  # Number of consecutive frames the eye must be below the threshold
ACTION_COOLDOWN_MS = 300  # Cooldown period after an action is triggered to avoid multiple triggers, i.e, debounce
SCROLL_AMOUNT = 250  # Amount to scroll on each blink action (POSTIVE for up, NEGATIVE for down ) #check this approach and maybe change


Frame = cv2.VideoCapture(1)

def time_calculator(start_inf, label=""):
    end_inf = time.time()
    calc_time = end_inf - start_inf
    print(f"{label} time: {calc_time*1000:.5f} ms")

LEFT_EYE = [362, 385, 387, 263, 373, 380] # Right eye indices from MediaPipe using dlib library
RIGHT_EYE = [33, 160, 158, 133, 153, 144] # Left eye indices from MediaPipe using dlib library

def now_ms(): #current time in milliseconds
    return int(time.time() * 1000)

def eye_center(landmarks, eye_indices): #
    xs = [landmarks[i][0] for i in eye_indices]
    ys = [landmarks[i][1] for i in eye_indices]
    return (sum(xs) / len(xs), sum(ys) / len(ys) ) #to get the center of the eye

def polygon_center(landmarks, idxs):
    xs = [landmarks[i][0] for i in idxs]
    ys = [landmarks[i][1] for i in idxs]
    return (sum(xs) / len(xs), sum(ys) / len(ys) ) #to get the center of the eye

import time

def profile_pipeline(cap, engine, detector, face_mesh, num_frames=30):
    """Profile each component to find bottlenecks."""
    
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING")
    print("="*80)
    
    timings = {
        'frame_read': [],
        'face_detection': [],
        'gaze_estimation': [],
        'face_mesh': [],
        'total': []
    }
    
    for i in range(num_frames):
        frame_start = time.time()
        
        # 1. Frame reading
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        timings['frame_read'].append(time.time() - t0)
        
        # 2. Face detection
        t0 = time.time()
        bboxes, _ = detector.detect(frame)
        timings['face_detection'].append(time.time() - t0)
        
        if len(bboxes) > 0:
            # 3. Gaze estimation
            x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                t0 = time.time()
                pitch, yaw = engine.estimate(face_img)
                timings['gaze_estimation'].append(time.time() - t0)
        
        # 4. Face mesh (for blink detection)
        t0 = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        timings['face_mesh'].append(time.time() - t0)
        
        timings['total'].append(time.time() - frame_start)
    
    # Print results
    print(f"\nTested {num_frames} frames:")
    print("-" * 80)
    
    for component, times in timings.items():
        if times:
            avg_time = np.mean(times) * 1000  # Convert to ms
            max_time = np.max(times) * 1000
            fps = 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            print(f"{component:20s}: {avg_time:6.1f}ms avg, {max_time:6.1f}ms max, {fps:5.1f} FPS")
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS:")
    print("="*80)
    
    # Find the slowest component
    avg_times = {k: np.mean(v) * 1000 for k, v in timings.items() if v and k != 'total'}
    slowest = max(avg_times, key=avg_times.get)
    print(f"\nSlowest component: {slowest} ({avg_times[slowest]:.1f}ms)")
    
    # Calculate percentages
    total_avg = np.mean(timings['total']) * 1000
    print(f"\nTime breakdown:")
    for component, times in timings.items():
        if times and component != 'total':
            pct = (np.mean(times) * 1000 / total_avg) * 100
            print(f"  {component:20s}: {pct:5.1f}%")

class FPSCounter:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

def right_eye_vertical_gaze(landmakrs, right_eye_indices): #function to determine vertical gaze direction for scrolling 
    """" 
    Hueristics to determine vertical gaze direction based on eye landmarks.: if the iris landmakrs exists
    (refine_landmarks=True in mp.solutions.face_mesh.FaceMesh), we can use the iris center vs eyelid box.
    Else, use eyeball center proxy from the eye polygon.
    returns Up or down
    """
    top_candidates = [min(right_eye_indices, key = lambda i: landmakrs[i][1])]
    bottom_candidates = [max(right_eye_indices, key = lambda i: landmakrs[i][1])]
    top_y = sum(landmakrs[i][1] for i in top_candidates) / len(top_candidates)
    bottom_y = sum(landmakrs[i][1] for i in bottom_candidates) / len(bottom_candidates)

    #choose iris center if available
    try:
        iris_cx, iris_cy = polygon_center(landmakrs, RIGHT_EYE)
        center_y = iris_cy
    except:
        #use gometric eye center
        _,center_y = eye_center(landmakrs, RIGHT_EYE)
    
    # nromalized position of the center within the eye box ( 0= top, 1=bottom)
    denom = max(1, bottom_y - top_y)
    ratio = (center_y - top_y) / denom

    #Thresholds can be adjusted based on user testing
    if ratio < 0.35:
        return "UP"
    elif ratio > 0.65:
        return "DOWN"
    else:
        return "CENTER"

#set up and store calibration data pairs: [gaze_predection] -> [screen_point]
calibration_gaze= []
calibration_screen= []

#define 5 calibration points on the screen (normalized coordinates [0,1])

calibration_points =[
    (0.1,0.1), #top left
    (0.9,0.1), #top right
    (0.5,0.5), #center
    (0.1,0.9), #bottom left
    (0.9,0.9)  #bottom right
]


def extract_eyes_with_mediapipe(frame, face_landmarks, face_bbox):
    """
    Extract eye regions using MediaPipe landmarks.
    """
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = map(int, face_bbox[:4])
    
    # MediaPipe eye landmark indices
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385]
    
    # Get landmarks
    landmarks = face_landmarks.landmark
    
    # Get bounding box for both eyes
    left_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
    right_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]
    
    all_eye_points = left_eye_points + right_eye_points
    
    # Find bounding box around both eyes
    xs = [p[0] for p in all_eye_points]
    ys = [p[1] for p in all_eye_points]
    
    eye_x_min = max(0, min(xs) - 20)
    eye_x_max = min(w, max(xs) + 20)
    eye_y_min = max(0, min(ys) - 20)
    eye_y_max = min(h, max(ys) + 20)
    
    # Extract eye region
    eye_region = frame[eye_y_min:eye_y_max, eye_x_min:eye_x_max]
    
    return eye_region

def extract_eye_region_simple(face_img):
    """
    Extract eye region from face image.
    Simple version without landmarks.
    """
    h, w = face_img.shape[:2]
    
    # Eye region is typically in upper-middle 40% of face
    eye_y_start = int(h * 0.25)
    eye_y_end = int(h * 0.55)
    eye_x_start = int(w * 0.15)
    eye_x_end = int(w * 0.85)
    
    eye_region = face_img[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
    
    if eye_region.size == 0:
        return face_img  # Fallback to full face
    
    return eye_region

def normalize_for_mpiigaze(image):
    """
    Normalize to match MPIIGaze training data.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR for model
    bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return bgr

def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Draws gaze direction on a frame given bounding box and gaze angles."""
    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Calculate center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Handle grayscale frames by converting them to BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Calculate the direction of the gaze
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)

    # Draw gaze direction
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(
        frame,
        point1,
        point2,
        color=color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25
    )



def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    width = x_max - x_min
    height = y_max - y_min

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_bbox_gaze(frame: np.ndarray, bbox, pitch, yaw):
    draw_bbox(frame, bbox)
    draw_gaze(frame, bbox, pitch, yaw)

def eye_aspect_ratio(landmarks, eye_indices) -> float:
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Tracking")
    parser.add_argument("--source", type=str, required=True, help="Camera index or video path")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.onnx)")
    parser.add_argument("--output", type=str, default=None, help="Output video path, ingore this")
    return parser.parse_args()

class GazeEstimationTorch:
    def __init__(self, model_path: str):
        self.input_size = (224, 224)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self._bins = 28
        self._binwidth = 3.0
        self._angle_offset = 42.0
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        # CRITICAL: Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=sess_options
        )
        
        print(f"ONNX Providers: {self.session.get_providers()}")
        
        # Pre-compute normalization values for speed
        self.mean = np.array(self.input_mean).reshape(1, 3, 1, 1).astype(np.float32)
        self.std = np.array(self.input_std).reshape(1, 3, 1, 1).astype(np.float32)
        
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_fast(self, image: np.ndarray) -> np.ndarray:
        """Faster preprocessing using pure numpy/cv2."""
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # To float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Normalize with mean and std
        img = (img - self.mean) / self.std
        
        return img

    def estimate(self, face_image: np.ndarray, use_eye_extraction=True) -> Tuple[float, float]:
        """
        Estimate gaze with optional eye extraction.
        """
        if use_eye_extraction:
            # Extract eye region
            eye_region = extract_eye_region_simple(face_image)
            # Normalize
            processed = normalize_for_mpiigaze(eye_region)
        else:
            processed = face_image
        
        # Preprocess for model
        input_tensor = self.preprocess_fast(processed)
        
        # ONNX inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch, yaw = outputs
        
        # Softmax
        pitch_exp = np.exp(pitch - np.max(pitch, axis=1, keepdims=True))
        pitch = pitch_exp / np.sum(pitch_exp, axis=1, keepdims=True)
        
        yaw_exp = np.exp(yaw - np.max(yaw, axis=1, keepdims=True))
        yaw = yaw_exp / np.sum(yaw_exp, axis=1, keepdims=True)
        
        # Convert to angles
        pitch = np.sum(pitch * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw * self.idx_tensor) * self._binwidth - self._angle_offset

        return np.radians(pitch), np.radians(yaw)




def run_calibration(cap, engine, detector):
    """Enhanced calibration with visual feedback and validation."""
    global calibration_gaze, calibration_screen
    
    # Reset calibration data
    calibration_gaze = []
    calibration_screen = []
    
    root = tk.Tk()
    root.withdraw()  # hide the window
    swidth = root.winfo_screenwidth()
    sheight = root.winfo_screenheight()
    root.destroy()
    print("\n" + "="*80)
    print("CALIBRATION STARTING")
    print("="*80)
    print(f"Screen size: {swidth}x{sheight}")
    print(f"Calibration points: {len(calibration_points)}")
    print("\nInstructions:")
    print("  - Look at the GREEN circle when it appears")
    print("  - Keep your head still during calibration")
    print("  - The system will capture your gaze automatically")
    print("\nGet ready...")
    
    # 5-second countdown before calibration starts
    for countdown in range(5, 0, -1):
        dummy_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
        
        # Large countdown number in center
        text = str(countdown)
        font_scale = 8
        thickness = 15
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (swidth - text_size[0]) // 2
        text_y = (sheight + text_size[1]) // 2
        
        cv2.putText(dummy_frame, text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # Instruction text at top
        instruction = "Calibration starting in..."
        cv2.putText(dummy_frame, instruction, 
                   (swidth//2 - 250, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Additional instruction at bottom
        bottom_text = "Look at the green circles and keep your head still"
        cv2.putText(dummy_frame, bottom_text, 
                   (swidth//2 - 400, sheight - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Calibration", dummy_frame)
        cv2.waitKey(1000)  # Wait 1 second
        print(f"  {countdown}...")
    
    print("  Starting calibration NOW!\n")
    
    successful_points = 0
    
    for idx, (norm_x, norm_y) in enumerate(calibration_points):
        target_x, target_y = int(norm_x * swidth), int(norm_y * sheight)
        
        print(f"\nPoint {idx + 1}/{len(calibration_points)}: ({target_x}, {target_y})")
        
        # Show countdown for each calibration point
        for countdown in range(3, 0, -1):
            calib_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
            
            # Draw target circle
            cv2.circle(calib_frame, (target_x, target_y), 30, (0, 255, 0), -1)
            cv2.circle(calib_frame, (target_x, target_y), 35, (0, 255, 0), 3)
            
            # Countdown inside circle
            cv2.putText(calib_frame, str(countdown), 
                       (target_x - 20, target_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Point number at top
            cv2.putText(calib_frame, f"Point {idx + 1}/{len(calibration_points)}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Instruction
            cv2.putText(calib_frame, "Look at the GREEN circle", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", calib_frame)
            cv2.waitKey(1000)
        
        # Capture multiple frames and average
        gaze_samples = []
        for sample_idx in range(5):  # Take 5 samples
            ret, frame = cap.read()
            if not ret:
                print("  ✗ Failed to read frame")
                continue
            
            bboxes, _ = detector.detect(frame)
            if len(bboxes) == 0:
                print("  ✗ No face detected")
                continue
            
            x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                print("  ✗ Invalid face crop")
                continue
            
            try:
                pitch, yaw = engine.estimate(face_img)
                gaze_samples.append((pitch, yaw))
                
                # Show visual feedback during capture
                calib_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
                
                # Pulsing circle effect
                radius = 30 + (sample_idx * 5)
                cv2.circle(calib_frame, (target_x, target_y), radius, (0, 255, 0), -1)
                
                # Progress indicator
                cv2.putText(calib_frame, f"Capturing {sample_idx + 1}/5", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show gaze angles
                cv2.putText(calib_frame, f"Pitch: {np.degrees(pitch):.1f}deg", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(calib_frame, f"Yaw: {np.degrees(yaw):.1f}deg", 
                           (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Progress bar
                bar_width = 400
                bar_height = 30
                bar_x = (swidth - bar_width) // 2
                bar_y = sheight - 100
                
                cv2.rectangle(calib_frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), 
                            (255, 255, 255), 2)
                
                progress = int((sample_idx + 1) / 5 * bar_width)
                cv2.rectangle(calib_frame, (bar_x, bar_y), 
                            (bar_x + progress, bar_y + bar_height), 
                            (0, 255, 0), -1)
                
                cv2.imshow("Calibration", calib_frame)
                cv2.waitKey(100)
                
            except Exception as e:
                print(f"  ✗ Estimation failed: {e}")
                continue
        
        # Average the samples
        if len(gaze_samples) >= 3:  # Need at least 3 good samples
            avg_pitch = np.mean([p for p, y in gaze_samples])
            avg_yaw = np.mean([y for p, y in gaze_samples])
            calibration_gaze.append((avg_pitch, avg_yaw))
            calibration_screen.append((target_x, target_y))
            successful_points += 1
            print(f"  ✓ Captured (pitch: {np.degrees(avg_pitch):.1f}°, yaw: {np.degrees(avg_yaw):.1f}°)")
            
            # Show success feedback
            success_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
            cv2.circle(success_frame, (target_x, target_y), 50, (0, 255, 0), -1)
            cv2.putText(success_frame, "SUCCESS!", 
                       (target_x - 70, target_y - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", success_frame)
            cv2.waitKey(500)
        else:
            print(f"  ✗ Failed - only got {len(gaze_samples)}/5 samples")
            
            # Show failure feedback
            fail_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
            cv2.circle(fail_frame, (target_x, target_y), 50, (0, 0, 255), -1)
            cv2.putText(fail_frame, "FAILED", 
                       (target_x - 60, target_y - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibration", fail_frame)
            cv2.waitKey(500)
    
    # Show completion message
    complete_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
    cv2.putText(complete_frame, "CALIBRATION COMPLETE", 
               (swidth//2 - 300, sheight//2 - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(complete_frame, f"{successful_points}/{len(calibration_points)} points captured", 
               (swidth//2 - 200, sheight//2 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Calibration", complete_frame)
    cv2.waitKey(2000)
    
    cv2.destroyWindow("Calibration")
    
    print("\n" + "="*80)
    print(f"CALIBRATION COMPLETE: {successful_points}/{len(calibration_points)} points")
    print("="*80)
    
    if successful_points < 3:
        print("⚠ WARNING: Not enough calibration points! Using fallback projection.")
        return None
    
    # Fit regression model
    calibration_gaze = np.array(calibration_gaze)
    calibration_screen = np.array(calibration_screen)
    X = np.hstack([calibration_gaze, np.ones((len(calibration_gaze), 1))])
    Y = calibration_screen
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    
    # Calculate calibration error
    predicted = X @ coeffs
    errors = np.linalg.norm(predicted - Y, axis=1)
    mean_error = np.mean(errors)
    print(f"\nCalibration accuracy:")
    print(f"  Mean error: {mean_error:.1f} pixels")
    print(f"  Max error: {np.max(errors):.1f} pixels")
    
    return coeffs

def predict_with_calibration(pitch, yaw, coeffs):
    gaze_vector = np.array([pitch, yaw, 1.0])#add bias term
    screen_point = gaze_vector@coeffs
    return int(screen_point[0]), int(screen_point[1])

def project_to_2d(pitch_rad: float, yaw_rad: float, width: int, height: int, 
                 face_center_x: int, face_center_y: int) -> Tuple[int, int]:
    arrow_dx = -np.sin(pitch_rad) * np.cos(yaw_rad)
    arrow_dy = -np.sin(yaw_rad)
    screen_scale = min(width, height) * 0.8
    gaze_x = face_center_x + int(arrow_dx * screen_scale)
    gaze_y = face_center_y + int(arrow_dy * screen_scale)
    return (
        np.clip(gaze_x, 0, width-1),
        np.clip(gaze_y, 0, height-1)
    )

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("GAZE TRACKING SYSTEM")
    print("="*80)
    
    left_blink_frames = 0
    right_blink_frames = 0
    both_blink_frames = 0
    last_blink_ms = 0 
    
    # Initialize components
    print("\nInitializing model...")
    engine = GazeEstimationTorch(args.model)
    print("✓ Model loaded")
    
    print("Initializing face detector...")
    detector = uniface.RetinaFace()
    print("✓ Face detector loaded")
    
    # Open camera
    print(f"\nOpening camera {args.source}...")
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        print("Try: --source 0, --source 1, or --source 2")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Run calibration
    coeffs = run_calibration(cap, engine, detector)
    
    # Initialize face mesh
    print("\nInitializing face mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ Face mesh loaded")
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # Video writer setup
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 
                               cap.get(cv2.CAP_PROP_FPS) or 30, (width, height))
    
    print("\n" + "="*80)
    print("TRACKING STARTED - Press 'q' to quit")
    print("="*80 + "\n")
    
    frame_count = 0
    # Run profiling
    """print("\nRunning performance analysis...")
    profile_pipeline(cap, engine, detector, face_mesh, num_frames=30)
    
    response = input("\nContinue with tracking? (y/n): ")
    if response.lower() != 'y':
        return
    """
    mouse = Controller()
    root = tk.Tk() #get screen size
    root.withdraw() #hide the window
    screen_width = root.winfo_screenwidth() #get screen size
    screen_height = root.winfo_screenheight() #get screen size
    root.destroy()    
    # Main tracking loop
    while cap.isOpened():
        start_time = time.time() #TIME CALC
        ret, frame = cap.read()
        time_calculator(start_time, "Cam capture") #TIME END
        if not ret:
            break
        
        frame_count += 1
        fps_counter.update()
        start_time = time.time() #TIME CALC
        bboxes, _ = detector.detect(frame)
        time_calculator(start_time, "Face detect") #TIME END
        
        for bbox in bboxes:
            start_time = time.time() 
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                continue
            
            # Extract eye region for model
            eye_region = extract_eye_region_simple(face_img)
            normalized = normalize_for_mpiigaze(eye_region)
            # Get gaze estimation
            pitch, yaw = engine.estimate(face_img, use_eye_extraction=True)
            time_calculator(start_time, "Inference Time") #TIME END
            # CRITICAL: Get gaze estimation
            # pitch, yaw = engine.estimate(face_img)
            start_time = time.time() #TIME CALC
            # Visualization on video frame
            draw_bbox_gaze(frame, bbox, pitch, yaw)
            face_center_x, face_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            
            # OPTION 1: If calibrated, map directly to screen coordinates
            if coeffs is not None:
                # This gives SCREEN coordinates (0 to screen_width, 0 to screen_height)
                screen_x, screen_y = predict_with_calibration(pitch, yaw, coeffs)
                
                # Clamp to screen bounds
                screen_x = np.clip(screen_x, 0, screen_width - 1)
                screen_y = np.clip(screen_y, 0, screen_height - 1)
                gaze_x, gaze_y = project_to_2d(
                    pitch, yaw, frame.shape[1], frame.shape[0], 
                    face_center_x, face_center_y
                )
                # Move cursor to ABSOLUTE screen position
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                speed = 20  # pixels per frame
                dx = gaze_x - frame_center_x
                dy = gaze_y - frame_center_y
                norm = math.hypot(dx, dy)
                move_x = int(speed * dx / norm)
                move_y = int(speed * dy / norm)
                mouse.move(move_x, move_y)
                
                # For visualization on frame, project screen coords to frame coords
                frame_x = int(screen_x * frame.shape[1] / screen_width)
                frame_y = int(screen_y * frame.shape[0] / screen_height)
                cv2.circle(frame, (frame_x, frame_y), 10, (0, 0, 255), -1)
            
            # OPTION 2: If not calibrated, use relative movement with deadzone
            else:
                # Project gaze to frame coordinates
                gaze_x, gaze_y = project_to_2d(
                    pitch, yaw, frame.shape[1], frame.shape[0], 
                    face_center_x, face_center_y
                )
                
                cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
                
                # Calculate relative movement with deadzone
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                deadzone_threshold = 0.3 * min(frame.shape[1], frame.shape[0]) / 2
                
                dx = gaze_x - frame_center_x
                dy = gaze_y - frame_center_y
                distance = math.hypot(dx, dy)
                
                if distance > deadzone_threshold:
                    # Scale movement speed based on distance from center
                    speed = 20  # pixels per frame
                    move_x = int(speed * dx / distance)
                    move_y = int(speed * dy / distance)
                    
                    # RELATIVE movement
                    mouse.move(move_x, move_y)
            time_calculator(start_time, "Moving mouse") #TIME END
            start_time = time.time() #TIME CALC
            # Blink detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                
                left_closed = left_ear < EAR_CLOSE_THRESHOLD
                right_closed = right_ear < EAR_CLOSE_THRESHOLD
                left_open = left_ear > EAR_OPEN_THRESHOLD
                right_open = right_ear > EAR_OPEN_THRESHOLD
                
                if left_closed and right_closed:
                    both_blink_frames += 1
                    left_blink_frames = 0
                    right_blink_frames = 0
                elif left_closed and right_open:
                    left_blink_frames += 1
                    right_blink_frames = 0
                    both_blink_frames = 0
                elif right_closed and left_open:
                    right_blink_frames += 1
                    left_blink_frames = 0
                    both_blink_frames = 0
                else:
                    left_blink_frames = 0
                    right_blink_frames = 0
                    both_blink_frames = 0
                
                now = now_ms()
                if now - last_blink_ms >= ACTION_COOLDOWN_MS:
                    if both_blink_frames >= 2:
                        #mouse.click(Button.left, 1)
                        last_blink_ms = now
                        both_blink_frames = 0

                    elif right_blink_frames >= 2:
                        #mouse.click(Button.right, 1)
                        last_blink_ms = now
                        right_blink_frames = 0

                    elif left_blink_frames >= 2:
                        direction = right_eye_vertical_gaze(landmarks, RIGHT_EYE)
                        #if direction == "UP":
                            # Positive scroll value → scroll up
                            #mouse.scroll(0, SCROLL_AMOUNT)
                        #elif direction == "DOWN":
                            # Negative scroll value → scroll down
                            #mouse.scroll(0, -SCROLL_AMOUNT)
                        last_blink_ms = now
                        left_blink_frames = 0
            time_calculator(start_time, "Blinking ") #TIME END
        # Display FPS and info on frame
        fps = fps_counter.get_fps()  
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show calibration status
        if coeffs is not None:
            cv2.putText(frame, "Calibrated", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Calibration", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if writer:
            writer.write(frame)
        
        cv2.imshow("Gaze Tracking", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):  # Press 'c' to recalibrate
            print("\nRecalibrating...")
            coeffs = run_calibration(cap, engine, detector)
    
    # Cleanup
    print(f"\n\nShutting down...")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps_counter.get_fps():.1f}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# pip install -r reqs.txt*/
# python fullpipeline_FIXIT.py --source 0 --model best_model.onnx
