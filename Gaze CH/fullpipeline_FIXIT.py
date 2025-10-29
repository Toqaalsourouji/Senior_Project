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
from pynput.mouse import Controller, Button
import uniface
from typing import Tuple
import onnxruntime as ort
import time 
import tkinter as tk
from typing import Tuple, Optional

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
    
    # MediaPipe eye landmark indices change to harcascade eye
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




def gaze_to_screen_projection(pitch: float, yaw: float, screen_width:int, 
                            screen_hight: int, face_x: float, face_y: float,
                            frame_width: int, frame_height: int) -> Tuple[int, int]:
    """
        Convert the gaze angles to screen coordinates using a consistent formula
        args:
            pitch: Gaze pitch angle in radians (positive = looking down, negative = looking up)
            yaw: Gaze yaw angle in radians (positive = looking right, negative = looking left)
            screen_width: Width of the screen in pixels (in pixels)
            screen_height: Height of the screen in pixels (in pixels)
            face_x: X coordinate of the face center in the frame (face center positions in frame coordinates)
            face_y: Y coordinate of the face center in the frame (face center positions in frame coordinates)
            frame_width: Width of the video frame in pixels (camera frame dimensions)
            frame_height: Height of the video frame in pixels (camera frame dimensions)
        returns: (screen_x, screen_y): Predicted screen coordinates in pixels
    """
    # Sensitivity factors to map angles to screen movement
    # the gaze angles are small, so we need to scale them up to screen size
    #gaze angles from -42 to +42 degrees -> map to screen size

    center_x = screen_width / 2 #get screen center
    center_y = screen_hight / 2 #get screen center

    #convert angles to screen displacement
    #scale factor: how many pixels per radian 
    # screen_width / (max expected yaw range in radians)

    horizontal_scale = screen_width / np.radians(84) #assuming -42 to +42 degrees (this can be adjusted by testing)
    vertical_scale = screen_hight / np.radians(84)  #assuming -42 to +42 degrees (this can be adjusted by testing)

    #apply projection formula 
    screen_x = center_x + (yaw * horizontal_scale)
    screen_y = center_y + (pitch * vertical_scale)

    #optional for u hassan, add head position compensation here
    #If face ,oves right in the frame, the gaze point should move right on the screen and vice versa and so on \
    face_offset_x = (face_x / frame_width - 0.5) * screen_width * 0.2 #20% of screen width 
    face_offset_y = (face_y / frame_height - 0.5) * screen_hight * 0.2 #20% of screen height

    #clamp to screne bounds
    screen_x = np.clip(screen_x + face_offset_x, 0, screen_width - 1)
    screen_y = np.clip(screen_y + face_offset_y, 0, screen_hight - 1)

    return int(screen_x), int(screen_y)

def run_offset_calibration(cap, engine, detector , calibration_points):
    """
        This calibration measures the model's systematic error (offset)

        how it works:
        1. show calibration points on screen
        2. for each point, capture the model predection (using projection to screen coordinates)
        3. calculate the difference between the predictions and the actual points
        4. average the differences to get the offsets
        5. return the offsets to apply to all future predictions 
    """

    #get screen size
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    #get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from video source for calibration")
        return None
    frame_height, frame_width = test_frame.shape[:2]

    print("\n" + "="*80)
    print("OFFSET CALIBRATION")
    print("="*80)
    print(f"Screen: {screen_width}x{screen_height}")
    print(f"Camera: {frame_width}x{frame_height}")
    print(f"Points: {len(calibration_points)}")
    print("\nInstructions:")
    print("  1. Look DIRECTLY at each GREEN circle")
    print("  2. Keep your head STILL")
    print("  3. Stay at the SAME distance from screen")
    print("\nStarting in...")

    #counting down 
    for countdown in range (3, 0, -1):
        print(f"  {countdown}...")
        dummy_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, str(countdown), 
                   (screen_width//2 - 50, screen_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        cv2.imshow("Calibration", dummy_frame)
        cv2.waitKey(1000)
    
    #store errors for each calibration point
    errors_x = []
    errors_y = []
    successful_points = 0

    for idx, (norm_x, norm_y) in enumerate(calibration_points):
        #calculate target screen position
        target_x = int(norm_x * screen_width)
        target_y = int(norm_y * screen_height)

        print(f"\nPoint {idx+1}/{len(calibration_points)}: Look at ({target_x}, {target_y})")

        #show calibration point for 2 seconds
        for countdonw in range(2, 0, -1):
            calib_frame = np.zeros((screen_height, screen_width, 3), dtype = np.uint8)

            #draw the target
            cv2.circle(calib_frame, (target_x, target_y), 50, (0, 255, 0), -1)
            cv2.circle(calib_frame, (target_x, target_y), 55, (225, 255, 225), 3)

            #countdown text
            cv2.putText(calib_frame, f"Capturing in {countdonw}",(target_x -25, target_y +15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(calib_frame, f"Pount {idx+1}/{len(calibration_points)} - look at the circle",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", calib_frame)
            cv2.waitKey(1000)

        #Capture frame and get model prediction
        samples_pitch = []
        samples_yaw = []
        samples_face_x = [] 
        samples_face_y = []

        print("  Capturing samples...", end = " ")

        for sample_idx in range(6): #capture 6 samples for averaging
            ret, frame = cap.read()
            if not ret: 
                print("[ERROR] Could not read from video source during calibration")
                continue
            bboxes, _ = detector.detect (frame)
            if len(bboxes) == 0:
                print(".", end="", flush=True)
                continue
            x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
            face_img = frame[y_min:y_max, x_min:x_max]

            if face_img.size == 0:
                print(".", end="", flush=True)
                continue
            try: 
                #get gaze estimation 
                pitch, yaw = engine.estimate(face_img, use_eye_extraction=True)

                #store face postion 
                face_center_x = (x_min + x_max) / 2
                face_center_y = (y_min + y_max) / 2

                samples_pitch.append(pitch)
                samples_yaw.append(yaw)
                samples_face_x.append(face_center_x)
                samples_face_y.append(face_center_y)

                #visual feedback 
                calib_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                radius = 50 + sample_idx * 3
                cv2.circle(calib_frame, (target_x, target_y), radius, (0, 255, 0), -1)
                cv2.putText(calib_frame, f"Capturing samples... {sample_idx+1}/6", 
                        (target_x - 30, target_y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Calibration", calib_frame)
                cv2.waitKey(50)

            except Exception as e:
                print(f"\n Sample {sample_idx} failed: {e}")
                continue

            #process the samples
        if len(samples_pitch) < 3:
            print("Not enough valid samples captured, skipping point.")
            continue
            
        #remove outliers using the median
        # Remove outliers
        def remove_outliers(data):
            """Remove outliers using median absolute deviation."""
            if len(data) < 3:
                return np.array(data)
                
            data = np.array(data)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
                
            if mad < 1e-6:
                return data
                
            threshold = 2.0  # Z-score threshold
            z_scores = np.abs((data - median) / (mad + 1e-6))
            return data[z_scores < threshold]
            
        clean_pitch = remove_outliers(np.array(samples_pitch))
        clean_yaw = remove_outliers(np.array(samples_yaw))

        #average the clean samples 
        avg_pitch = np.median(clean_pitch)
        avg_yaw = np.median(clean_yaw)
        avg_face_x = np.median(samples_face_x)
        avg_face_y = np.median(samples_face_y)

        #project to screen using our fomrula 
        predicted_x, predicted_y = gaze_to_screen_projection(
            avg_pitch, avg_yaw, screen_width, screen_height,
            avg_face_x, avg_face_y, frame_width, frame_height
        )

        #calculate error (how bad our model did :) )
        error_x = target_x - predicted_x
        error_y = target_y - predicted_y
        error_dist = np.sqrt(error_x **2 + error_y **2) #euclidean distance 

        errors_x.append(error_x)
        errors_y.append(error_y)
        successful_points += 1
        print(f"Done. Prediction: ({predicted_x}, {predicted_y}), Error: ({error_x}, {error_y}), Distance: {error_dist:.1f}px")
        print(f" used {len(clean_pitch)}/{len(samples_pitch)} samples after outlier removal.")

        #show reuslts 
        result_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # draw target
        cv2.circle(result_frame, (target_x, target_y), 50, (0, 255, 0), -1)
        cv2.putText(result_frame, "Target", (target_x - 60, target_y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        #draw prediction
        cv2.circle(result_frame, (predicted_x, predicted_y), 50, (255, 0, 0), -1)
        cv2.putText(result_frame, "Prediction", (predicted_x - 80, predicted_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            

    cv2.destroyAllWindows()

    #calculate average offsets
    if successful_points < 3:
        print("[ERROR] Not enough successful calibration points, calibration failed.")
        return None
    offset_x = np.median(errors_x)
    offset_y = np.median(errors_y)

    mean_error = np.mean([np.sqrt(ex**2 + ey**2) for ex, ey in zip(errors_x, errors_y)])
    median_error = np.median([np.sqrt(ex**2 + ey**2) for ex, ey in zip(errors_x, errors_y)])
    max_error = np.max([np.sqrt(ex**2 + ey**2) for ex, ey in zip(errors_x, errors_y)])

    print("\n" + "="*80)
    print(f"CALIBRATION COMPLETE: {successful_points}/{len(calibration_points)} points")
    print("="*80)
    print(f"\nCalibration Offset: ({offset_x:.0f}, {offset_y:.0f}) pixels")
    print(f"\nAccuracy Metrics:")
    print(f"  Mean error:   {mean_error:.1f} pixels")
    print(f"  Median error: {median_error:.1f} pixels")
    print(f"  Max error:    {max_error:.1f} pixels")
    
    if mean_error < 150:
        print(f"  ✓ GOOD calibration!")
    elif mean_error < 300:
        print(f"  ⚠ Acceptable, but could be better")
    else:
        print(f"  ✗ Poor calibration - consider recalibrating")
    
    # Return calibration data
    return {
        'offset_x': offset_x,
        'offset_y': offset_y,
        'screen_width': screen_width,
        'screen_height': screen_height,
        'frame_width': frame_width,
        'frame_height': frame_height
    }

def predict_gaze_with_offset(pitch: float, yaw: float,
                            face_x: float, face_y: float,
                            calibration_data: dict) -> Tuple[int, int]:
    """
    Predict screen coordinates with offset correction.
    
    Args:
        pitch, yaw: Gaze angles from model
        face_x, face_y: Face position in frame
        calibration_data: Dict from run_offset_calibration()
    
    Returns:
        (screen_x, screen_y): Corrected screen coordinates
    """
    
    if calibration_data is None:
        # No calibration - use uncorrected projection
        # You'll need screen dimensions here
        return None, None
    
    # Step 1: Project using formula
    predicted_x, predicted_y = gaze_to_screen_projection(
        pitch, yaw,
        calibration_data['screen_width'],
        calibration_data['screen_height'],
        face_x, face_y,
        calibration_data['frame_width'],
        calibration_data['frame_height']
    )
    
    # Step 2: Apply offset correction
    corrected_x = predicted_x + calibration_data['offset_x']
    corrected_y = predicted_y + calibration_data['offset_y']
    
    # Step 3: Clamp to screen
    corrected_x = np.clip(corrected_x, 0, calibration_data['screen_width'] - 1)
    corrected_y = np.clip(corrected_y, 0, calibration_data['screen_height'] - 1)
    
    return int(corrected_x), int(corrected_y)

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


class KalmanFilter2D: 
    #kalman filter for smoothing the mouse movement
    def __init__(self, process_variance=1, measurement_variance=10): #we can play around with those values
        """
        Args:
            process_variance: How much we trust the model's predictions (lower = trust model more)
            measurement_variance: How much noise in gaze measurements (higher = more smoothing)
        """
        # State: [x, y, dx, dy] - position and velocity
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
    
        # Covariance matrix - uncertainty in our estimate
        self.covariance = np.eye(4) * 1000 #we start with high unceertainty first 

        # State transition matrix - how state evolves
        # Assumes constant velocity: new_x = x + dx, new_dx = dx

        self.F = np.array([
            [1, 0, 1, 1], #x = x + dx
            [0, 1, 0, 1], #y = y + dy
            [0, 0, 1 ,0], #dx = dx (const velocity)
            [0, 0, 0, 1] # dy = dy
        ])

        # Measurement matrix - we only measure position, not velocity
        self.H = np.array ([
            [1, 0 ,0 ,0], # measured_x = 1*x + 0*y + 0*dx + 0*dy
            [0, 1, 0, 0]  # measured_y = 0*x + 1*y + 0*dx + 0*dy
        ])

        # Process noise covariance
        self.Q = np.eye(4) * process_variance

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_variance

        self.initialized = False

    def update(self, measurement_x, measurement_y): 
        """
        Update filter with new gaze measurement.
        
        Args:
            measurement_x: Raw gaze x position
            measurement_y: Raw gaze y position
            
        Returns:
            (smoothed_x, smoothed_y): Filtered position
        """
        measurement = np.array([measurement_x,measurement_y])

        # Initialize with first measurement
        if not self.initialized:
            self.state[0] = measurement_x
            self.state[1] = measurement_y
            self.initialized = True
            return measurement_x, measurement_y
        
        # PREDICT STEP
        # Predict next state based on motion model
        predicted_state = self.F @ self.state
        predicted_covariance  = self.F @ self.covariance @ self.F.T + self.Q

        # UPDATE STEP
        # Calculate Kalman gain (how much to trust measurement vs prediction)
        S = self.H @ predicted_covariance @ self.H.T + self.R
        K = predicted_covariance @ self.H.T @ np.linalg.inv(S)

        # Update state with measurement
        innovation = measurement - (self.H @ predicted_state)
        self.state = predicted_state + K @ innovation
        self.covariance = (np.eye(4) - K @ self.H) @ predicted_covariance

        return self.state[0], self.state[1]
    
    def reset(self):
        """Reset filter state."""
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.covariance = np.eye(4) * 1000
        self.initialized = False 

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
    
    # ============================================================================
    # NEW: Run offset-based calibration
    # ============================================================================
    calibration_data = run_offset_calibration(cap, engine, detector, calibration_points)
    
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
    print("TRACKING STARTED - Press 'q' to quit, 'c' to recalibrate")
    print("="*80 + "\n")
    
    frame_count = 0
    mouse = Controller()
    
    # Get screen size from calibration data or manually
    if calibration_data is not None:
        screen_width = calibration_data['screen_width']
        screen_height = calibration_data['screen_height']
    else:
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    
    # Kalman filter initialization
    kalman = KalmanFilter2D(
        process_variance=1.0,      # HOW MUCH MODEL NOISE
        measurement_variance=25.0  # HOW MUCH MEASUREMENT NOISE
    )
    
    # Main tracking loop
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        time_calculator(start_time, "Cam capture")
        
        if not ret:
            break
        
        frame_count += 1
        fps_counter.update()
        
        start_time = time.time()
        bboxes, _ = detector.detect(frame)
        time_calculator(start_time, "Face detect")
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                continue
            
            # Get gaze estimation
            start_time = time.time()
            pitch, yaw = engine.estimate(face_img, use_eye_extraction=True)
            time_calculator(start_time, "Gaze estimation")
            
            # Visualization on video frame
            draw_bbox_gaze(frame, bbox, pitch, yaw)
            
            # Get face center position
            face_center_x = (x_min + x_max) / 2
            face_center_y = (y_min + y_max) / 2
            
            # ============================================================================
            # NEW: Use offset-based calibration
            # ============================================================================
            if calibration_data is not None:
                # Predict screen coordinates with offset correction
                start_time = time.time()
                raw_x, raw_y = predict_gaze_with_offset(
                    pitch, yaw,
                    face_center_x, face_center_y,
                    calibration_data
                )
                time_calculator(start_time, "Gaze to screen projection")
                
                if raw_x is not None and raw_y is not None:
                    # Clamp raw prediction to screen bounds
                    raw_x = np.clip(raw_x, 0, screen_width - 1)
                    raw_y = np.clip(raw_y, 0, screen_height - 1)
                    
                    # Apply Kalman filter for smoothing
                    start_time = time.time()
                    smooth_x, smooth_y = kalman.update(raw_x, raw_y)
                    time_calculator(start_time, "Kalman filter")
                    
                    # ======== ADVANCED SPEED CONTROL ========
                    # 1. Get current mouse position
                    current_x, current_y = mouse.position
                    
                    # 2. Calculate movement delta
                    delta_x = smooth_x - current_x
                    delta_y = smooth_y - current_y
                    
                    # 3. Apply speed multiplier
                    SPEED_MULTIPLIER = 0.15  # ← MAIN CONTROL: 0.05=slow, 0.3=medium, 1.0=fast
                    delta_x *= SPEED_MULTIPLIER
                    delta_y *= SPEED_MULTIPLIER
                    
                    # 4. Limit maximum speed per frame
                    MAX_SPEED = 10  # ← Maximum pixels per frame (5=very slow, 20=medium, 50=fast)
                    distance = math.sqrt(delta_x**2 + delta_y**2)
                    if distance > MAX_SPEED:
                        scale = MAX_SPEED / distance
                        delta_x *= scale
                        delta_y *= scale
                    
                    # 5. Add deadzone (optional - prevents tiny jitters)
                    DEADZONE = 2  # Don't move if change is less than 2 pixels
                    if abs(delta_x) >= DEADZONE or abs(delta_y) >= DEADZONE:
                        # 6. Calculate new position
                        new_x = int(current_x + delta_x)
                        new_y = int(current_y + delta_y)
                        
                        # 7. Clamp to screen bounds
                        new_x = np.clip(new_x, 0, screen_width - 1)
                        new_y = np.clip(new_y, 0, screen_height - 1)
                        
                        # 8. Move cursor
                        start_time = time.time()
                        mouse.position = (new_x, new_y)
                        time_calculator(start_time, "Mouse movement")
                    
                    # ========================================
                    
                    # Visualization (show where you're looking vs where cursor is)
                    frame_smooth_x = int(smooth_x * frame.shape[1] / screen_width)
                    frame_smooth_y = int(smooth_y * frame.shape[0] / screen_height)
                    cv2.circle(frame, (frame_smooth_x, frame_smooth_y), 8, (0, 255, 0), 2)  # Green = target
                    
                    frame_cursor_x = int(new_x * frame.shape[1] / screen_width)
                    frame_cursor_y = int(new_y * frame.shape[0] / screen_height)
                    cv2.circle(frame, (frame_cursor_x, frame_cursor_y), 6, (255, 0, 0), -1)  # Blue = cursor
            
            # ============================================================================
            # FALLBACK: If not calibrated, use relative movement
            # ============================================================================
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
                    speed = 0.05  # pixels per frame
                    move_x = int(speed * dx / distance)
                    move_y = int(speed * dy / distance)
                    
                    # RELATIVE movement
                    mouse.move(move_x, move_y)
            
            # ============================================================================
            # Blink detection
            # ============================================================================
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            results = face_mesh.process(image_rgb)
            time_calculator(start_time, "Face mesh processing")
            
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
                        mouse.click(Button.left, 1)
                        last_blink_ms = now
                        both_blink_frames = 0
                        print("✓ Left click")
                    
                    elif right_blink_frames >= 2:
                        mouse.click(Button.right, 1)
                        last_blink_ms = now
                        right_blink_frames = 0
                        print("✓ Right click")
                    
                    elif left_blink_frames >= 2:
                        direction = right_eye_vertical_gaze(landmarks, RIGHT_EYE)
                        if direction == "UP":
                            mouse.scroll(0, SCROLL_AMOUNT)
                            print(f"✓ Scroll up")
                        elif direction == "DOWN":
                            mouse.scroll(0, -SCROLL_AMOUNT)
                            print(f"✓ Scroll down")
                        last_blink_ms = now
                        left_blink_frames = 0
            
            time_calculator(start_time, "Blink detection")
        
        # Display FPS and info on frame
        fps = fps_counter.get_fps()  
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show calibration status
        if calibration_data is not None:
            cv2.putText(frame, "Calibrated", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Show calibration offset
            offset_text = f"Offset: ({calibration_data['offset_x']:.0f}, {calibration_data['offset_y']:.0f})px"
            cv2.putText(frame, offset_text, (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No Calibration", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if writer:
            writer.write(frame)
        
        cv2.imshow("Gaze Tracking", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\n✓ Quitting...")
            break
        elif key == ord('c'):  # Press 'c' to recalibrate
            print("\n🔄 Recalibrating...")
            start_time = time.time()
            calibration_data = run_offset_calibration(cap, engine, detector, calibration_points)
            kalman.reset()  # Reset Kalman filter after recalibration
            time_calculator(start_time, "Recalibration")
            print("✓ Kalman filter reset")
            
            if calibration_data is not None:
                screen_width = calibration_data['screen_width']
                screen_height = calibration_data['screen_height']
    
    # Cleanup
    print(f"\n\n{'='*80}")
    print("SHUTTING DOWN")
    print('='*80)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps_counter.get_fps():.1f}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
# pip install -r reqs.txt
# python fullp_kf.py --source 0 --model best_model.onnx
