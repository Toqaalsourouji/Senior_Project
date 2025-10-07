import cv2
import logging
import argparse
import warnings
import numpy as np
import math
import mediapipe as mp
import uniface
from typing import Tuple
import onnxruntime as ort
import time
from collections import deque
from pynput.mouse import Controller, Button
mouse = Controller()


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

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
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    return parser.parse_args()


def softmax_np(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
class GazeEstimationTorch:
    def __init__(self, model_path: str):
        # Model config
        self.input_size = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.input_mean, dtype=np.float32)) / np.array(self.input_std, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(image, axis=0).astype(np.float32)


    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch, yaw = outputs  # adjust if output format differs


        pitch = softmax_np(pitch)
        yaw   = softmax_np(yaw) 

        pitch = np.sum(pitch * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw * self.idx_tensor) * self._binwidth - self._angle_offset

        return np.radians(pitch), np.radians(yaw)

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

def time_calculator(start_inf, label=""):
    end_inf = time.time()
    calc_time = end_inf - start_inf
    print(f"{label} time: {calc_time*1000:.5f} ms")

def main():
    args = parse_args()
    
    # Initialize components - ONLY pass model_path here
    engine = GazeEstimationTorch(args.model)  # Fixed this line
    detector = uniface.RetinaFace()
    # detector.detect(cv2.resize(frame, (640, 360)))
    # Rest of your main function remains the same...
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    blink_counter = 0


    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 
                               cap.get(cv2.CAP_PROP_FPS) or 30, (width, height))

    FRAME_SKIP = 15  # detect face once every N frames
    tracker = None
    frame_count = 0
    last_bbox = None
    # Rest of your gaze tracking and blink detection code...
    while cap.isOpened():
        prev_time = time.time() #fps boi
        start_time = time.time() #TIME CALC
        ret, frame = cap.read()
        time_calculator(start_time, "Cam capture") #TIME END
        if not ret:
            break
        frame_count += 1

        start_time = time.time() #TIME CALC
        if tracker is None or frame_count % FRAME_SKIP == 0:
           
            # Detect face once every N frames
            bboxes, _ = detector.detect(frame)
            if len(bboxes) > 0:
                last_bbox = bboxes[0]

                # Use lightweight MOSSE tracker instead of CSRT
                tracker = cv2.legacy.TrackerMOSSE_create()


                x_min, y_min, x_max, y_max = map(int, last_bbox[:4])
                tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))
            else:
                tracker = None
        else:
            # Update tracker for skipped frames
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                last_bbox = (x, y, x + w, y + h)
            else:
                tracker = None  # force re-detection next frame

        time_calculator(start_time, "detector")

        if last_bbox is None:
            cv2.imshow("Gaze Tracking", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue


        start_time = time.time() #TIME CALC
        x_min, y_min, x_max, y_max = map(int, last_bbox[:4])
        face_img = frame[y_min:y_max, x_min:x_max]
        
        if face_img.size == 0:
            continue
        
         # --- Gaze Estimation ---
        start_time = time.time()
        pitch, yaw = engine.estimate(face_img)
        time_calculator(start_time, "Gaze Inference")

        # --- Draw results ---
        start_time = time.time()
        draw_bbox_gaze(frame, last_bbox, pitch, yaw)
        face_center_x, face_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        gaze_x, gaze_y = project_to_2d(
            pitch, yaw, frame.shape[1], frame.shape[0],
            face_center_x, face_center_y
        )
        cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
        time_calculator(start_time, "bbox draw")

        # --- Mouse control ---
        start_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        deadzone_threshold = 0.5 * min(frame_width, frame_height) / 2
        speed = 15
        dx = gaze_x - frame_center_x
        dy = gaze_y - frame_center_y

        dx = -dx 
        if abs(dx) > deadzone_threshold or abs(dy) > deadzone_threshold:
            dir_x = 1 if dx > 0 else -1 if dx < 0 else 0
            dir_y = 1 if dy > 0 else -1 if dy < 0 else 0
            new_x, new_y = mouse.position
            if abs(dx) > abs(dy):
                new_x += dir_x * speed
            else:
                new_y += dir_y * speed
            mouse.position = (new_x, new_y)
        else:
            cv2.putText(frame, "DEADZONE", (frame.shape[1] - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        time_calculator(start_time, "mouse move end")

        # --- Blink detection ---
        start_time = time.time()
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
            else:
                blink_counter = 0

            if blink_counter >= BLINK_CONSEC_FRAMES:
                print("BLINKING")
               # mouse.click(Button.left, 1)
        time_calculator(start_time, "Blinking")

        # --- Display frame ---
        if writer:
            writer.write(frame)


        cv2.imshow("Gaze Tracking", frame)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

BLINK_CONSEC_FRAMES = 1
mp_face_mesh = mp.solutions.face_mesh

if __name__ == "__main__":
    main()


# main()
# pip install -r gaze-estimation-main/requirements.txt*/
# python frame_skipping.py --source 0 --model mobileone_s0_gaze.onnx or model name
# FPS 14 AND 7, AVG IS 13.5 || WOW 32FPS NEW and 10 FPS
#Cam capture: 5.98
#Detector: 70
#Gaze inference: 27.67 | 17N
#BBox draw: 1.00
#Mouse ctrl calc: 0.00
#Mouse move: 0.00
#Blinking: 6.98


#WITH DETECTOR GET
#Cam capture time: 7.97892 ms
#detector time: 72.80517 ms
#Gaze Inference time: 16.95490 ms
#bbox draw time: 0.00000 ms
#mouse move end time: 0.00000 ms
#Blinking time: 5.98216 ms
#FPS: 9.46

#WITHOUT DETECTOR GET
#Cam capture time: 7.97749 ms
#detector time: 0.99826 ms
#Gaze Inference time: 16.97326 ms
#bbox draw time: 0.00000 ms
#mouse move end time: 0.00000 ms
#Blinking time: 6.98137 ms
#FPS: 27.84
