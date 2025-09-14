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


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

Frame = cv2.VideoCapture(0)

blink_counter = 0
BLINK_CONSEC_FRAMES = 1

LEFT_EYE = [362, 385, 387, 263, 373, 380] # Right eye indices from MediaPipe using dlib library
RIGHT_EYE = [33, 160, 158, 133, 153, 144] # Left eye indices from MediaPipe using dlib library

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
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.input_mean, std=self.input_std)
        ])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image).unsqueeze(0).numpy()  # to numpy for ONNX
        return tensor

    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch, yaw = outputs  # adjust if output format differs
        print("[DEBUG] ONNX output shapes:", [o.shape for o in outputs])


        pitch = torch.softmax(torch.tensor(pitch), dim=1).numpy()
        yaw = torch.softmax(torch.tensor(yaw), dim=1).numpy()

        pitch = np.sum(pitch * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw * self.idx_tensor) * self._binwidth - self._angle_offset

        return np.radians(pitch), np.radians(yaw)


def run_calibration(cap, engine, detector):
    global calibration_gaze, calibration_screen
    swidth, sheight = pyautogui.size()

    for norm_x, norm_y in calibration_points:
        target_x, target_y = int(norm_x * swidth), int(norm_y * sheight)

        #show target point
        calib_frame = np.zeros((500, 500, 3), dtype=np.uint8) #black screen
        calib_frame = np.zeros((sheight, swidth, 3), dtype=np.uint8)
        cv2.circle(calib_frame, (target_x, target_y), 20, (0, 255, 0), -1)
        cv2.imshow("Calibration", calib_frame)
        cv2.imshow("Calibration", calib_frame)
        cv2.waitKey(2000) #wait 2 seconds

        #capture frame and detect face abd get the predection
        ret, frame = cap.read()
        if not ret: continue

        bboxes, _ = detector.detect(frame)
        if len(bboxes) == 0: continue #no face detected, skip

        x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
        face_img = frame[y_min:y_max, x_min:x_max]
        pitch, yaw = engine.estimate(face_img)
        if face_img.size == 0: continue #invalid face image, skip
        calibration_gaze.append((pitch, yaw))
        calibration_screen.append((target_x, target_y))

    cv2.destroyAllWindows()
    print("Calibration complete.")

    #fit a polynomial regression model to map gaze to screen points
    calibration_gaze = np.array(calibration_gaze)
    calibration_screen = np.array(calibration_screen)
    X = np.hstack([calibration_gaze, np.ones((len(calibration_gaze), 1))]) #add bias term
    Y = calibration_screen
    coeffs,_,_,_ = np.linalg.lstsq(X, Y, rcond=None) #linear regression
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
    #print(f"[DEBUG] Using model: {args.model}")
    #print(f"[DEBUG] Using source: {args.source}")
    
    # Initialize components - ONLY pass model_path here
    engine = GazeEstimationTorch(args.model)  # Fixed this line

    detector = uniface.RetinaFace()
    
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return
    coeffs = run_calibration(cap, engine, detector)

    # Rest of your main function remains the same...
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 
                               cap.get(cv2.CAP_PROP_FPS) or 30, (width, height))

    # Rest of your gaze tracking and blink detection code...
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, _ = detector.detect(frame)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                continue
                
            pitch, yaw = engine.estimate(face_img)
            
            # Visualization and mouse control code...
            draw_bbox_gaze(frame, bbox, pitch, yaw)
            face_center_x, face_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            if coeffs is not None:
                gaze_x, gaze_y = predict_with_calibration(pitch, yaw, coeffs)
            else:
                gaze_x, gaze_y = project_to_2d(pitch, yaw, frame.shape[1], frame.shape[0], 
                                         face_center_x, face_center_y)
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

            # Mouse control implementation...
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            deadzone_threshold = 0.5 * min(frame_width, frame_height) / 2
            speed = 10
            dx = gaze_x - frame_center_x
            dy = gaze_y - frame_center_y
            clear_console()
            if abs(dx) > deadzone_threshold or abs(dy) > deadzone_threshold:
                dir_x = 1 if dx > 0 else -1 if dx < 0 else 0
                dir_y = 1 if dy > 0 else -1 if dy < 0 else 0
                
                if abs(dx) > abs(dy):
                    if dx > 0:
                        print("Looking RIGHT")
                    else:
                        print("Looking LEFT")
                else:
                    if dy > 0:
                        print("Looking DOWN")
                    else:
                        print("Looking UP")

                if abs(dx) > abs(dy):
                    pyautogui.moveRel(-dir_x * speed, 0, duration=0.05)
                else:
                    pyautogui.moveRel(0, dir_y * speed, duration=0.05)
            else:
                print("DEADZONE")
                #cv2.putText(frame, "DEADZONE", (frame.shape[1] - 200, 60), 
                 #         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # BLINKING START
            is_blinking = False
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
                    is_blinking = False
    
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    is_blinking = True
                    print("BLINKING")
                    pyautogui.leftClick()

        if writer:
            writer.write(frame)
            
        #cv2.imshow("Gaze Tracking", frame)
        #if cv2.waitKey(1) == ord('q'):
        #    break

    cap.release()
    if writer:
        writer.release()
    #cv2.destroyAllWindows()


mp_face_mesh = mp.solutions.face_mesh
def clear_console():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
if __name__ == "__main__":
    main()
# pip install -r reqs.txt*/
# python test2.py --source 0 --model mobileone_s0_gaze.onnx
