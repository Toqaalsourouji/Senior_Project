#!/usr/bin/env python3
import argparse
import json
import os
import math
from enum import Enum
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
import onnxruntime
from pynput.mouse import Controller as MouseController, Button
from demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter
import mediapipe as mp 


MOUTH_OPEN_THRESHOLD = 18.0     # tune by testing
mouse_frozen = False
mouth_was_open = False  # NEW


def mouth_open_amount(landmark):
    """Compute vertical distance between upper/lower inner lip."""
    upper = landmark[13]
    lower = landmark[14]
    return np.linalg.norm(lower - upper)


#media pipe init
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#get landmarks using mediapipe
def get_mediapipe_landmarks(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    mesh = result.multi_face_landmarks[0].landmark
    landmark = np.array([(p.x * w, p.y * h) for p in mesh], dtype=np.float32)
    return landmark


# Import the new calibration system components
try:
    from gaze_calibrator import GazeCalibrationSystem
    from main_app_example import GazeDirectionInterpreter
    NEW_CALIBRATION_AVAILABLE = True
except ImportError:
    print("Warning: New calibration system not available. Using original calibration only.")
    NEW_CALIBRATION_AVAILABLE = False


class ControlMode(Enum):
    """Control modes for mouse operation."""
    ANALOG = "analog"      # Original continuous mouse movement


class IntegratedGazeCalibrator:
    """
    Integrated calibrator that manages both calibration systems:
    - Original: Single center-point calibration
    - Directional: 9-point calibration for discrete directions
    """
    
    def __init__(self, center_calibration_file="gaze_calibration.json",
                 directional_calibration_file="gaze_direction_calibration.json"):
        """
        Initialize the integrated calibrator.
        
        Args:
            center_calibration_file: File for center-point calibration
            directional_calibration_file: File for 9-direction calibration
        """
        # Original center calibration
        self.center_calibration_file = center_calibration_file
        self.center_reference_pitchyaw = None
        self.center_calibration_data = []
        self.is_center_calibrating = False
        self.center_calibration_start_time = None
        self.center_sample_duration = 2.0
        
        # Load center calibration if exists
        self.load_center_calibration()
        
        # Directional calibration
        self.directional_calibration_file = directional_calibration_file
        self.directional_interpreter = None
        if NEW_CALIBRATION_AVAILABLE:
            self.directional_interpreter = GazeDirectionInterpreter(directional_calibration_file)
        
        # Current calibration mode
        self.current_mode = "center"  # "center" or "directional"
    
    def load_center_calibration(self):
        """Load center calibration from file."""
        if os.path.exists(self.center_calibration_file):
            try:
                with open(self.center_calibration_file, 'r') as f:
                    calib_data = json.load(f)
                    self.center_reference_pitchyaw = (calib_data['pitch'], calib_data['yaw'])
                    print(f"\nLoaded center calibration: pitch={self.center_reference_pitchyaw[0]:.4f}, "
                          f"yaw={self.center_reference_pitchyaw[1]:.4f}")
            except Exception as e:
                print(f"Failed to load center calibration: {e}")
    
    def save_center_calibration(self):
        """Save center calibration to file."""
        if self.center_reference_pitchyaw is not None:
            calib_data = {
                'pitch': float(self.center_reference_pitchyaw[0]),
                'yaw': float(self.center_reference_pitchyaw[1])
            }
            try:
                with open(self.center_calibration_file, 'w') as f:
                    json.dump(calib_data, f)
                print(f"Center calibration saved to {self.center_calibration_file}")
            except Exception as e:
                print(f"Failed to save center calibration: {e}")
    
    def start_center_calibration(self):
        """Start center-point calibration."""
        self.is_center_calibrating = True
        self.center_calibration_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        self.center_calibration_data = []
        print("\n" + "="*60)
        print("CENTER CALIBRATION STARTED")
        print("="*60)
        print("Look at the center of the screen...")
        print("="*60 + "\n")
    
    def add_center_sample(self, pitchyaw):
        """Add a sample for center calibration."""
        if not self.is_center_calibrating:
            return False
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - self.center_calibration_start_time
        
        if elapsed < self.center_sample_duration:
            self.center_calibration_data.append(pitchyaw)
            return False
        else:
            self.is_center_calibrating = False
            if len(self.center_calibration_data) > 0:
                samples = np.array(self.center_calibration_data)
                self.center_reference_pitchyaw = tuple(np.median(samples, axis=0))
                print("\n" + "="*60)
                print("CENTER CALIBRATION COMPLETE!")
                print(f"Reference: {self.center_reference_pitchyaw}")
                print("="*60 + "\n")
                self.save_center_calibration()
                return True
            return False
    
    def is_calibrated(self, mode="center"):
        """Check if calibration is complete for specified mode."""
        if mode == "center":
            return self.center_reference_pitchyaw is not None
        elif mode == "directional":
            return self.directional_interpreter is not None and self.directional_interpreter.is_loaded
        return False
    
    def get_center_reference(self):
        """Get the center calibration reference."""
        return self.center_reference_pitchyaw
    
    def interpret_direction(self, pitch, yaw):
        """Interpret gaze direction using directional calibration."""
        if self.directional_interpreter and self.directional_interpreter.is_loaded:
            return self.directional_interpreter.interpret_gaze(pitch, yaw)
        return None
    
    def get_direction_confidence(self, pitch, yaw):
        """Get confidence scores for all directions."""
        if self.directional_interpreter and self.directional_interpreter.is_loaded:
            return self.directional_interpreter.get_all_directions_confidence(pitch, yaw)
        return {}



class AnalogGazeMouseController:
    """Original analog mouse controller with continuous movement."""
    
    def __init__(self, dead_zone_deg=5.0, max_angle_deg=25.0,
                 min_speed=0.0, max_speed=15.0, smooth_factor=0.3):
        self.dead_zone_deg = dead_zone_deg
        self.max_angle_deg = max_angle_deg
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.smooth = smooth_factor
        self.mouse = MouseController()
        self.current_velocity = np.zeros(2, dtype=np.float32)
        self.enabled = True

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def _gaze_to_velocity(self, pitch_deg, yaw_deg):
        x = yaw_deg
        y = -pitch_deg
        v = np.array([x, y], dtype=np.float32)
        mag = np.linalg.norm(v)

        if mag < self.dead_zone_deg:
            return np.zeros(2, dtype=np.float32)

        direction = v / (mag + 1e-6)
        clipped = min(mag, self.max_angle_deg)
        t = (clipped - self.dead_zone_deg) / (self.max_angle_deg - self.dead_zone_deg)
        t = np.clip(t, 0.0, 1.0)
        t = t**1.2
        speed = self.min_speed + t * (self.max_speed - self.min_speed)
        return direction * speed

    def update(self, pitch_deg, yaw_deg):
        if not self.enabled:
            return

        target_v = self._gaze_to_velocity(pitch_deg, yaw_deg)
        self.current_velocity = (
            self.smooth * self.current_velocity +
            (1.0 - self.smooth) * target_v
        )

        if np.linalg.norm(self.current_velocity) < 0.1:
            self.current_velocity[:] = 0.0
            return

        try:
            x, y = self.mouse.position
            dx, dy = self.current_velocity
            self.mouse.position = (x + dx, y + dy)
        except Exception:
            pass


# Keep all the original gaze detection functions
face_model = np.float32([
    [-63.833572,  63.223045,  41.1674  ],
    [-12.44103 ,  66.60398 ,  64.561584],
    [ 12.44103 ,  66.60398 ,  64.561584],
    [ 63.833572,  63.223045,  41.1674  ],
    [-49.670784,  51.29701 ,  37.291245],
    [-16.738844,  50.439426,  41.27281 ],
    [ 16.738844,  50.439426,  41.27281 ],
    [ 49.670784,  51.29701 ,  37.291245],
    [-18.755981,  13.184412,  57.659172],
    [ 18.755981,  13.184412,  57.659172],
    [-25.941687, -19.458733,  47.212223],
    [ 25.941687, -19.458733,  47.212223],
    [  0.      , -29.143637,  57.023403],
    [  0.      , -69.34913 ,  38.065376]
])

cam_w, cam_h = 640, 480
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / np.tan(60 / 2 * np.pi / 180)
f_y = f_x
camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

def yolox_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r



def extract_critical_landmarks(landmark):
    # Use only first 468 canonical FaceMesh points
    base_landmark = landmark[:468]

    # Critical facial landmarks for headpose
    # These indices match the 3D face_model you already have
    TRACKED_POINTS = [
        105,  # right eyebrow outer
        66,   # right eyebrow inner
        296,  # left eyebrow inner
        334,  # left eyebrow outer
        33,   # right eye outer
        133,  # right eye inner
        362,  # left eye inner
        263,  # left eye outer
        168,  # nose ridge top
        197,  # nose ridge middle
        50,   # right cheek
        280,  # left cheek
        1,    # nose tip
        152   # chin
    ]

    pts = base_landmark[TRACKED_POINTS]  # shape should be (8, 2 or 3)

    # Keep only X,Y (ignore Z)
    pts = pts[:, :2].astype(np.float32)

    return pts



def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec

def vec_to_euler(x,y,z):
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return theta, phi

def rtvec_to_euler(rvec, tvec, unit="radian"):
    rvec_matrix = cv2.Rodrigues(rvec)[0]
    proj_matrix = np.hstack((rvec_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    if unit == "degree":
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
    return pitch, yaw, roll

def estimateHeadPose(landmarks, iterate=False):
    landmarks = extract_critical_landmarks(landmarks)
    # print("landmarks SHAPE:", landmarks.shape)
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera_matrix, camera_distortion)
    ## further optimize
    # if iterate:
    #     ret, rvec, tvec = cv2.solvePnP(facePts, landmarks, camera_matrix, camera_distortion, rvec, tvec, True)
    return rvec, tvec

def normalizeDataForInference(img, hr, ht):
    focal_norm = 960
    distance_norm_eye = 700
    distance_norm_face = 1200
    roiSize_eye = (60, 60)
    roiSize_face = (120, 120)
    img_u = img

    ht = ht.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face_model.T) + ht

    re = 0.5*(Fc[:,4] + Fc[:,5]).T
    le = 0.5*(Fc[:,6] + Fc[:,7]).T
    fe = (1./6.)*(Fc[:,4] + Fc[:,5] + Fc[:,6] + Fc[:,7] + Fc[:,10] + Fc[:,11]).T

    data = []
    for distance_norm, roiSize, et in zip([distance_norm_eye, distance_norm_eye, distance_norm_face], 
                                         [roiSize_eye, roiSize_eye, roiSize_face], [re, le, fe]):
        distance = np.linalg.norm(et)
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
        
        img_warped = cv2.warpPerspective(img_u, W, roiSize)
        data.append(img_warped)

        if distance_norm == distance_norm_face:
             data.append(R)
    return data



def detect_face(img, session, timer, score_thr=0.5, input_shape=(160, 128)) -> np.ndarray:
    img, ratio = yolox_preprocess(img, input_shape)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    timer.start_record("face_detection")
    output = session.run(None, ort_inputs)
    timer.end_record("face_detection")
    timer.start_record("face_detection_postprocess")
    predictions = demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr)
    timer.end_record("face_detection_postprocess")
    if dets is not None:
        final_boxes, final_scores = dets[:, :4], dets[:, 4]
        return np.array([[*final_box, final_score] for final_box, final_score in zip(final_boxes, final_scores)])
    else:
        return None




def estimate_gaze(img, landmark, session, timer) -> np.ndarray:
    timer.start_record("gaze_estimation_preprocess")
    rvec, tvec = estimateHeadPose(landmark)
    data = normalizeDataForInference(img, rvec, tvec)
    timer.end_record("gaze_estimation_preprocess")
    leye_image, reye_image, face_image, R = data

    leye_image = np.ascontiguousarray(leye_image, dtype=np.float32) / 255.0
    reye_image = np.ascontiguousarray(reye_image, dtype=np.float32) / 255.0
    face_image = np.ascontiguousarray(face_image, dtype=np.float32) / 255.0
    leye_image = np.transpose(np.expand_dims(leye_image, 0), (0,3,1,2))
    reye_image = np.transpose(np.expand_dims(reye_image, 0), (0,3,1,2))
    face_image = np.transpose(np.expand_dims(face_image, 0), (0,3,1,2))

    ort_inputs = {session.get_inputs()[0].name: leye_image,
                  session.get_inputs()[1].name: reye_image,
                  session.get_inputs()[2].name: face_image}
    timer.start_record("gaze_estimation")
    pred_pitchyaw_aligned = session.run(None, ort_inputs)[0][0]
    timer.end_record("gaze_estimation")
    pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
    pred_vec_aligned = euler_to_vec(*pred_pitchyaw_aligned)
    pred_vec_cam = np.dot(np.linalg.inv(R), pred_vec_aligned)
    pred_vec_cam /= np.linalg.norm(pred_vec_cam)
    pred_pitchyaw_cam = np.array(vec_to_euler(*pred_vec_cam))
    return pred_pitchyaw_cam, rvec, tvec

def draw_integrated_ui(image, calibrator, control_mode):
    """Draw UI for the integrated system."""
    h, w = image.shape[:2]
    
    # Mode indicator
    mode_text = f"Mode: {control_mode.value.upper()}"
    cv2.putText(image, mode_text, (w - 200, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Calibration status
    if calibrator.is_center_calibrating:
        center_x, center_y = w // 2, h // 2
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - calibrator.center_calibration_start_time
        progress = min(elapsed / calibrator.center_sample_duration, 1.0)
        
        cv2.circle(image, (center_x, center_y), 30, (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
        
        bar_width = 300
        bar_height = 30
        bar_x = w // 2 - bar_width // 2
        bar_y = h - 80
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (200, 200, 200), -1)
        fill_width = int(bar_width * progress)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (0, 0, 0), 2)
    
    return image


def visualize(img, face=None, landmark=None, gaze_pitchyaw=None, headpose=None, 
              calibrator=None, direction=None):
    """Enhanced visualization with direction indicator."""
    if face is not None:
        bbox = face[:4].astype(int)
        score = face[4]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0,255,0), 2)
        text = f'conf: {score * 100:.1f}%'
        txt_color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, text, (bbox[0], bbox[1]-5), font, 0.5, txt_color, thickness=1)
    
    if landmark is not None:
        EYE_BROW_EYE_POINTS = [
            33, 133, 362, 263,     # Eye corners
            159, 145, 386, 374,    # Iris/eyelid region (optional)
            65, 66, 107, 105,      # Right eyebrow
            295, 296, 334, 336     # Left eyebrow
        ]
        for idx in EYE_BROW_EYE_POINTS:
            x, y = landmark[idx].astype(int)
            cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=-1)
    
    if gaze_pitchyaw is not None and landmark is not None:
        eye_pos = landmark[-2:].mean(0)
        ref_pitchyaw = None
        if calibrator and calibrator.is_calibrated("center"):
            ref_pitchyaw = calibrator.get_center_reference()
        draw_gaze(img, eye_pos, gaze_pitchyaw, 300, 4, reference_pitchyaw=ref_pitchyaw)
    
    
    if headpose is not None:
        rvec = headpose[0]
        tvec = headpose[1]
        axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50], [0, 0, 0]])
        
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
        modelpts, _ = cv2.projectPoints(face_model, rvec, tvec, camera_matrix, camera_distortion)
        imgpts = imgpts.astype(int)
        modelpts = modelpts.astype(int)
        delta = modelpts[-1].ravel() - imgpts[-1].ravel()
        imgpts += delta
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    
    return img

def make_parser():
    parser = argparse.ArgumentParser("Integrated Gaze Control System")
    parser.add_argument("--source", default="/dev/video0", type=str)
    parser.add_argument("--save-video", default=None, type=str, required=False)
    parser.add_argument("--center-calibration", default="gaze_calibration.json", type=str)
    parser.add_argument("--directional-calibration", default="gaze_direction_calibration.json", type=str)
    parser.add_argument("--control-mode", default="analog", 
                        choices=["analog", "directional", "hybrid"],
                        help="Initial control mode")
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # Initialize ONNX sessions
    face_detection_session = onnxruntime.InferenceSession("./models/face_detection.onnx")
    gaze_estimation_session = onnxruntime.InferenceSession("./models/gaze_estimation.onnx")
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    timer = Timer()
    
    # Initialize smoothers
    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=478, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    
    # Initialize integrated calibrator
    calibrator = IntegratedGazeCalibrator(
        center_calibration_file=args.center_calibration,
        directional_calibration_file=args.directional_calibration
    )
    
    # Initialize mouse controllers
    analog_mouse = AnalogGazeMouseController(
        dead_zone_deg=5.0,
        max_angle_deg=25.0,
        min_speed=0.0,
        max_speed=15.0,
        smooth_factor=0.35
    )
    

    # Set initial control mode
    control_mode = ControlMode[args.control_mode.upper()]
    
    # Video writer if needed
    writer = None
    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
    
    cnt = 0
    mouse_enabled = True
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        timer.start_record("whole_pipeline")
        show_frame = frame.copy()
        CURRENT_TIMESTAMP = timer.get_current_timestamp()
        
        cnt += 1
        if cnt % 2 == 1:
            faces = detect_face(frame, face_detection_session, timer)
        
        direction = None
        direction_confidence = 0.0
        
        if faces is not None:
            face = faces[0]
            x1, y1, x2, y2 = face[:4]
            [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
            face = np.array([x1,y1,x2,y2,face[-1]])
            
            landmark = get_mediapipe_landmarks(frame)
            if landmark is None:
                continue 
            landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
            
            mouth_dist = mouth_open_amount(landmark)
            # --- Toggle Freeze on Mouth-Open Event ---
            mouth_is_open = mouth_dist > MOUTH_OPEN_THRESHOLD

            # If mouth just transitioned from CLOSED → OPEN
            if mouth_is_open and not mouth_was_open:
                # Toggle freeze state
                mouse_frozen = not mouse_frozen
                analog_mouse.enabled = not mouse_frozen
                if mouse_frozen:
                    print("Mouse FROZEN (mouth-open toggle)")
                else:
                    print("Mouse UNFROZEN (mouth-open toggle)")

            # Update state for next frame
            mouth_was_open = mouth_is_open

            
            gaze_pitchyaw, rvec, tvec = estimate_gaze(frame, landmark, gaze_estimation_session, timer)
            gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
            
            # Handle center calibration
            if calibrator.is_center_calibrating:
                calibrator.add_center_sample(gaze_pitchyaw)
            
            # Convert to degrees
            pitch_deg = np.degrees(gaze_pitchyaw[0])
            yaw_deg = np.degrees(gaze_pitchyaw[1])
            
            # Apply center calibration offset
            if calibrator.is_calibrated("center"):
                ref = calibrator.get_center_reference()
                calibrated_pitch_deg = np.degrees(gaze_pitchyaw[0] - ref[0])
                calibrated_yaw_deg = np.degrees(gaze_pitchyaw[1] - ref[1])
            else:
                calibrated_pitch_deg = pitch_deg
                calibrated_yaw_deg = yaw_deg
            
            # Get direction if directional calibration exists
            if calibrator.is_calibrated("directional"):
                direction = calibrator.interpret_direction(gaze_pitchyaw[0], gaze_pitchyaw[1])
                confidences = calibrator.get_direction_confidence(gaze_pitchyaw[0], gaze_pitchyaw[1])
                if direction and direction in confidences:
                    direction_confidence = confidences[direction]
            
            # Apply mouse control based on mode
            if mouse_enabled:
                if control_mode == ControlMode.ANALOG:
                    analog_mouse.update(calibrated_pitch_deg, calibrated_yaw_deg)

            
            # Visualize
            timer.start_record("visualize")
            show_frame = visualize(show_frame, face, landmark, gaze_pitchyaw, 
                                 [rvec, tvec], calibrator, direction)
            timer.end_record("visualize")
        
        # Draw UI
        show_frame = draw_integrated_ui(show_frame, calibrator, control_mode)
        
        timer.end_record("whole_pipeline")
        show_frame = timer.print_on_image(show_frame)
        
        if writer:
            writer.write(show_frame)
        
        cv2.imshow("Integrated Gaze Control", show_frame)
        
        # Handle key presses
        code = cv2.waitKey(1) & 0xFF
        if code == 27 or code == ord('q') or code == ord('Q'):
            break
        elif code == ord('c') or code == ord('C'):
            calibrator.start_center_calibration()
        elif code == ord('d') or code == ord('D'):
            if NEW_CALIBRATION_AVAILABLE:
                print("Starting directional calibration...")
                cap.release()  # Release camera for calibration
                
                # Run directional calibration
                from gaze_calibrator import GazeCalibrationSystem
                directional_calibrator = GazeCalibrationSystem(
                    samples_per_position=20,
                    sample_collection_time=3.0,
                    output_file=args.directional_calibration
                )
                if directional_calibrator.run_calibration():
                    # Reload calibration
                    calibrator.directional_interpreter = GazeDirectionInterpreter(
                        args.directional_calibration
                    )
                
                # Reopen camera
                cap = cv2.VideoCapture(1)
            else:
                print("Directional calibration not available. Please install required modules.")
        elif code == ord('r') or code == ord('R'):
            calibrator.center_reference_pitchyaw = None
            if calibrator.directional_interpreter:
                calibrator.directional_interpreter.is_loaded = False
            print("Calibrations reset")
        elif code == ord('m') or code == ord('M'):
            mouse_enabled = not mouse_enabled
            analog_mouse.enabled = mouse_enabled
            # directional_mouse.enabled = mouse_enabled
            # hybrid_mouse.enabled = mouse_enabled
            print(f"Mouse control: {'ENABLED' if mouse_enabled else 'DISABLED'}")
        elif code == ord('1'):
            control_mode = ControlMode.ANALOG
            print("Switched to ANALOG mode")
        elif code == ord('2'):
            if calibrator.is_calibrated("directional"):
                control_mode = ControlMode.DIRECTIONAL
                print("Switched to DIRECTIONAL mode")
            else:
                print("Directional calibration required. Press 'D' to calibrate.")
        elif code == ord('3'):
            if calibrator.is_calibrated("directional"):
                control_mode = ControlMode.HYBRID
                print("Switched to HYBRID mode")
            else:
                print("Directional calibration required for hybrid mode. Press 'D' to calibrate.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
    
    # Save calibrations on exit
    if calibrator.is_calibrated("center"):
        calibrator.save_center_calibration()
        print(f"\nCenter calibration saved: {calibrator.get_center_reference()}")