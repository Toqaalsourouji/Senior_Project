#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime
from pynput.mouse import Controller as MouseController
import queue
import threading
import serial
from demo_utils import multiclass_nms, demo_postprocess
from smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter
import mediapipe as mp 
from gaze_utils import GazeDirectionInterpreter
import warnings 
warnings.filterwarnings("ignore", category = UserWarning , module = "google.protobuf") 

MOUTH_OPEN_THRESHOLD = 18.0     # tune by testing
mouse_frozen = False
mouth_was_open = False  # NEW


def mouth_open_amount(landmark):
    """Compute vertical distance between upper/lower inner lip."""
    upper = landmark[13]
    lower = landmark[14]
    return np.linalg.norm(lower - upper)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, #changed from true to false to boost fps (removes iris detection)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_mediapipe_landmarks(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    mesh = result.multi_face_landmarks[0].landmark
    landmark = np.array([(p.x * w, p.y * h) for p in mesh], dtype=np.float32)
    return landmark


class CenterCalibrator:
    """Simple center-point calibration for analog mouse control."""
    
    def __init__(self):
        self.center_reference_pitchyaw = None
        self.calibration_data = []
        self.is_calibrating = False
        self.calibration_start_time = None
        self.sample_duration = 2.0
        self.calibration_completed = False
        print("0L")
    
    def start_calibration(self):
        print("1L")
        self.is_calibrating = True
        self.calibration_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        self.calibration_data = []
        print("2L")
    
    def add_sample(self, pitchyaw):
        print("3L")
        if not self.is_calibrating:
            return False
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - self.calibration_start_time
        print("4L")
        if elapsed < self.sample_duration:
            self.calibration_data.append(pitchyaw)
            print("8L")
            return False
        else:
            print("9L")
            self.is_calibrating = False
            if len(self.calibration_data) > 0:
                print("10L")
                samples = np.array(self.calibration_data)
                self.center_reference_pitchyaw = tuple(np.median(samples, axis=0))
                self.calibration_completed = True
                print("11L")
                return True
            print("12L")
            return False
    
    def is_calibrated(self):
        return self.center_reference_pitchyaw is not None
    
    def needs_calibration(self):
        return not self.calibration_completed
    
    def get_reference(self):
        print("7L")
        return self.center_reference_pitchyaw



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
    base_landmark = landmark[:468]

    TRACKED_POINTS = [
        105, 66, 296, 334,
        33, 133, 362, 263,
        168, 197, 50, 280,
        1, 152
    ]

    pts = base_landmark[TRACKED_POINTS]
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

def estimateHeadPose(landmarks):
    landmarks = extract_critical_landmarks(landmarks)
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera_matrix, camera_distortion)
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


def detect_face(img, session, score_thr=0.5, input_shape=(160, 128)) -> np.ndarray:
    img, ratio = yolox_preprocess(img, input_shape)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
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
    if dets is not None:
        final_boxes, final_scores = dets[:, :4], dets[:, 4]
        return np.array([[*final_box, final_score] for final_box, final_score in zip(final_boxes, final_scores)])
    else:
        return None


def estimate_gaze(img, landmark, session) -> np.ndarray:
    rvec, tvec = estimateHeadPose(landmark)
    data = normalizeDataForInference(img, rvec, tvec)
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
    pred_pitchyaw_aligned = session.run(None, ort_inputs)[0][0]
    pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
    pred_vec_aligned = euler_to_vec(*pred_pitchyaw_aligned)
    pred_vec_cam = np.dot(np.linalg.inv(R), pred_vec_aligned)
    pred_vec_cam /= np.linalg.norm(pred_vec_cam)
    pred_pitchyaw_cam = np.array(vec_to_euler(*pred_vec_cam))
    return pred_pitchyaw_cam



sender_queue = queue.Queue()
initialized = False 

def start_gaze_loop(sender_queue, frame, timestamp, FDS, GDS): # For Toqa : FDS = Face Detection Session, GDS = Gaze Detection Session
    global face_detection_session
    global gaze_estimation_session
    global gaze_smoother
    global landmark_smoother
    global bbox_smoother
    global calibrator
    global direction_interpreter
    global initialized
    global calibration_triggered
    global mouth_was_open
    global mouth_frozen
    print("initalized : ", 1)
    if not initialized:
        #face_detection_session = onnxruntime.InferenceSession("./models/face_detection.onnx")
        face_detection_session = FDS
        gaze_estimation_session = GDS
        #gaze_estimation_session = onnxruntime.InferenceSession("./models/gaze_estimation.onnx")
        gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
        landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=478, min_cutoff=0.1, beta=1.0)
        bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
        calibrator = CenterCalibrator()
        direction_interpreter = GazeDirectionInterpreter(calibration_file="gaze_direction_calibration.json") # Loads Direction Thresholds
        initialized = True
        calibration_triggered = False

    
    cnt = 0
    # mouse_enabled = True
    #print("Frame = ", frame)
    

    faces = detect_face(frame, face_detection_session)
    if faces is None:
        print("Failed to detect faces")
        return ""
        
    


    
    landmark = get_mediapipe_landmarks(frame)
        
    #print("LANDMARK = ", landmark)

    if landmark is None:
        print("Failed to load landmarks")
        return ""
        
    print("gettin readings")
    landmark = landmark_smoother(landmark, t=timestamp)
    
       
    #print("After Landmark detection")
    #print("Frame = ", frame)
    #print("landmark = ", landmark)
    #print("gaze_estimation_seassion = ", gaze_estimation_session)
    gaze_pitchyaw = estimate_gaze(frame, landmark, gaze_estimation_session)
    
    mouth_dist = mouth_open_amount(landmark)
    mouth_is_open = mouth_dist > MOUTH_OPEN_THRESHOLD
    


    # If mouth just transitioned from CLOSED → OPEN
    if mouth_is_open and not mouth_was_open:
        print(2)
        # Toggle freeze state
        mouse_frozen = not mouse_frozen
        print(3)
        #analog_mouse.enabled = not mouse_frozen
        if mouse_frozen:
            print("Mouse FROZEN (mouth-open toggle)")
        else:
            print("Mouse UNFROZEN (mouth-open toggle)")
            
    mouth_was_open = mouth_is_open  
    
 
    
    sender_queue.put(f"GAZE:x={gaze_pitchyaw[1]},y={gaze_pitchyaw[0]},m={mouth_is_open}")
    print(f"GAZE:x={gaze_pitchyaw[1]},y={gaze_pitchyaw[0]},m={mouth_is_open}")
    
    # gaze_pitchyaw is what will be sent. 
    gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=timestamp)
    pitch_rad = float(gaze_pitchyaw[0])
    yaw_rad   = float(gaze_pitchyaw[1])
    printXBF= np.degrees(pitch_rad)
    printYBF= np.degrees(yaw_rad)
    print(f"Raw: pitch_rad={printXBF}, yaw_rad={printYBF}") 
            


    if calibrator.needs_calibration() and not calibration_triggered:
       calibrator.start_calibration()
       calibration_triggered = True

    if calibrator.is_calibrating:
       calibrator.add_sample(gaze_pitchyaw)


    # for the directional calibration
    direction = None
    if direction_interpreter.is_loaded:
       direction = direction_interpreter.interpret_gaze(pitch_rad, yaw_rad)
            
    if calibrator.is_calibrated():
        ref_pitch_rad, ref_yaw_rad = calibrator.get_reference()
        calibrated_pitch_deg = np.degrees(pitch_rad - ref_pitch_rad)
        calibrated_yaw_deg   = np.degrees(yaw_rad   - ref_yaw_rad)
    else:
        calibrated_pitch_deg = np.degrees(pitch_rad)
        calibrated_yaw_deg = np.degrees(yaw_rad)


    print(f"GAZE:x={calibrated_yaw_deg},y={calibrated_pitch_deg}")
    sender_queue.put(f"GAZE:x={calibrated_yaw_deg},y={calibrated_pitch_deg}") #here we send the gaze data to the queue, GAZE:x={calibrated_pitch_deg},y={calibrated_yaw_deg}

    # if mouse_enabled and calibrator.is_calibrated() and not calibrator.is_calibrating:
    #     analog_mouse.update(calibrated_pitch_deg, calibrated_yaw_deg)


# testing function that the outputs from here send to the queue and we can get them from the queue 
def test_queue_sender(sender_queue):
    """Test-only function to check queue sending."""
    sender_queue.put("Did the gaze reach? Yes it did!")
















