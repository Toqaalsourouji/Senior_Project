#!/usr/bin/env python3
"""
Gaze Detection Integration Module
==================================
This module provides the actual gaze detection implementation that integrates
with the calibration system, replacing the mock implementation.
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple, List
import mediapipe as mp
# Import from the original gaze tracking code
from demo_utils import multiclass_nms, demo_postprocess


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class GazeDetector:
    """
    Actual implementation of gaze detection using the provided models.
    This class bridges the gap between the calibration system and the 
    existing gaze tracking pipeline.
    """
    
    # Face model for head pose estimation (from original code)
    FACE_MODEL = np.float32([
        [-63.833572,  63.223045,  41.1674  ],  # RIGHT_EYEBROW_RIGHT
        [-12.44103 ,  66.60398 ,  64.561584],  # RIGHT_EYEBROW_LEFT
        [ 12.44103 ,  66.60398 ,  64.561584],  # LEFT_EYEBROW_RIGHT
        [ 63.833572,  63.223045,  41.1674  ],  # LEFT_EYEBROW_LEFT
        [-49.670784,  51.29701 ,  37.291245],  # RIGHT_EYE_RIGHT
        [-16.738844,  50.439426,  41.27281 ],  # RIGHT_EYE_LEFT
        [ 16.738844,  50.439426,  41.27281 ],  # LEFT_EYE_RIGHT
        [ 49.670784,  51.29701 ,  37.291245],  # LEFT_EYE_LEFT
        [-18.755981,  13.184412,  57.659172],  # NOSE_RIGHT
        [ 18.755981,  13.184412,  57.659172],  # NOSE_LEFT
        [-25.941687, -19.458733,  47.212223],  # MOUTH_RIGHT
        [ 25.941687, -19.458733,  47.212223],  # MOUTH_LEFT
        [  0.      , -29.143637,  57.023403],  # LOWER_LIP
        [  0.      , -69.34913 ,  38.065376]   # CHIN
    ])
    
    # Tracked landmark points for critical features
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
    
    def __init__(self, face_session, landmark_session, gaze_session, 
                 cam_width: int = 640, cam_height: int = 480):
        """
        Initialize the gaze detector with pre-loaded ONNX sessions.
        
        Args:
            face_session: ONNX session for face detection
            landmark_session: ONNX session for landmark detection
            gaze_session: ONNX session for gaze estimation
            cam_width: Camera width in pixels
            cam_height: Camera height in pixels
        """
        self.face_session = face_session
        self.gaze_session = gaze_session
        
        # Camera parameters
        self.cam_w = cam_width
        self.cam_h = cam_height
        
        # Calculate camera matrix
        c_x = cam_width / 2
        c_y = cam_height / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x
        self.camera_matrix = np.float32([
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0]
        ])
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    
    
    def get_mediapipe_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe Face Mesh and return 2D landmarks in image coords."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        mesh = result.multi_face_landmarks[0].landmark
        landmark = np.array([(p.x * w, p.y * h) for p in mesh], dtype=np.float32)
        return landmark
    
    
    def yolox_preprocess(self, img: np.ndarray, input_size: Tuple[int, int], 
                        swap: Tuple[int, int, int] = (2, 0, 1)) -> Tuple[np.ndarray, float]:
        """Preprocess image for YOLOX face detection."""
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
    
    def detect_face(self, img: np.ndarray, score_thr: float = 0.5, 
                   input_shape: Tuple[int, int] = (160, 128)) -> Optional[np.ndarray]:
        """
        Detect faces in the image.
        
        Args:
            img: Input image
            score_thr: Score threshold for detection
            input_shape: Input shape for the model
        
        Returns:
            Array of face detections or None if no faces found
        """
        img_preprocessed, ratio = self.yolox_preprocess(img, input_shape)
        ort_inputs = {self.face_session.get_inputs()[0].name: img_preprocessed[None, :, :, :]}
        output = self.face_session.run(None, ort_inputs)
        
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
            return np.array([[*final_box, final_score] 
                           for final_box, final_score in zip(final_boxes, final_scores)])
        return None
    
    
    def extract_critical_landmarks(self, landmark: np.ndarray) -> np.ndarray:
        """Extract critical landmarks for head pose estimation (MP indices)."""
        base_landmark = landmark[:468]   # use only canonical 468 points
        critical_landmarks = base_landmark[self.TRACKED_POINTS]
        return critical_landmarks[:, :2].astype(np.float32)
    
    def estimate_head_pose(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate head pose from full MediaPipe landmarks.

        Returns:
            (rvec, tvec)
        """
        landmarks_2d = self.extract_critical_landmarks(landmarks)
        ret, rvec, tvec = cv2.solvePnP(
            self.FACE_MODEL,
            landmarks_2d,
            self.camera_matrix,
            self.camera_distortion
        )
        return rvec, tvec
    
    def normalize_data_for_inference(self, img: np.ndarray, hr: np.ndarray, 
                                    ht: np.ndarray) -> List:
        """
        Normalize image data for gaze inference.
        
        Args:
            img: Input image
            hr: Head rotation vector
            ht: Head translation vector
        
        Returns:
            List containing normalized eye and face images with rotation matrix
        """
        focal_norm = 960
        distance_norm_eye = 700
        distance_norm_face = 1200
        roiSize_eye = (60, 60)
        roiSize_face = (120, 120)
        
        ht = ht.reshape((3,1))
        hR = cv2.Rodrigues(hr)[0]
        Fc = np.dot(hR, self.FACE_MODEL.T) + ht
        
        re = 0.5*(Fc[:,4] + Fc[:,5]).T
        le = 0.5*(Fc[:,6] + Fc[:,7]).T
        fe = (1./6.)*(Fc[:,4] + Fc[:,5] + Fc[:,6] + Fc[:,7] + Fc[:,10] + Fc[:,11]).T
        
        data = []
        for distance_norm, roiSize, et in zip(
            [distance_norm_eye, distance_norm_eye, distance_norm_face],
            [roiSize_eye, roiSize_eye, roiSize_face],
            [re, le, fe]
        ):
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
            
            W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(self.camera_matrix)))
            img_warped = cv2.warpPerspective(img, W, roiSize)
            data.append(img_warped)
            
            if distance_norm == distance_norm_face:
                data.append(R)
        
        return data
    
    def euler_to_vec(self, theta: float, phi: float) -> np.ndarray:
        """Convert Euler angles to direction vector."""
        x = -1 * np.cos(theta) * np.sin(phi)
        y = -1 * np.sin(theta)
        z = -1 * np.cos(theta) * np.cos(phi)
        vec = np.array([x, y, z])
        vec = vec / np.linalg.norm(vec)
        return vec
    
    def vec_to_euler(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Convert direction vector to Euler angles."""
        theta = np.arcsin(-y)
        phi = np.arctan2(-x, -z)
        return theta, phi
    
    def estimate_gaze(self, img: np.ndarray, landmark: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Estimate gaze direction from image and landmarks.
        
        Args:
            img: Input image
            landmark: Facial landmarks
        
        Returns:
            Tuple of (pitch, yaw) in radians, or None if estimation failed
        """
        try:
            # Estimate head pose
            rvec, tvec = self.estimate_head_pose(landmark)
            
            # Normalize data for inference
            data = self.normalize_data_for_inference(img, rvec, tvec)
            leye_image, reye_image, face_image, R = data
            
            # Prepare inputs for gaze model
            leye_image = np.ascontiguousarray(leye_image, dtype=np.float32) / 255.0
            reye_image = np.ascontiguousarray(reye_image, dtype=np.float32) / 255.0
            face_image = np.ascontiguousarray(face_image, dtype=np.float32) / 255.0
            
            leye_image = np.transpose(np.expand_dims(leye_image, 0), (0,3,1,2))
            reye_image = np.transpose(np.expand_dims(reye_image, 0), (0,3,1,2))
            face_image = np.transpose(np.expand_dims(face_image, 0), (0,3,1,2))
            
            # Run gaze estimation
            ort_inputs = {
                self.gaze_session.get_inputs()[0].name: leye_image,
                self.gaze_session.get_inputs()[1].name: reye_image,
                self.gaze_session.get_inputs()[2].name: face_image
            }
            
            pred_pitchyaw_aligned = self.gaze_session.run(None, ort_inputs)[0][0]
            pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
            
            # Convert to camera coordinates
            pred_vec_aligned = self.euler_to_vec(*pred_pitchyaw_aligned)
            pred_vec_cam = np.dot(np.linalg.inv(R), pred_vec_aligned)
            pred_vec_cam /= np.linalg.norm(pred_vec_cam)
            pred_pitchyaw_cam = np.array(self.vec_to_euler(*pred_vec_cam))
            
            return tuple(pred_pitchyaw_cam)
            
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            return None
    
    def get_gaze_from_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Complete pipeline to get gaze from a single frame.
        
        Args:
            frame: Input frame from camera
        
        Returns:
            Tuple of (pitch, yaw) in radians, or None if detection failed
        """
        # Detect face
        faces = self.detect_face(frame)
        if faces is None or len(faces) == 0:
            return None
        
        # Detect landmarks
        landmark = self.get_mediapipe_landmarks(frame)
        if landmark is None:
            return None

        
        # Estimate gaze
        gaze_pitchyaw = self.estimate_gaze(frame, landmark)
        
        return gaze_pitchyaw


def integrate_gaze_detector_with_calibrator(calibrator_instance):
    """
    Function to integrate the actual GazeDetector with the calibration system.
    This replaces the mock _get_current_gaze method in GazeCalibrationSystem.
    
    Usage:
        calibrator = GazeCalibrationSystem()
        integrate_gaze_detector_with_calibrator(calibrator)
        calibrator.run_calibration()
    """
    # Create the gaze detector with the loaded sessions
    gaze_detector = GazeDetector(
        calibrator_instance.face_detection_session,
        calibrator_instance.landmark_detection_session,
        calibrator_instance.gaze_estimation_session
    )
    
    # Replace the mock method with actual implementation
    def _get_current_gaze_actual(self) -> Optional[Tuple[float, float]]:
        """Get the current gaze direction from the camera using actual detection."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        # Use the actual gaze detector
        gaze_data = gaze_detector.get_gaze_from_frame(frame)
        
        if gaze_data is not None:
            # Apply smoothing if available
            if hasattr(self, 'gaze_smoother'):
                import time
                timestamp = time.time()
                gaze_data = self.gaze_smoother(gaze_data, t=timestamp)
        
        return gaze_data
    
    # Bind the new method to the calibrator instance
    calibrator_instance._get_current_gaze = _get_current_gaze_actual.__get__(
        calibrator_instance, calibrator_instance.__class__
    )
    
    print("Gaze detector integrated with calibration system successfully")