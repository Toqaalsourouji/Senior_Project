import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller, Button
import time
import platform
import argparse
import os
import pyautogui 
from collections import deque
from time import sleep     



EAR_CLOSE_THRESHOLD = 0.20  # Eye Aspect Ratio threshold to indicate closed eyes
EAR_OPEN_THRESHOLD = 0.25   # Eye Aspect Ratio threshold to indicate open eyes
BLINK_CONSEC_FRAMES = 3    # Number of consecutive frames the eye must be below the threshold
ACTION_COOLDOWN_MS = 300    # Cooldown period after an action is triggered
SCROLL_AMOUNT = 250         # Amount to scroll on each blink action
num_of_clicks = 1           # Number of clicks for left click action


# MediaPipe eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def now_ms():
    return int(time.time() * 1000)

def eye_center(landmarks, eye_indices):
    xs = [landmarks[i][0] for i in eye_indices]
    ys = [landmarks[i][1] for i in eye_indices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def polygon_center(landmarks, idxs):
    xs = [landmarks[i][0] for i in idxs]
    ys = [landmarks[i][1] for i in idxs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def eye_aspect_ratio(landmarks, eye_indices) -> float:

    #Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance 1
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance 2
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

def right_eye_vertical_gaze(landmarks, right_eye_indices):
    top_candidates = [min(right_eye_indices, key=lambda i: landmarks[i][1])]
    bottom_candidates = [max(right_eye_indices, key=lambda i: landmarks[i][1])]
    top_y = sum(landmarks[i][1] for i in top_candidates) / len(top_candidates)
    bottom_y = sum(landmarks[i][1] for i in bottom_candidates) / len(bottom_candidates)

    # Choose iris center if available
    try:
        iris_cx, iris_cy = polygon_center(landmarks, RIGHT_EYE)
        center_y = iris_cy
    except:
        _, center_y = eye_center(landmarks, RIGHT_EYE)
    
    # Normalized position of the center within the eye box (0 = top, 1 = bottom)
    denom = max(1, bottom_y - top_y)
    ratio = (center_y - top_y) / denom

    # Thresholds can be adjusted based on user testing
    if ratio < 0.35:
        return "UP"
    elif ratio > 0.65:
        return "DOWN"
    else:
        return "CENTER"


def open_virtual_keyboard():
    try:
        if platform.system() == "Windows":
            # Use keyboard shortcut: Ctrl + Windows + O
            #pyautogui.hotkey('ctrl', 'win', 'o')
            time.sleep(0.5)  # Give keyboard time to open
           # print("Touch keyboard opened (Ctrl+Win+O).")
            return True
        else:
            print("Virtual keyboard opening not supported on this OS.")
    except Exception as e:
        print(f"Error opening touch keyboard: {e}")
    return False

def close_virtual_keyboard():
    try:
        if platform.system() == "Windows":
            # Ctrl + Win + O toggles the keyboard
            pyautogui.hotkey('ctrl', 'win', 'o')
            time.sleep(0.3)
            print("Touch keyboard closed (Ctrl+Win+O).")
            return True
        else:
            print("Virtual keyboard closing not supported on this OS.")
    except Exception as e:
        print(f"Error closing touch keyboard: {e}")
    return False

def is_virtual_keyboard_open():
    try:
        if platform.system() == "Windows":
            # Check for TabTip or TextInputHost process (Windows 11)
            import psutil
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in ['TabTip.exe', 'TextInputHost.exe']:
                    return True
        return False
    except:
        # If we can't detect, assume it follows toggle behavior
        return False

def toggle_virtual_keyboard():
    try:
        if platform.system() == "Windows":
            # Ctrl + Win + O is a toggle shortcut
            pyautogui.hotkey('ctrl', 'win', 'o')
            time.sleep(0.3)
            
            # Try to detect current state
            is_open = is_virtual_keyboard_open()
            status = "opened" if is_open else "toggled"
            #print(f"✓ Touch keyboard {status}.")
            return True
        else:
            print("Virtual keyboard not supported on this OS.")
    except Exception as e:
        print(f"Error toggling keyboard: {e}")
    return False



class BlinkDetection:
    def __init__(self):
        # Single blink counters
        self.left_blink_frames = 0
        self.right_blink_frames = 0
        self.both_blink_frames = 0

        # Double blink timing
        self.last_both_blink_ms = 0
        self.both_blink_count = 0
        self.double_blink_interval_ms = 500  # Max time between blinks for double blink
        self.pending_single_click = False    # Track pending single click
        self.pending_click_time = 0          # When the pending click was registered

        # Action cooldown
        self.last_action_ms = 0
        self.action_cooldown_ms = ACTION_COOLDOWN_MS

        # Keyboard state
        self.keyboard_open = False
        #Store latest action for the external access
        self.latest_action = None
        self.scroll_mode_active = False

    def reset_blink_counters(self):
        self.left_blink_frames = 0
        self.right_blink_frames = 0
        self.both_blink_frames = 0
    
    def can_trigger_action(self, now_ms):
        return (now_ms - self.last_action_ms) >= self.action_cooldown_ms
    
    def update_blink_state(self, left_closed, right_closed, left_open, right_open):
        if left_closed and right_closed:
            self.both_blink_frames += 1
            self.left_blink_frames = 0
            self.right_blink_frames = 0
            return 'both'
        elif left_closed and right_open:
            self.left_blink_frames += 1
            self.right_blink_frames = 0
            self.both_blink_frames = 0
            return 'left'
        elif right_closed and left_open:
            self.right_blink_frames += 1
            self.left_blink_frames = 0
            self.both_blink_frames = 0
            return 'right'
        else:
            # Eyes open - check if we just completed a blink
            had_both_blink = self.both_blink_frames >= BLINK_CONSEC_FRAMES

            self.reset_blink_counters()

            if had_both_blink:
                return 'both_complete'
            return None
        
    def process_double_blink(self, now_ms):
        time_since_last_blink = now_ms - self.last_both_blink_ms

        # First blink in sequence
        if self.both_blink_count == 0:
            self.both_blink_count = 1
            self.last_both_blink_ms = now_ms
            self.pending_single_click = True
            self.pending_click_time = now_ms
            return 'wait'
        
        # Second blink within interval → DOUBLE BLINK!
        elif time_since_last_blink < self.double_blink_interval_ms:
            self.both_blink_count = 0
            self.last_both_blink_ms = 0
            self.pending_single_click = False
            return 'double_blink'
        
        # Too slow - reset and treat first blink as single
        else:
            self.both_blink_count = 1
            self.last_both_blink_ms = now_ms
            self.pending_single_click = True
            self.pending_click_time = now_ms
            return 'single_click'
    
    def check_pending_click(self, now_ms):
        if self.pending_single_click:
            time_since_pending = now_ms - self.pending_click_time
            if time_since_pending >= self.double_blink_interval_ms:
                self.pending_single_click = False
                return True
        return False
    
    def handle_blink_actions(self, blink_type, landmarks, mouse, now_ms):
        # FIRST: Check if we have a pending single click that timed out
        if self.check_pending_click(now_ms):
            if self.can_trigger_action(now_ms):
                mouse.click(Button.left, num_of_clicks )  # 1 distinguish from double blink   number of clicks 
                self.last_action_ms = now_ms
                return "left_click" 
        
        if not self.can_trigger_action(now_ms):

            return None
        
        #BOTH EYES BLINK COMPLETE
        if blink_type == 'both_complete':
            blink_result = self.process_double_blink(now_ms)
            
            if blink_result == 'both_complete':
                # DOUBLE BLINK → Toggle keyboard
                toggle_virtual_keyboard()
                self.keyboard_open = is_virtual_keyboard_open()
                self.last_action_ms = now_ms
               # print(f"[DEBUG] Double blink detected! Keyboard state: {self.keyboard_open}")
                return "keyboard_toggle"
            
            elif blink_result == 'single_click':
                # Previous blink timed out, this is a new first blink
                return None
            
            elif blink_result == 'wait':
                # First blink, waiting for potential second
                #print(f"[DEBUG] First blink detected, waiting for second...")
                return None
        
        #  RIGHT EYE BLINK 
        elif blink_type == 'right' and self.right_blink_frames >= BLINK_CONSEC_FRAMES:
            mouse.click(Button.right, 1)
            self.last_action_ms = now_ms
            self.reset_blink_counters()
            return "right_click"
        
        #LEFT EYE BLINK
        elif blink_type == 'left' and self.left_blink_frames >= BLINK_CONSEC_FRAMES:
            toggle_virtual_keyboard()
            self.keyboard_open = is_virtual_keyboard_open()
            self.last_action_ms = now_ms
            print(f"[DEBUG] Double blink detected! Keyboard state: {self.keyboard_open}")
            return "keyboard_toggle"
        
        return None


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



def eyebrow_eye_distance(landmarks, eyebrow_idxs, eye_idxs):
    eye_cx, eye_cy = np.mean([landmarks[i][0] for i in eye_idxs]), np.mean([landmarks[i][1] for i in eye_idxs])
    brow_cx, brow_cy = np.mean([landmarks[i][0] for i in eyebrow_idxs]), np.mean([landmarks[i][1] for i in eyebrow_idxs])
    return eye_cy - brow_cy  # positive when brow is higher

#we are trying to measure how much the eybrows move upwards relative to the eyes frame by frame
#bec Mediapipe gives you the 2D pixel coordinates of landmarks
#the system sees the face as a set of points in a flat (x, y) space where each pixel represents a small distance on the screen
#so abs_thresh is the literal image pixel representuing how far a landmark moved
#in the frame and the ratio thresh is the relative movement comapred to the normal 
#distance btw the eye and the eyebrow. 
#we are using two thesh to compensate for diff faces and distance from camera

class ScrollModeDetector:
    
    def __init__(self):
        self.prev_left_dist = None
        self.prev_right_dist = None
        self.scroll_mode = False
        self.last_toggle_ms = 0
        self.cooldown_ms = 800  # ms between toggles
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)
        self.abs_thresh = 2       #changed from 4.0 to 2 pixels
        self.rel_thresh_ratio = 0.05 #changed from 0.15 to 0.05
        self.raise_frames = 0
        self.required_frames = 2 #needed frames to go in scroll mode


        
    def update(self, landmarks, now_ms, eyes_open=True):
        
        # Skip detection if eyes are closed to avoid blink interference
        if not eyes_open:
            # reset counter so a blink doesn’t accumulate raise_frames
            self.raise_frames = 0
            return self.scroll_mode
        #for each frame the left and right sides of the face we calc 
        #the vertical gap (in pixles) btw the eye center and the eyebrow center
        #and we expect an upward distance increse +-15 px
        left_dist = eyebrow_eye_distance(landmarks, [70, 63, 105], LEFT_EYE)
        right_dist = eyebrow_eye_distance(landmarks, [300, 293, 334], RIGHT_EYE)

        #on first frame, initialize baseline
        if self.prev_left_dist is None:
            self.prev_left_dist, self.prev_right_dist = left_dist, right_dist
            self.left_history.extend([left_dist] * 5)
            self.right_history.extend([right_dist] * 5)
           # print(f"[INIT] Baseline set. Left: {left_dist:.2f}, Right: {right_dist:.2f}")
            return self.scroll_mode

        
        self.left_history.append(left_dist)
        self.right_history.append(right_dist)
        smooth_left = np.mean(self.left_history)
        smooth_right = np.mean(self.right_history)
        
        #change is calc by averaing 5 frames and then comapring the new avg to the prev avg 
        #change = how many pixels your eyebrows moved upward since the last frame 
        #+ means motion up 
        change = ((left_dist - self.prev_left_dist) +
                  (right_dist - self.prev_right_dist)) / 2
        
        #for the relative ratio, the baseline is the neutral eyebrow and the rel_change 
        #is the change/baseline and this ratio is unit free and adjusts for face scale and camera distance
        
        baseline = (self.prev_left_dist + self.prev_right_dist) / 2
        rel_change = change / baseline if baseline != 0 else 0

        # Print live eyebrow distances and changes
        # print(f"[DEBUG] LeftDist={left_dist:.2f}, RightDist={right_dist:.2f}, Δ={change:.2f}")

        # Multi-frame sustained raise detection
        if change > self.abs_thresh or rel_change > self.rel_thresh_ratio:
            self.raise_frames += 1
        else:
            self.raise_frames = 0  # reset if not sustained

        # Require 3+ consecutive frames above threshold
        if self.raise_frames >= 1 and (now_ms - self.last_toggle_ms) > self.cooldown_ms:
            self.scroll_mode = not self.scroll_mode
            self.last_toggle_ms = now_ms
            self.raise_frames = 0
            print(f"Scroll mode toggled: {'ON' if self.scroll_mode else 'OFF'}")

        # Update stored distances
        self.prev_left_dist, self.prev_right_dist = smooth_left, smooth_right
        return self.scroll_mode


class BlinkEngine:
    def __init__(self):
        # Initialize mediapipe ONCE
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Persistent components (same as laptop version)
        self.blink_state = BlinkDetection()
        self.scroll_detector = ScrollModeDetector()
        self.mouse = Controller()
        self.fps_counter = FPSCounter()
        self.prev_scroll_mode = False
        
        # Frame counter (must persist)
        self.frame_count = 0

    def process(self, frame, pitch):
        self.frame_count += 1
        self.fps_counter.update()

        action = None
        scroll_mode_active = False

        # Process every 2 frames (same as laptop version)
        if self.frame_count % 2 == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h))
                             for pt in face_landmarks.landmark]

                # EAR
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

                # Eye states
                left_closed = left_ear < EAR_CLOSE_THRESHOLD
                right_closed = right_ear < EAR_CLOSE_THRESHOLD
                left_open = left_ear > EAR_OPEN_THRESHOLD
                right_open = right_ear > EAR_OPEN_THRESHOLD
                eyes_open = left_open and right_open

                # Blink type
                blink_type = self.blink_state.update_blink_state(
                    left_closed, right_closed, left_open, right_open)

                # Blink actions
                now = now_ms()
                action = self.blink_state.handle_blink_actions(
                    blink_type, landmarks, self.mouse, now)

                # Eyebrow scroll detector
                scroll_mode_active = self.scroll_detector.update(
                    landmarks, now, eyes_open=eyes_open)
            
                self.blink_state.scroll_mode_active = scroll_mode_active

                if scroll_mode_active != self.prev_scroll_mode and eyes_open:
                    if scroll_mode_active:
                        action = "scroll_on"
                    else:
                        action = "scroll_off"
                    self.prev_scroll_mode = scroll_mode_active

                if action:
                    self.blink_state.latest_action = action

        return action

    

