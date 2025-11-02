import cv2
import time
# import system
from pynput.mouse import Button, Controller as MouseController

# ==========================
# LOAD HAAR CASCADE MODELS
# ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if cascades loaded
if face_cascade.empty() or left_eye_cascade.empty() or right_eye_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar cascades")
    exit()

# ==========================
# INITIALIZE MOUSE CONTROLLER
# ==========================
mouse = MouseController()

# ==========================
# BLINK DETECTION PARAMETERS
# ==========================
LEFT_COUNTER = 0
RIGHT_COUNTER = 0
BOTH_COUNTER = 0
THRESHOLD = 1      # frames missing before confirming blink
ACTION_DELAY = 0.5  # seconds between actions
last_action_time = time.time()

# ==========================
# MAIN DETECTION LOOP
# ==========================
def detect_blinks():
    global LEFT_COUNTER, RIGHT_COUNTER, BOTH_COUNTER, last_action_time

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Starting blink detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        left_eye_detected = False
        right_eye_detected = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes using all three cascades
            left_eyes_split = left_eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            right_eyes_split = right_eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            general_eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5, minSize=(20, 20))
            
            # 

            # Combine detections: use specific cascades + verify with general cascade
            if len(left_eyes_split) > 0 or len([e for e in general_eyes if e[0] < w // 2]) > 0:
                left_eye_detected = True
                LEFT_COUNTER = 0
                for (ex, ey, ew, eh) in left_eyes_split[:1]:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, "L", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                LEFT_COUNTER += 1

            if len(right_eyes_split) > 0 or len([e for e in general_eyes if e[0] >= w // 2]) > 0:
                right_eye_detected = True
                RIGHT_COUNTER = 0
                for (ex, ey, ew, eh) in right_eyes_split[:1]:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
                    cv2.putText(roi_color, "R", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                RIGHT_COUNTER += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # ==========================
        # BLINK DETECTION LOGIC
        # ==========================
        now = time.time()

        # Both eyes missing
        if LEFT_COUNTER > THRESHOLD and RIGHT_COUNTER > THRESHOLD:
            BOTH_COUNTER += 1
        else:
            BOTH_COUNTER = 0

        # Trigger actions
        if now - last_action_time > ACTION_DELAY:
            if BOTH_COUNTER > THRESHOLD * 2:
                print("⌨️  Open Keyboard")
                last_action_time = now
                BOTH_COUNTER = 0
            elif BOTH_COUNTER > THRESHOLD * 4:
                print("📜 Scroll Mode")
                last_action_time = now
                BOTH_COUNTER = 0
            elif LEFT_COUNTER > THRESHOLD and RIGHT_COUNTER < THRESHOLD:
                print("🖱️  Right Click")
                mouse.click(Button.right, 1)
                last_action_time = now
                LEFT_COUNTER = 0
            elif RIGHT_COUNTER > THRESHOLD and LEFT_COUNTER < THRESHOLD:
                print("🖱️  Left Click")
                mouse.click(Button.left, 1)
                last_action_time = now
                RIGHT_COUNTER = 0

        # Display
        cv2.putText(frame, f"L:{LEFT_COUNTER} R:{RIGHT_COUNTER}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Blink Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_blinks()
