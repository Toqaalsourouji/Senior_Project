from picamera2 import Picamera2
import cv2
from time import sleep

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size":(640,480),"format":"RGB888"}))
cam.start()
sleep(1)

while True:
    f = cam.capture_array()
    cv2.imshow("test", f)
    if cv2.waitKey(1) == 27:
        break

cam.stop()
cv2.destroyAllWindows()
