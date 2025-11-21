import serial
import serial.tools.list_ports
import time
import ctypes

def install_if_missing(package, pypi_name=None):
    if pypi_name is None:
        pypi_name = package

    if importlib.util.find_spec(package) is None:
        print(f"{package} not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pypi_name])
    else:
        print(f"{package} already installed.")

# tkinter (built-in, but fallback install available)
try:
    import tkinter
    print("Tkinter OK")
except:
    print("Tkinter missing. Installing fallback...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])

time.sleep(1)

# Now safely import your GUI
from tkinter import *
import tkinter.font as font
from tkinter import messagebox

# -----------------------------------
# 1. Get PC Screen Resolution
# -----------------------------------
def get_screen_resolution(): #better way
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    w = user32.GetSystemMetrics(0)
    h = user32.GetSystemMetrics(1)
    return w, h

SCREEN_W, SCREEN_H = get_screen_resolution()
CENTER_X = SCREEN_W // 2
CENTER_Y = SCREEN_H // 2
DEADZONE = SCREEN_W // 10  # 10% of width

print(f"PC Screen Resolution: {SCREEN_W}x{SCREEN_H}")

# -----------------------------------
# 2. Auto-find COM port from Raspberry Pi
# -----------------------------------
def find_com_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # Pi Zero 2W USB gadget shows as COM device on Windows
        # Check for common keywords in description
        if "USB" in p.description or "ACM" in p.description or "Serial" in p.description:
            return p.device
    return None

PORT = find_com_port()
if not PORT:
    print("⚠️ No COM port found from Raspberry Pi!")
    exit()

print(f"Using COM port: {PORT}")

# -----------------------------------
# 3. Open COM
# -----------------------------------
ser = serial.Serial(PORT, 9600, timeout=1)
time.sleep(2)

# -----------------------------------
# 4. Interpret gaze into directions
# -----------------------------------
def get_direction(x, y):
    dx = x - CENTER_X
    dy = y - CENTER_Y

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return "CENTER"

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

def get_directionXY(x, y):
    dx = x - CENTER_X
    dy = y - CENTER_Y

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return "CENTER"

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"
    
# -----------------------------------
# 5. Read from Raspberry Pi and interpret
# -----------------------------------
print("Listening for gaze data...")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        continue

    print("Received:", line)

    if line.startswith("Gaze:"):
        try:
            # Format: Gaze: x:NN, y:NN //check formatting
            parts = line.replace("Gaze:", "").replace(" ", "")
            xy = parts.split(",")

            x = int(xy[0].split(":")[1])
            y = int(xy[1].split(":")[1])

            direction = get_direction(x, y)
            print("➡️ Direction:", direction)

        except Exception as e:
            print("⚠️ Parse error:", e)
    else:
        print("Unknown:", line)


#pairing logic
#sudo systemctl enable bluetooth
#sudo systemctl start bluetooth
#systemctl status bluetooth
#bluetoothctl
#power on              # ensure Bluetooth is on
#agent on              # allow pairing
#scan on               # find PC2's Bluetooths
#pair XX:XX:XX:XX:XX:XX  # PC2 MAC address
#trust XX:XX:XX:XX:XX:XX # optional, auto-trust PC2
#connect XX:XX:XX:XX:XX:XX  # optional, test connection
#quit
#once paired, sudo rfcomm bind 0 XX:XX:XX:XX:XX:XX 1
#ls /dev/rfcomm* should see /dev/rfcomm0
#to test, echo "hello" | sudo tee /dev/rfcomm0


#pyooooooooo
#to list ls /dev/rfcomm*
#to create sudo rfcomm bind 0 XX:XX:XX:XX:XX:XX
#to test pairing echo "test" | sudo tee /dev/rfcomm0

