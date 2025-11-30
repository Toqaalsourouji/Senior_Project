import serial
import serial.tools.list_ports
import time
import ctypes
import importlib.util
import subprocess
import sys
import pynput
import threading
import queue
import signal
from pynput.mouse import Controller as MouseController
import numpy as np
from Calibration import CalibrationDisplay
from User_Guide import SightSyncWelcome 

#add all lirbaries tob einstalled
def install_if_missing(package, pypi_name=None):
    if pypi_name is None:
        pypi_name = package

    if importlib.util.find_spec(package) is None:
        print(f"{package} not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pypi_name])
    else:
        print(f"{package} already installed.")

# Tkinter
try:
    import tkinter
except:
    install_if_missing("tkinter", "tk")

# cv2
try:
    import cv2
except:
    install_if_missing("cv2", "opencv-python")


# Now safely import your GUI
from tkinter import *
import tkinter.font as font
from tkinter import messagebox

# -----------------------------------
# 1. Get PC Screen Resolution
# -----------------------------------


# -----------------------------------
# 2. Auto-find COM port
# -----------------------------------
def find_pi_port():
    for p in serial.tools.list_ports.comports():
        # Print everything for debugging
        print("Found:", p.device, p.description, "VID:", p.vid, "PID:", p.pid)

        # Pi Zero gadget Serial (g_serial)
        if p.vid == 0x525 and p.pid == 0xA4A7:
            return p.device
    return None

PORT = find_pi_port()
while not PORT:
    PORT = find_pi_port()
    print("⚠️ No COM port found from Raspberry Pi!")
    #exit()

print(f"Using COM port: {PORT}")


# -----------------------------------
# 3. Direction and Command Functions
# -----------------------------------



import tkinter as tk


class CenterCalibrator:
    """Simple center-point calibration for analog mouse control."""
    
    def __init__(self):
        self.center_reference_pitchyaw = None
        self.calibration_data = []
        self.is_calibrating = False
        self.calibration_start_time = None
        self.sample_duration = 2.0
        self.calibration_completed = False
    
    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        self.calibration_data = []
    
    def add_sample(self, pitchyaw):
        if not self.is_calibrating:
            return False
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - self.calibration_start_time
        
        if elapsed < self.sample_duration:
            self.calibration_data.append(pitchyaw)
            return False
        else:
            self.is_calibrating = False
            if len(self.calibration_data) > 0:
                samples = np.array(self.calibration_data)
                self.center_reference_pitchyaw = tuple(np.median(samples, axis=0))
                self.calibration_completed = True
                return True
            return False
    
    def is_calibrated(self):
        return self.center_reference_pitchyaw is not None
    
    def needs_calibration(self):
        return not self.calibration_completed
    
    def get_reference(self):
        return self.center_reference_pitchyaw

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

    def _gaze_to_velocity(self, yaw_deg, pitch_deg):
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

        t = t ** 1.2
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

# -----------------------------------
# 4. MCU Handler Class
# -----------------------------------
class MCUHandler:
    def __init__(self, root, keyboard):
        self.root = root
        self.keyboard = keyboard
        self.command_queue = queue.Queue()
        self.running = True  # Add flag
        self.analog_mouse = AnalogGazeMouseController(
            dead_zone_deg=5.0,
            max_angle_deg=20.0,
            min_speed=0.0,
            max_speed=50.0,
            smooth_factor=0.35
        )
        self.cali = CalibrationDisplay(30, 50)
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.scrollmode = False
        try:
            self.ser = serial.Serial(PORT, 115200, timeout=1)
        except:
            self.ser = None
            print("⚠️ Could not open serial port")
        self.reader_thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.reader_thread.start()
        self.keyboardstate = False
        self.process_commands()
        self.update_mouse_continuously()


    def update_mouse_continuously(self):
        """Continuously update mouse position with last known gaze"""
        if (self.keyboardstate == False and self.scrollmode == False):
            self.analog_mouse.update(self.last_pitch, self.last_yaw)        
        self.root.after(10, self.update_mouse_continuously)

        
    def velocity_to_direction(self, v, deadzone=0.1):
        dx, dy = v
        if np.linalg.norm(v) < deadzone:
            return "CENTER"
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"

    def serial_reader(self):
        """Blocking reader - runs in background thread"""
        while self.running:  # Check flag
            try:
                if self.ser:
                    line = self.ser.readline().decode(errors="ignore").strip()
                else:
                    # For testing without serial
                    line = ""
                    time.sleep(0.1)
                
                if line:
                    print("Received:", line)
                    self.command_queue.put(line)
            except Exception as e:
                if self.running:  # Only print if still running
                    print(f"Serial error: {e}")
                time.sleep(1)
    
    def stop(self):
        """Stop the serial reader"""
        self.running = False
        if self.ser:
            self.ser.close()
    
    def process_commands(self):
        """Non-blocking processor - runs on main thread"""
        try:
            while not self.command_queue.empty():
                line = self.command_queue.get_nowait()
                self.handle_command(line)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(10, self.process_commands)
    
    def handle_command(self, line):
        """Process a single command"""
        if line.startswith("CALST"):
            try:
                self.root.after(0, lambda: self.cali.start_calibration(self.cali.on_pi_signal_received))

            except Exception as e:
                print("⚠️ Parse error:", e)
        if line.startswith("CALNXT"):
            try:
                self.root.after(0, lambda: self.cali.on_pi_signal_received())
            except Exception as e:
                print("⚠️ Parse error:", e)
        elif line.startswith("GAZE:"):
            try:
                parts = line.replace("GAZE:", "").replace(" ", "")
                xy = parts.split(",")
                self.last_yaw = float(xy[0].split("=")[1])
                self.last_pitch = float(xy[1].split("=")[1]) 
            
                if (self.keyboardstate == False):
                    print("Moving Mouse to Yaw:", self.last_yaw, "Pitch:", self.last_pitch)
                        #self.analog_mouse.update(calibrated_pitch_deg, calibrated_yaw_deg)
                elif (self.keyboardstate == True):
                    velocity = self.analog_mouse._gaze_to_velocity(self.last_yaw, self.last_pitch)
                    direction = self.velocity_to_direction(velocity)
                    self.get_direction_VK(direction)

                elif (self.scrollmode == True):
                    if(self.last_pitch>10):
                        self.analog_mouse.mouse.scroll(0, -1)
                        print("Scrolled Down")
                    elif(self.last_pitch<-10):
                        self.analog_mouse.mouse.scroll(0, 1)
                        print("Scrolled Up")


            except Exception as e:
                print("⚠️ Parse error:", e)
        elif line.endswith("click"):
            self.get_and_map_clicks(line)
        
        elif line.startswith("scroll"):
            self.scroll_mode(line)
        
        elif line.startswith("keyboard_toggle"):
            self.vk_mode(line)
        
        else:
            print("Unknown:", line)

        
    def get_direction_VK(self, direction):
      """Navigate virtual keyboard"""
      if direction == "CENTER":
          return
      elif direction == "RIGHT":
          self.keyboard.navigate("right")
      elif direction == "LEFT":  # etc...
          self.keyboard.navigate("left")
      elif direction == "UP":
          self.keyboard.navigate("up")
      elif direction == "DOWN":
          self.keyboard.navigate("down")
    
    def get_and_map_clicks(self, clicks):
        if clicks == "right_click":
            self.analog_mouse.mouse.click(pynput.mouse.Button.right, 1)
            print("Right Clicked")
        elif clicks == "left_click":
            self.analog_mouse.mouse.click(pynput.mouse.Button.left, 1)
            print("Left Clicked")
        else:
            print("No Click Action")
    
    def scroll_mode(self, scroll_command):
        if scroll_command == "scroll_on":
            self.scrollmode = True
            print("Scrolled Up")
        elif scroll_command == "scroll_off":
            self.scrollmode = False
            print("Scrolled Down")
        else:
            print("No Scroll Action")
    
    def vk_mode(self, vk_command):
        if self.keyboardstate == False:
            self.keyboard.master.deiconify()  # Show keyboard
            self.keyboardstate = True
            print("Virtual Keyboard Opened")
        else:
            self.keyboard.master.withdraw()  # Hide keyboard
            self.keyboardstate = False
            print("Virtual Keyboard Closed")



# -----------------------------------
# 5. Main - Setup and Run
# -----------------------------------
# At the end of your script, before mainloop():
# At the end of your script, BEFORE creating the main root:
if __name__ == "__main__":


    welcome = SightSyncWelcome(5)
    welcome.run()

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    SCREEN_W = user32.GetSystemMetrics(0)
    SCREEN_H = user32.GetSystemMetrics(1)

    CENTER_X = SCREEN_W // 2
    CENTER_Y = SCREEN_H // 2
    DEADZONE = SCREEN_W // 10

    print(f"PC Screen Resolution: {SCREEN_W}x{SCREEN_H}")
    from keyko import main as vk_main

    # NOW create the main application (after welcome closes)
    root = tk.Tk()
    root.withdraw()
    
    keyboard = vk_main()
    keyboard.master.withdraw()
    
    mcu_handler = MCUHandler(root, keyboard)
    
    def signal_handler(sig, frame):
        print("\nShutting down...")
        mcu_handler.stop()
        root.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    keyboard.master.protocol("WM_DELETE_WINDOW", lambda: keyboard.master.withdraw())
    root.mainloop()


