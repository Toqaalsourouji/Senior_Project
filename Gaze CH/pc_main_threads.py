import serial
import serial.tools.list_ports
import time
import ctypes
import importlib.util
import subprocess
import sys
import pynput
import tkinter as tk
import threading
import queue
from keyko import main as vk_main
import signal

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
def get_screen_resolution():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    w = user32.GetSystemMetrics(0)
    h = user32.GetSystemMetrics(1)
    return w, h

SCREEN_W, SCREEN_H = get_screen_resolution()
CENTER_X = SCREEN_W // 2
CENTER_Y = SCREEN_H // 2
DEADZONE = SCREEN_W // 10

print(f"PC Screen Resolution: {SCREEN_W}x{SCREEN_H}")

# -----------------------------------
# 2. Auto-find COM port
# -----------------------------------
def find_com_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "USB" in p.description or "ACM" in p.description or "Serial" in p.description:
            return p.device
    return None

PORT = find_com_port()
if not PORT:
    print("⚠️ No COM port found from Raspberry Pi!")
    # exit()

print(f"Using COM port: {PORT}")


# -----------------------------------
# 3. Direction and Command Functions
# -----------------------------------



# -----------------------------------
# 4. MCU Handler Class
# -----------------------------------
class MCUHandler:
    def __init__(self, root, keyboard):
        self.root = root
        self.keyboard = keyboard
        self.command_queue = queue.Queue()
        self.running = True  # Add flag
        
        # Open serial connection
        try:
            self.ser = serial.Serial(PORT, 9600, timeout=1)
        except:
            self.ser = None
            print("⚠️ Could not open serial port")
        
        # Start blocking serial reader in background thread
        self.reader_thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.reader_thread.start()
        
        # Start non-blocking command processor on main thread
        self.process_commands()
    
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
        if line.startswith("Gaze:"):
            try:
                parts = line.replace("Gaze:", "").replace(" ", "")
                xy = parts.split(",")
                x = int(xy[0].split(":")[1])
                y = int(xy[1].split(":")[1])
                direction = self.get_direction_XY(self, x, y)
                print("➡️ Direction:", direction)
            except Exception as e:
                print("⚠️ Parse error:", e)
        
        elif line.startswith("directon:"):
            try:
                parts = line.replace("directon:", "").replace(" ", "")
                xy = parts.split(",")
                x = int(xy[0].split(":")[1])
                y = int(xy[1].split(":")[1])
                direction = self.get_direction_XY(x, y)
                self.get_direction_VK(direction)
            except Exception as e:
                print("⚠️ Parse error:", e)
        
        elif line.startswith("click_"):
            self.get_and_map_clicks(line)
        
        elif line.startswith("scroll"):
            self.scroll_mode(line)
        
        elif line.startswith("vk_"):
            self.vk_mode(line)
        
        else:
            print("Unknown:", line)

    def get_direction_XY(x, y):
        dx = x - CENTER_X
        dy = y - CENTER_Y

        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
            return "CENTER"

        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"
        
    def get_direction_VK(self, direction):
      """Navigate virtual keyboard"""
      if direction == "CENTER":  # Change from "directon_center"
          return
      elif direction == "RIGHT":  # Change from "directon_right"
          self.keyboard.navigate("right")
          print("➡️ Right Arrow")
      elif direction == "LEFT":  # etc...
          self.keyboard.navigate("left")
          print("⬅️ Left Arrow")
      elif direction == "UP":
          self.keyboard.navigate("up")
          print("⬆️ Up Arrow")
      elif direction == "DOWN":
          self.keyboard.navigate("down")
          print("⬇️ Down Arrow")
    
    def get_and_map_clicks(self, clicks):
        if clicks == "click_right":
            pynput.mouse.Controller().click(pynput.mouse.Button.right, 1)
            print("Right Clicked")
        elif clicks == "click_left":
            pynput.mouse.Controller().click(pynput.mouse.Button.left, 1)
            print("Left Clicked")
        elif clicks == "click_vk":
            print("Virtual Keyboard Toggled")
        else:
            print("No Click Action")
    
    def scroll_mode(self, scroll_command):
        if scroll_command == "scroll_up":
            pynput.mouse.Controller().scroll(0, 2)
            print("Scrolled Up")
        elif scroll_command == "scroll_down":
            pynput.mouse.Controller().scroll(0, -2)
            print("Scrolled Down")
        else:
            print("No Scroll Action")
    
    def vk_mode(self, vk_command):
        if vk_command == "vk_open":
            self.keyboard.master.deiconify()  # Show keyboard
            print("Virtual Keyboard Opened")
        elif vk_command == "vk_close":
            self.keyboard.master.withdraw()  # Hide keyboard
            print("Virtual Keyboard Closed")
        else:
            print("No VK Action")


# -----------------------------------
# 5. Main - Setup and Run
# -----------------------------------
# At the end of your script, before mainloop():
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    keyboard = vk_main()
    keyboard.master.withdraw()
    
    mcu_handler = MCUHandler(root, keyboard)
    
    # Ctrl+C stops everything
    def signal_handler(sig, frame):
        print("\nShutting down...")
        mcu_handler.stop()
        root.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Closing keyboard just hides it
    keyboard.master.protocol("WM_DELETE_WINDOW", lambda: keyboard.master.withdraw())
    

    # Open VK
    mcu_handler.vk_mode("vk_open")

    # Test navigation after 1 second
    root.after(1000, lambda: mcu_handler.get_direction_VK("RIGHT"))
    root.after(2000, lambda: mcu_handler.get_direction_VK("DOWN"))
    root.after(3000, lambda: mcu_handler.get_direction_VK("LEFT"))
    root.after(4000, lambda: mcu_handler.get_direction_VK("UP"))

    # Test mouse clicks after 4.5 seconds
    root.after(4500, lambda: mcu_handler.get_and_map_clicks("click_left"))
    root.after(4600, lambda: mcu_handler.get_and_map_clicks("click_right"))
    root.after(4700, lambda: mcu_handler.get_and_map_clicks("click_vk"))

    # Close VK after 5 seconds
    root.after(5000, lambda: mcu_handler.vk_mode("vk_close"))

    root.after(7000, lambda: mcu_handler.vk_mode("vk_open"))

    
    root.after(12000, lambda: mcu_handler.vk_mode("vk_close"))
    keyboard.master.protocol("WM_DELETE_WINDOW", lambda: keyboard.master.withdraw())
    root.mainloop()