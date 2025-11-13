import serial
import time
try:
    ser = serial.Serial('/dev/rfcomm0', 9600)
    print("Connected to PC over Bluetooth")
except serial.SerialException:
    print("Failed to open Bluetooth serial port")
    exit()

while True:
    cmd = input("Enter command (left/right/click): ")
    ser.write((cmd + "\n").encode())
    time.sleep(0.2)
