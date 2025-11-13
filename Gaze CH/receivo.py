import serial
import serial.tools.list_ports
import time

#find Pi Bluetooth COM port automatically
def find_bluetooth_com():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Bluetooth" in port.description or "BTHENUM" in port.hwid:
            return port.device
    return None

PORT = find_bluetooth_com()
if not PORT:
    print("⚠️ No Bluetooth COM port found. Make sure the Pi is paired.")
    exit()

print(f"✅ Found Bluetooth COM port: {PORT}")

BAUD = 9600

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print("⚠️ Failed to open serial port:", e)
    exit()

print("Listening for commands from Raspberry Pi...")

time.sleep(2)

while True:
    try:
        data = ser.readline().decode().strip()
        if not data:
            continue

        print("Received:", data)

        if data.lower() == "left":
           print("Command got:", data)
        else:
            print("Unknown command:", data)

    except serial.SerialException:
        print("⚠️ Serial connection lost.")
        break
    except KeyboardInterrupt:
        print("Exiting...")
        break


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