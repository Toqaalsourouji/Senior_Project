import serial
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import onnxruntime as ort

class PTC08Camera:
    """Interface for PTC08 Serial JPEG Camera Module"""
    
    # Command definitions (hex format)
    CMD_RESET = bytes([0x56, 0x00, 0x26, 0x00])
    CMD_CAPTURE_SINGLE = bytes([0x56, 0x00, 0x36, 0x01, 0x00])
    CMD_STOP_CAPTURE = bytes([0x56, 0x00, 0x36, 0x01, 0x03])
    CMD_GET_LENGTH = bytes([0x56, 0x00, 0x34, 0x01, 0x00])
    
    # Resolution commands
    CMD_RES_320x240 = bytes([0x56, 0x00, 0x31, 0x05, 0x04, 0x01, 0x00, 0x19, 0x11])
    CMD_RES_640x480 = bytes([0x56, 0x00, 0x31, 0x05, 0x04, 0x01, 0x00, 0x19, 0x00])
    CMD_RES_160x120 = bytes([0x56, 0x00, 0x31, 0x05, 0x04, 0x01, 0x00, 0x19, 0x22])
    
    def __init__(self, port='/dev/serial0', baudrate=38400, timeout=2):
        """Initialize serial connection to PTC08 camera
        
        Args:
            port: Serial port (default /dev/serial0 for Pi)
            baudrate: Baud rate (default 38400 for PTC08)
            timeout: Serial timeout in seconds
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera with proper settings"""
        print("Initializing PTC08 camera...")
        time.sleep(2.5)  # Required delay after power on
        
        # Set resolution to 640x480 for better quality
        self.set_resolution('640x480')
        time.sleep(0.5)
        
        # Reset camera
        self.reset()
        time.sleep(1)
        
        print("Camera initialized")
    
    def send_command(self, cmd: bytes) -> bytes:
        """Send command and receive response"""
        self.ser.write(cmd)
        time.sleep(0.1)
        response = self.ser.read(self.ser.in_waiting or 1)
        return response
    
    def reset(self):
        """Reset the camera"""
        response = self.send_command(self.CMD_RESET)
        if response[:2] == bytes([0x76, 0x00]):
            print("Reset successful")
        else:
            print(f"Reset failed: {response.hex()}")
    
    def set_resolution(self, resolution: str):
        """Set camera resolution
        
        Args:
            resolution: '320x240', '640x480', or '160x120'
        """
        resolutions = {
            '320x240': self.CMD_RES_320x240,
            '640x480': self.CMD_RES_640x480,
            '160x120': self.CMD_RES_160x120
        }
        
        if resolution in resolutions:
            response = self.send_command(resolutions[resolution])
            if response[:2] == bytes([0x76, 0x00]):
                print(f"Resolution set to {resolution}")
            else:
                print(f"Failed to set resolution: {response.hex()}")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture a single image from the camera
        
        Returns:
            OpenCV image array or None if capture failed
        """
        # Stop any ongoing capture
        self.send_command(self.CMD_STOP_CAPTURE)
        time.sleep(0.1)
        
        # Capture single image
        response = self.send_command(self.CMD_CAPTURE_SINGLE)
        if response[:2] != bytes([0x76, 0x00]):
            print("Capture command failed")
            return None
        
        time.sleep(0.5)  # Wait for capture to complete
        
        # Get image data length
        response = self.send_command(self.CMD_GET_LENGTH)
        if len(response) < 9 or response[:2] != bytes([0x76, 0x00]):
            print("Failed to get image length")
            return None
        
        # Extract image length (bytes 5 and 6)
        img_length = (response[5] << 8) | response[6]
        print(f"Image size: {img_length} bytes")
        
        # Read image data
        jpeg_data = self.read_image_data(img_length)
        if jpeg_data is None:
            return None
        
        # Convert JPEG to OpenCV array
        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    
    def read_image_data(self, length: int) -> Optional[bytes]:
        """Read image data from camera
        
        Args:
            length: Number of bytes to read
            
        Returns:
            JPEG image data or None if read failed
        """
        # Build read command
        # Format: 56 00 32 0C 00 0D 00 00 XX XX 00 00 YY YY 00 FF
        cmd = bytearray([0x56, 0x00, 0x32, 0x0C, 0x00, 0x0D])
        cmd.extend([0x00, 0x00, 0x00, 0x00])  # Start address (0)
        cmd.extend([0x00, 0x00, (length >> 8) & 0xFF, length & 0xFF])  # Length
        cmd.extend([0x00, 0xFF])
        
        self.ser.write(bytes(cmd))
        time.sleep(0.1)
        
        # Read response header (5 bytes)
        header = self.ser.read(5)
        if header[:2] != bytes([0x76, 0x00]):
            print("Failed to read image data")
            return None
        
        # Read JPEG data
        jpeg_data = bytearray()
        bytes_to_read = length
        
        while bytes_to_read > 0:
            chunk_size = min(bytes_to_read, 2048)
            chunk = self.ser.read(chunk_size)
            if not chunk:
                print("Timeout reading image data")
                return None
            jpeg_data.extend(chunk)
            bytes_to_read -= len(chunk)
        
        # Read footer (5 bytes)
        footer = self.ser.read(5)
        
        # Verify JPEG markers
        if jpeg_data[:2] == bytes([0xFF, 0xD8]) and jpeg_data[-2:] == bytes([0xFF, 0xD9]):
            print("Valid JPEG received")
            return bytes(jpeg_data)
        else:
            print("Invalid JPEG data")
            return None
    
    def close(self):
        """Close serial connection"""
        if self.ser.is_open:
            self.ser.close()


class GazeTrackerWithPTC08:
    """Gaze tracker using PTC08 camera and ONNX model"""
    
    def __init__(self, model_path: str, camera_port='/dev/serial0'):
        """Initialize gaze tracker with PTC08 camera
        
        Args:
            model_path: Path to ONNX model file
            camera_port: Serial port for PTC08 camera
        """
        # Initialize camera
        self.camera = PTC08Camera(camera_port)
        
        # Initialize ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Model parameters (from your original code)
        self.input_size = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.input_mean) / self.input_std
        
        # Add batch dimension and transpose to CHW format
        batch = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(batch, axis=0).astype(np.float32)
        
        return batch
    
    def estimate_gaze(self, image: np.ndarray) -> Tuple[float, float]:
        """Estimate gaze from image
        
        Args:
            image: Input image
            
        Returns:
            (pitch, yaw) in radians
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pitch_logits, yaw_logits = outputs
        
        # Apply softmax
        pitch_prob = np.exp(pitch_logits) / np.sum(np.exp(pitch_logits))
        yaw_prob = np.exp(yaw_logits) / np.sum(np.exp(yaw_logits))
        
        # Calculate angles
        pitch = np.sum(pitch_prob * self.idx_tensor) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_prob * self.idx_tensor) * self._binwidth - self._angle_offset
        
        return np.radians(pitch), np.radians(yaw)
    
    def capture_and_track(self):
        """Capture image and track gaze"""
        print("Capturing image...")
        img = self.camera.capture_image()
        
        if img is not None:
            print(f"Image captured: {img.shape}")
            
            # Estimate gaze
            pitch, yaw = self.estimate_gaze(img)
            print(f"Gaze angles - Pitch: {np.degrees(pitch):.2f}°, Yaw: {np.degrees(yaw):.2f}°")
            
            # Display image with gaze direction
            self.visualize_gaze(img, pitch, yaw)
            
            return img, pitch, yaw
        else:
            print("Failed to capture image")
            return None, None, None
    
    def visualize_gaze(self, img: np.ndarray, pitch: float, yaw: float):
        """Visualize gaze direction on image"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate arrow endpoint
        arrow_length = min(w, h) // 4
        dx = int(arrow_length * np.sin(pitch) * np.cos(yaw))
        dy = int(arrow_length * np.sin(yaw))
        endpoint = (center[0] + dx, center[1] + dy)
        
        # Draw arrow
        cv2.arrowedLine(img, center, endpoint, (0, 0, 255), 3)
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        
        # Show image
        cv2.imshow("Gaze Tracking", img)
        cv2.waitKey(0)
    
    def run_continuous(self, delay=1):
        """Run continuous gaze tracking
        
        Args:
            delay: Delay between captures in seconds
        """
        print("Starting continuous tracking. Press 'q' to quit.")
        
        while True:
            img, pitch, yaw = self.capture_and_track()
            
            if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.camera.close()


# Example usage
if __name__ == "__main__":
    # Initialize tracker with your model
    tracker = GazeTrackerWithPTC08(
        model_path="mobileone_s0_gaze.onnx",
        camera_port="/dev/serial0"
    )
    
    # Single capture
    img, pitch, yaw = tracker.capture_and_track()
    
    # Or run continuous tracking
    # tracker.run_continuous(delay=2)
