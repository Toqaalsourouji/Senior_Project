#!/usr/bin/env python3
"""
Optimized sender for Raspberry Pi Zero 2W
Constraints respected:
- Camera stays at 640×480 (required)
- Blink detection processes every frame (required)
- Optimizations focus on threading and internal downsampling
"""

import serial
import time
import sys
import logging
import os
import queue
import cv2
import numpy as np
import threading
from collections import deque
from time import sleep 
from picamera2 import Picamera2
from Gaze import start_gaze_loop
from Blinking_PI_v2_TOQA import BlinkEngine 
from Calibration_Pi import GazeCalibrationSystem
import onnxruntime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera resolution (CONSTRAINT: Must stay at 640×480)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Internal downsampling for gaze only (blink uses full res)
GAZE_PROCESSING_WIDTH = 640 # ignore
GAZE_PROCESSING_HEIGHT = 480 # ignore

# Threading mode (True = much better performance)
USE_ASYNC_PROCESSING = True

# Face detection cache (detect every N frames, track in between)
FACE_CACHE_INTERVAL = 30

# ONNX optimization settings
ONNX_INTRA_THREADS = 2
ONNX_INTER_THREADS = 1

# Queue management
SENDER_QUEUE_MAXSIZE = 5  # Prevent queue backup

# Logging
os.makedirs("/tmp", exist_ok=True)
logging.basicConfig(
    filename="/tmp/sender.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("=== Optimized boot script started ===")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Thread-safe buffers
frame_buffer = deque(maxlen=2)
frame_lock = threading.Lock()

# Model sessions (loaded once)
face_detection_session = None
gaze_estimation_session = None

# Queues
sender_queue = queue.Queue(maxsize=SENDER_QUEUE_MAXSIZE)

# Face detection cache
face_bbox_cache = None
face_cache_counter = 0
face_cache_lock = threading.Lock()

# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

def find_serial_port():
    """Find USB gadget serial port."""
    try:
        ser = serial.Serial("/dev/ttyGS0", 115200, timeout=1)
        logging.info("✓ Using /dev/ttyGS0 (USB Gadget Serial)")
        return ser
    except Exception as e:
        logging.warning(f"Cannot open /dev/ttyGS0: {e}")
        return None


# ============================================================================
# ONNX MODEL LOADING
# ============================================================================

def load_optimized_onnx_model(model_path):
    """Load ONNX model with ARM CPU optimizations."""
    try:
        session_options = onnxruntime.SessionOptions()
        
        # Enable all graph optimizations
        session_options.graph_optimization_level = \
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Thread settings optimized for Pi Zero 2W
        session_options.intra_op_num_threads = ONNX_INTRA_THREADS
        session_options.inter_op_num_threads = ONNX_INTER_THREADS
        
        # Memory optimizations
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Load with CPU provider
        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        logging.info(f"✓ Loaded {model_path} with optimizations")
        return session
        
    except Exception as e:
        logging.error(f"✗ Failed to load {model_path}: {e}")
        raise


# ============================================================================
# SYNCHRONOUS PROCESSING (Fallback)
# ============================================================================

def camera_loop_sync(ser, picam2, blink_engine):
    """
    Synchronous camera loop with optimizations.
    Used if threading is disabled or as fallback.
    """
    logging.info("Starting SYNCHRONOUS camera loop")
    
    
    
    while True:
        try:
            # Capture frame at full resolution
            frame_full = picam2.capture_array()
            frame_full = frame_full[:, :, :3]
            timestamp = cv2.getTickCount() / cv2.getTickFrequency()
            
            
            start_gaze_loop(
                sender_queue, 
                frame_full,  
                timestamp, 
                face_detection_session, 
                gaze_estimation_session
            )
            
            # Send gaze data if available
            try:
                if not sender_queue.empty():
                    gaze_data = sender_queue.get_nowait()
                    if gaze_data:
                        ser.write((gaze_data + "\n").encode())
            except queue.Empty:
                pass
            
            # ---- Blink Processing (EVERY frame at full resolution) ----
            blink_action = blink_engine.process(frame_full, pitch= gaze_data[1])
            if blink_action:
                ser.write((blink_action + "\n").encode())
            
            # Small yield
            time.sleep(0.001)
            
        except KeyboardInterrupt:
            logging.info("Camera loop interrupted")
            break
        except Exception as e:
            logging.error(f"Error in camera loop: {e}", exc_info=True)
            time.sleep(0.1)


# ============================================================================
# ASYNCHRONOUS PROCESSING (Recommended)
# ============================================================================

def camera_capture_thread(picam2):
    """
    Dedicated thread for continuous camera capture at full resolution.
    Blink detection will use these frames directly.
    """
    logging.info("Camera capture thread started (640×480)")
    
    while True:
        try:
            frame = picam2.capture_array()
            frame = frame[:, :, :3]
            timestamp = time.time()
            
            # Store full-resolution frame
            with frame_lock:
                frame_buffer.append((frame.copy(), timestamp))
            
            time.sleep(0.001)
            
        except Exception as e:
            logging.error(f"Camera capture error: {e}")
            time.sleep(0.1)


def gaze_processing_thread(ser):
    """
    Dedicated thread for gaze detection.
    Uses downsampled frames for faster processing.
    """
    logging.info(f"Gaze processing thread started (downsampling to {GAZE_PROCESSING_WIDTH}×{GAZE_PROCESSING_HEIGHT})")
    
    while True:
        try:
            # Get latest frame
            with frame_lock:
                if not frame_buffer:
                    time.sleep(0.01)
                    continue
                frame_full, timestamp = frame_buffer[-1]
           
            # Process gaze
            start_gaze_loop(
                sender_queue, 
                frame_full, 
                timestamp, 
                face_detection_session, 
                gaze_estimation_session
            )
            
            # Send results (non-blocking)
            try:
                # Only send if queue not backed up
                if sender_queue.qsize() < SENDER_QUEUE_MAXSIZE:
                    gaze_data = sender_queue.get_nowait()
                    if gaze_data:
                        ser.write((gaze_data + "\n").encode())
            except queue.Empty:
                pass
            
            # Small yield to other threads
            time.sleep(0.001)
            
        except Exception as e:
            logging.error(f"Gaze processing error: {e}", exc_info=True)
            time.sleep(0.1)


def blink_processing_thread(ser, blink_engine):
    """
    Dedicated thread for blink detection.
    Processes EVERY frame at FULL resolution (constraint requirement).
    """
    logging.info("Blink processing thread started (every frame, 640×480)")
    
    while True:
        try:
            # Get latest frame at FULL resolution
            with frame_lock:
                if not frame_buffer:
                    time.sleep(0.01)
                    continue
                frame_full, _ = frame_buffer[-1]
            
            # Process blink at full resolution (required)
            blink_action = blink_engine.process(frame_full, pitch=0)
            
            if blink_action:
                ser.write((blink_action + "\n").encode())
            
            time.sleep(0.001)
            
        except Exception as e:
            logging.error(f"Blink processing error: {e}", exc_info=True)
            time.sleep(0.1)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    global face_detection_session, gaze_estimation_session
    
    logging.info("="*60)
    logging.info("SENDER - Raspberry Pi Zero 2W")
    logging.info("Constraints: 640×480 camera, every-frame blink detection")
    logging.info("="*60)
    
    # ---- Serial Connection ----
    logging.info("Connecting to serial port...")
    ser = None
    
    for attempt in range(10):
        ser = find_serial_port()
        if ser:
            logging.info(f"✓ Serial connected on attempt {attempt + 1}")
            break
        logging.info(f"Attempt {attempt + 1}/10: Retrying in 2s...")
        time.sleep(2)
    
    if not ser:
        logging.error("✗ Failed to connect to serial after 10 attempts")
        return
    
    # ---- Load ONNX Models ----
    logging.info("Loading ONNX models with optimizations...")
    try:
        face_detection_session = load_optimized_onnx_model("./models/face_detection.onnx")
        gaze_estimation_session = load_optimized_onnx_model("./models/gaze_estimation.onnx")
        
        logging.info("✓ All models loaded successfully")
    except Exception as e:
        logging.error(f"✗ Failed to load models: {e}")
        return
    
    # ---- Initialize Camera ----
    logging.info(f"Initializing camera at {CAMERA_WIDTH}×{CAMERA_HEIGHT}...")
    try:
        picam2 = Picamera2()
        
        config = picam2.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),  # Full res required
                "format": "RGB888"
            },
            buffer_count=2
        )
        
        picam2.configure(config)
        picam2.start()
        logging.info("✓ Camera started")
        
        # Warm-up
        time.sleep(2)
        
    except Exception as e:
        logging.error(f"✗ Camera initialization failed: {e}")
        return
    
    # ---- Initialize Blink Engine ----
    logging.info("Initializing blink detection...")
    try:
        blink_engine = BlinkEngine()
        logging.info("✓ Blink engine ready")
    except Exception as e:
        logging.error(f"✗ Blink engine failed: {e}")
        return
    
    # ---- Start Processing ----
    if USE_ASYNC_PROCESSING:
        logging.info("Starting ASYNCHRONOUS processing mode")
        logging.info("  - Camera: Captures at 640×480")
        logging.info("  - Gaze: Processes at 320×240 (downsampled)")
        logging.info("  - Blink: Processes every frame at 640×480")
        
        # Create threads
        capture_thread = threading.Thread(
            target=camera_capture_thread,
            args=(picam2,),
            daemon=True,
            name="CameraCapture"
        )
        
        gaze_thread = threading.Thread(
            target=gaze_processing_thread,
            args=(ser,),
            daemon=True,
            name="GazeProcessing"
        )
        
        blink_thread = threading.Thread(
            target=blink_processing_thread,
            args=(ser, blink_engine),
            daemon=True,
            name="BlinkProcessing"
        )
        
        # Start all threads
        capture_thread.start()
        time.sleep(0.5)  # Camera stabilize
        gaze_thread.start()
        blink_thread.start()
        
        logging.info("✓ All processing threads started")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
                # Optional: Add health monitoring here
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
    
    else:
        logging.info("Starting SYNCHRONOUS processing mode")
        camera_loop_sync(ser, picam2, blink_engine)
    
    # ---- Cleanup ----
    logging.info("Shutting down...")
    if ser:
        ser.close()
        logging.info("Serial closed")
    
    if picam2:
        picam2.stop()
        logging.info("Camera stopped")
    
    logging.info("Program terminated")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
