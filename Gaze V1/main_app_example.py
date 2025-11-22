#!/usr/bin/env python3

import json
import os
from typing import Dict, Optional, Tuple
import numpy as np


class GazeDirectionInterpreter:
    """
    Interprets raw gaze data into discrete directions using calibration data.
    This class is used by the main application to convert continuous gaze
    values into discrete directional commands.
    """
    
    def __init__(self, calibration_file: str = "gaze_direction_calibration.json"):
        """
        Initialize the interpreter with calibration data.
        
        Args:
            calibration_file: Path to the calibration JSON file
        """
        self.calibration_file = calibration_file
        self.calibration_data = None
        self.is_loaded = False
        
        # Load calibration data
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """
        Load calibration data from JSON file.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(self.calibration_file):
            print(f"Calibration file not found: {self.calibration_file}")
            print("Please run gaze_calibrator.py first to generate calibration data")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Remove metadata if present
            if "_metadata" in self.calibration_data:
                metadata = self.calibration_data.pop("_metadata")
                print(f"Loaded calibration from: {metadata.get('calibration_date', 'unknown')}")
                print(f"Screen resolution: {metadata.get('screen_resolution', 'unknown')}")
            
            self.is_loaded = True
            print(f"Successfully loaded calibration with {len(self.calibration_data)} directions")
            return True
            
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            return False
    
    def interpret_gaze(self, pitch: float, yaw: float) -> Optional[str]:
        """
        Interpret raw gaze values into a discrete direction.
        
        Args:
            pitch: Pitch value in radians (vertical component)
            yaw: Yaw value in radians (horizontal component)
        
        Returns:
            Direction name (e.g., "center", "up", "left") or None if no match
        """
        if not self.is_loaded:
            return None
        
        # Check each direction's thresholds
        for direction_name, thresholds in self.calibration_data.items():
            x_thresh = thresholds.get("x_threshold", [])
            y_thresh = thresholds.get("y_threshold", [])
            
            if len(x_thresh) != 2 or len(y_thresh) != 2:
                continue
            
            # Check if gaze falls within this direction's thresholds
            # Note: yaw corresponds to x (horizontal), pitch to y (vertical)
            if (x_thresh[0] <= yaw <= x_thresh[1] and
                y_thresh[0] <= pitch <= y_thresh[1]):
                return direction_name
        
        return None  # No matching direction found
    
    def get_direction_confidence(self, pitch: float, yaw: float, 
                                direction: str) -> float:
        """
        Calculate confidence score for a specific direction.
        
        Args:
            pitch: Pitch value in radians
            yaw: Yaw value in radians
            direction: Direction to check
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.is_loaded or direction not in self.calibration_data:
            return 0.0
        
        thresholds = self.calibration_data[direction]
        x_thresh = thresholds.get("x_threshold", [])
        y_thresh = thresholds.get("y_threshold", [])
        
        if len(x_thresh) != 2 or len(y_thresh) != 2:
            return 0.0
        
        # Calculate distance from center of threshold region
        x_center = (x_thresh[0] + x_thresh[1]) / 2
        y_center = (y_thresh[0] + y_thresh[1]) / 2
        x_range = (x_thresh[1] - x_thresh[0]) / 2
        y_range = (y_thresh[1] - y_thresh[0]) / 2
        
        # Normalize distance
        if x_range > 0 and y_range > 0:
            x_dist = abs(yaw - x_center) / x_range
            y_dist = abs(pitch - y_center) / y_range
            
            # Confidence decreases with distance from center
            # Using Euclidean distance in normalized space
            normalized_dist = np.sqrt(x_dist**2 + y_dist**2)
            confidence = max(0.0, 1.0 - normalized_dist)
            
            return confidence
        
        return 0.0
    
    def get_all_directions_confidence(self, pitch: float, yaw: float) -> Dict[str, float]:
        """
        Get confidence scores for all directions.
        
        Args:
            pitch: Pitch value in radians
            yaw: Yaw value in radians
        
        Returns:
            Dictionary mapping direction names to confidence scores
        """
        if not self.is_loaded:
            return {}
        
        confidences = {}
        for direction in self.calibration_data.keys():
            confidences[direction] = self.get_direction_confidence(pitch, yaw, direction)
        
        return confidences
    
    def print_calibration_summary(self):
        """Print a summary of the loaded calibration data."""
        if not self.is_loaded:
            print("No calibration data loaded")
            return
        
        print("\n" + "="*60)
        print("CALIBRATION DATA SUMMARY")
        print("="*60)
        
        for direction in sorted(self.calibration_data.keys()):
            thresholds = self.calibration_data[direction]
            x_thresh = thresholds.get("x_threshold", [])
            y_thresh = thresholds.get("y_threshold", [])
            
            if len(x_thresh) == 2 and len(y_thresh) == 2:
                print(f"{direction:12} -> "
                      f"Yaw: [{np.degrees(x_thresh[0]):6.2f}°, {np.degrees(x_thresh[1]):6.2f}°] "
                      f"Pitch: [{np.degrees(y_thresh[0]):6.2f}°, {np.degrees(y_thresh[1]):6.2f}°]")
        print("="*60 + "\n")


def demo_usage():
    """
    Demonstrate how to use the GazeDirectionInterpreter in a main application.
    """
    print("="*60)
    print("GAZE DIRECTION INTERPRETER DEMO")
    print("="*60)
    
    # Initialize interpreter with calibration data
    interpreter = GazeDirectionInterpreter("gaze_direction_calibration.json")
    
    if not interpreter.is_loaded:
        print("Please run: python gaze_calibrator.py")
        return
    
    # Print calibration summary
    interpreter.print_calibration_summary()
    
    # Simulate some gaze data and interpret directions
    print("\n" + "="*60)
    print("SIMULATED GAZE INTERPRETATION")
    print("="*60)
    
    # Test cases: (pitch, yaw) in radians
    test_cases = [
        (0.0, 0.0),      # Should be center
        (0.3, 0.0),      # Should be up
        (-0.3, 0.0),     # Should be down
        (0.0, 0.3),      # Should be right
        (0.0, -0.3),     # Should be left
        (0.2, 0.2),      # Should be up-right
        (0.2, -0.2),     # Should be up-left
        (-0.2, 0.2),     # Should be down-right
        (-0.2, -0.2),    # Should be down-left
    ]
    
    for pitch, yaw in test_cases:
        # Get interpreted direction
        direction = interpreter.interpret_gaze(pitch, yaw)
        
        # Get confidence scores
        confidences = interpreter.get_all_directions_confidence(pitch, yaw)
        
        # Find top confidence
        if confidences:
            top_direction = max(confidences, key=confidences.get)
            top_confidence = confidences[top_direction]
        else:
            top_direction = "unknown"
            top_confidence = 0.0
        
        print(f"\nGaze: pitch={np.degrees(pitch):6.2f}°, yaw={np.degrees(yaw):6.2f}°")
        print(f"  Interpreted: {direction or 'NONE'}")
        print(f"  Top confidence: {top_direction} ({top_confidence:.2%})")
        
        # Show top 3 confidence scores
        if confidences:
            sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
            print("  Top 3 directions:")
            for dir_name, conf in sorted_conf:
                print(f"    - {dir_name:12} {conf:.2%}")
    
    print("\n" + "="*60)


def example_integration_with_live_tracking():
    """
    Example of how to integrate with live gaze tracking.
    This would be part of your main application.
    """
    print("""
    Example integration code for main application:
    
    ```python
    # In your main gaze tracking loop:
    
    # Initialize interpreter once
    gaze_interpreter = GazeDirectionInterpreter()
    
    # In your main loop:
    while True:
        # Get gaze from your tracking system
        pitch, yaw = get_current_gaze()  # Your existing function
        
        # Interpret direction
        direction = gaze_interpreter.interpret_gaze(pitch, yaw)
        
        # Use the direction for your application logic
        if direction == "up":
            # Scroll up, move cursor up, etc.
            perform_up_action()
        elif direction == "down":
            # Scroll down, move cursor down, etc.
            perform_down_action()
        elif direction == "left":
            # Navigate left
            perform_left_action()
        # ... etc for other directions
        
        # Optional: Use confidence scores for smoother control
        confidences = gaze_interpreter.get_all_directions_confidence(pitch, yaw)
        # Use confidences for weighted actions or threshold-based activation
    ```
    """)


if __name__ == "__main__":
    # Run the demo
    demo_usage()
    
    # Show integration example
    print("\n" + "="*60)
    example_integration_with_live_tracking()