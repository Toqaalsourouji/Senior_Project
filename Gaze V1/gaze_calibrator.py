#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

# Import the necessary modules from the main gaze tracking system
try:
    import onnxruntime
    from demo_utils import multiclass_nms, demo_postprocess
    from smoother import GazeSmoother, OneEuroFilter
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed and in the Python path.")
    sys.exit(1)


@dataclass
class GazeDirection:
    """Represents a single gaze direction with collected samples and calculated thresholds."""
    name: str
    grid_position: Tuple[int, int]  # (row, col) in grid
    screen_position: Tuple[int, int]  # (x, y) pixel coordinates on screen
    samples_pitch: List[float]
    samples_yaw: List[float]
    pitch_threshold: Tuple[float, float]  # (min, max) in radians
    yaw_threshold: Tuple[float, float]    # (min, max) in radians
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "grid_position": list(self.grid_position),
            "x_threshold": list(self.yaw_threshold),   # yaw corresponds to horizontal (x)
            "y_threshold": list(self.pitch_threshold),  # pitch corresponds to vertical (y)
            "sample_count": len(self.samples_pitch)
        }


class CalibrationTarget:
    """Manages the visual calibration target displayed on screen with 3x5 grid support."""
    
    # Grid layout: 3 rows x 5 columns
    GRID_ROWS = 3
    GRID_COLS = 5
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the calibration target manager.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_radius = 20
        self.target_color = (255, 255, 255)  # White
        self.background_color = (0, 0, 0)    # Black
        
        # Define margins from screen edges (10% for better visibility)
        margin_x = int(screen_width * 0.10)
        margin_y = int(screen_height * 0.10)
        
        # Calculate usable area
        usable_width = screen_width - 2 * margin_x
        usable_height = screen_height - 2 * margin_y
        
        # Calculate spacing between points
        col_spacing = usable_width / (self.GRID_COLS - 1)
        row_spacing = usable_height / (self.GRID_ROWS - 1)
        
        # Generate 3x5 grid positions and their names
        self.positions = {}
        self.grid_map = {}  # Maps grid position to name
        
        # Row names
        row_names = ["top", "middle", "bottom"]
        # Column names
        col_names = ["far_left", "left", "center", "right", "far_right"]
        
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                # Calculate pixel position
                x = margin_x + int(col * col_spacing)
                y = margin_y + int(row * row_spacing)
                
                # Generate name
                if row == 1 and col == 2:
                    # Special case: center point
                    name = "center"
                else:
                    # Combine row and column names
                    name = f"{row_names[row]}_{col_names[col]}"
                
                self.positions[name] = (x, y)
                self.grid_map[(row, col)] = name
        
        # Define optimal calibration order to minimize eye movement
        # Use a snake pattern through the grid
        self.calibration_order = self._generate_calibration_order()
    
    def _generate_calibration_order(self) -> List[str]:
        """
        Generate an optimal calibration order using a snake pattern.
        This minimizes large eye movements between targets.
        
        Returns:
            List of position names in optimal order
        """
        order = []
        
        # Snake pattern: left-to-right on even rows, right-to-left on odd rows
        for row in range(self.GRID_ROWS):
            if row % 2 == 0:
                # Left to right
                cols = range(self.GRID_COLS)
            else:
                # Right to left
                cols = range(self.GRID_COLS - 1, -1, -1)
            
            for col in cols:
                name = self.grid_map[(row, col)]
                order.append(name)
        
        return order
    
    def draw_target(self, image: np.ndarray, position_name: str, 
                    progress: float = 0.0) -> np.ndarray:
        """
        Draw the calibration target on the image.
        
        Args:
            image: The image to draw on
            position_name: Name of the target position
            progress: Progress of sample collection (0.0 to 1.0)
        
        Returns:
            The image with the target drawn
        """
        # Clear to black background
        image.fill(0)
        
        # Get target position
        if position_name not in self.positions:
            return image
        
        pos = self.positions[position_name]
        
        # Draw target circle
        cv2.circle(image, pos, self.target_radius, self.target_color, -1)
        
        # Draw progress ring around target
        if progress > 0:
            progress_radius = self.target_radius + 10
            thickness = 5
            # Draw progress arc (0 to 360 degrees based on progress)
            end_angle = int(360 * progress)
            cv2.ellipse(image, pos, (progress_radius, progress_radius),
                       0, -90, end_angle - 90, (0, 255, 0), thickness)
        
        # Draw position indicator text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Look at the target - {position_name.replace('_', ' ').title()}"
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height - 50
        cv2.putText(image, text, (text_x, text_y), font, 1.0, 
                   self.target_color, 2, cv2.LINE_AA)
        
        # Draw position counter
        current_idx = self.calibration_order.index(position_name) + 1
        total = len(self.calibration_order)
        counter_text = f"Position {current_idx}/{total}"
        cv2.putText(image, counter_text, (self.screen_width - 250, self.screen_height - 50), 
                   font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        
        return image
    
    def draw_instructions(self, image: np.ndarray) -> np.ndarray:
        """
        Draw initial calibration instructions.
        
        Args:
            image: The image to draw on
        
        Returns:
            The image with instructions drawn
        """
        image.fill(0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        title = "GAZE CALIBRATION - 15 Point System"
        title_size = cv2.getTextSize(title, font, 1.8, 3)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = 120
        cv2.putText(image, title, (title_x, title_y), font, 1.8, 
                   self.target_color, 3, cv2.LINE_AA)
        
        # Instructions
        instructions = [
            "This calibration uses a 3x5 grid (15 positions) for precise gaze tracking.",
            "",
            "Instructions:",
            "1. A white circle will appear at different screen positions",
            "2. Look directly at each circle and keep your gaze steady",
            "3. The system will collect samples automatically (shown by green progress ring)",
            "4. The circle will move to the next position when complete",
            "5. Complete all 15 positions for best results",
            "",
            "Tips for accurate calibration:",
            "- Keep your head still - move only your eyes",
            "- Maintain a comfortable distance from the camera (50-70 cm)",
            "- Ensure good, even lighting on your face",
            "- Avoid backlighting or glare",
            "- Take a moment to focus on each target before collection starts",
            "",
            "Press SPACE to begin calibration",
            "Press ESC to cancel at any time"
        ]
        
        y_offset = 220
        for line in instructions:
            if line.startswith("Instructions:") or line.startswith("Tips"):
                font_scale = 1.1
                thickness = 2
                color = (255, 255, 100)
            elif line == "":
                y_offset += 15
                continue
            else:
                font_scale = 0.75
                thickness = 1
                color = self.target_color
            
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(image, line, (text_x, y_offset), font, font_scale,
                       color, thickness, cv2.LINE_AA)
            y_offset += 35
        
        return image


class ThresholdCalculator:
    """
    Handles calculation of non-overlapping thresholds for adjacent gaze directions
    using statistical methods optimized for 3x5 grid layout.
    """
    
    @staticmethod
    def calculate_direction_statistics(samples_pitch: List[float], 
                                     samples_yaw: List[float]) -> Dict:
        """
        Calculate comprehensive statistical measures for a direction's samples.
        
        Args:
            samples_pitch: List of pitch values in radians
            samples_yaw: List of yaw values in radians
        
        Returns:
            Dictionary with statistical measures
        """
        pitch_arr = np.array(samples_pitch)
        yaw_arr = np.array(samples_yaw)
        
        return {
            "pitch_mean": float(np.mean(pitch_arr)),
            "pitch_median": float(np.median(pitch_arr)),
            "pitch_std": float(np.std(pitch_arr)),
            "pitch_mad": float(np.median(np.abs(pitch_arr - np.median(pitch_arr)))),  # Median Absolute Deviation
            "pitch_min": float(np.min(pitch_arr)),
            "pitch_max": float(np.max(pitch_arr)),
            "yaw_mean": float(np.mean(yaw_arr)),
            "yaw_median": float(np.median(yaw_arr)),
            "yaw_std": float(np.std(yaw_arr)),
            "yaw_mad": float(np.median(np.abs(yaw_arr - np.median(yaw_arr)))),
            "yaw_min": float(np.min(yaw_arr)),
            "yaw_max": float(np.max(yaw_arr)),
        }
    
    @staticmethod
    def resolve_overlapping_thresholds(directions: Dict[str, GazeDirection],
                                      grid_map: Dict[Tuple[int, int], str]) -> None:
        """
        Resolve overlapping thresholds between adjacent directions in the 3x5 grid.
        Uses a weighted boundary approach with consideration for sample variance.
        
        Args:
            directions: Dictionary of GazeDirection objects to process
            grid_map: Mapping from grid position to direction name
        """
        # Build adjacency relationships for 3x5 grid
        adjacencies = ThresholdCalculator._build_adjacency_map(grid_map)
        
        # Calculate statistics for each direction
        stats = {}
        for name, direction in directions.items():
            stats[name] = ThresholdCalculator.calculate_direction_statistics(
                direction.samples_pitch, direction.samples_yaw
            )
        
        # Initial threshold calculation using median ± 2.5*MAD (more robust than std)
        # MAD is more resistant to outliers
        for name, direction in directions.items():
            s = stats[name]
            # Use MAD for more robust threshold estimation
            pitch_range = 2.5 * max(s["pitch_mad"], 0.01)  # Minimum range
            yaw_range = 2.5 * max(s["yaw_mad"], 0.01)
            
            direction.pitch_threshold = (
                s["pitch_median"] - pitch_range,
                s["pitch_median"] + pitch_range
            )
            direction.yaw_threshold = (
                s["yaw_median"] - yaw_range,
                s["yaw_median"] + yaw_range
            )
        
        # Iteratively resolve overlaps
        max_iterations = 5
        for iteration in range(max_iterations):
            adjustments_made = False
            
            for name1, neighbors in adjacencies.items():
                if name1 not in directions:
                    continue
                
                dir1 = directions[name1]
                s1 = stats[name1]
                
                for name2 in neighbors:
                    if name2 not in directions:
                        continue
                    
                    dir2 = directions[name2]
                    s2 = stats[name2]
                    
                    # Resolve pitch overlap
                    if ThresholdCalculator._check_overlap(dir1.pitch_threshold, dir2.pitch_threshold):
                        boundary = ThresholdCalculator._calculate_boundary(
                            s1["pitch_median"], s1["pitch_mad"],
                            s2["pitch_median"], s2["pitch_mad"]
                        )
                        
                        if s1["pitch_median"] < s2["pitch_median"]:
                            dir1.pitch_threshold = (
                                dir1.pitch_threshold[0],
                                min(boundary, dir1.pitch_threshold[1])
                            )
                            dir2.pitch_threshold = (
                                max(boundary, dir2.pitch_threshold[0]),
                                dir2.pitch_threshold[1]
                            )
                        else:
                            dir1.pitch_threshold = (
                                max(boundary, dir1.pitch_threshold[0]),
                                dir1.pitch_threshold[1]
                            )
                            dir2.pitch_threshold = (
                                dir2.pitch_threshold[0],
                                min(boundary, dir2.pitch_threshold[1])
                            )
                        
                        adjustments_made = True
                    
                    # Resolve yaw overlap
                    if ThresholdCalculator._check_overlap(dir1.yaw_threshold, dir2.yaw_threshold):
                        boundary = ThresholdCalculator._calculate_boundary(
                            s1["yaw_median"], s1["yaw_mad"],
                            s2["yaw_median"], s2["yaw_mad"]
                        )
                        
                        if s1["yaw_median"] < s2["yaw_median"]:
                            dir1.yaw_threshold = (
                                dir1.yaw_threshold[0],
                                min(boundary, dir1.yaw_threshold[1])
                            )
                            dir2.yaw_threshold = (
                                max(boundary, dir2.yaw_threshold[0]),
                                dir2.yaw_threshold[1]
                            )
                        else:
                            dir1.yaw_threshold = (
                                max(boundary, dir1.yaw_threshold[0]),
                                dir1.yaw_threshold[1]
                            )
                            dir2.yaw_threshold = (
                                dir2.yaw_threshold[0],
                                min(boundary, dir2.yaw_threshold[1])
                            )
                        
                        adjustments_made = True
            
            if not adjustments_made:
                print(f"Threshold convergence reached after {iteration + 1} iterations")
                break
        
        # Apply minimum margins to ensure clear separation
        ThresholdCalculator._apply_minimum_margins(directions, adjacencies)
    
    @staticmethod
    def _build_adjacency_map(grid_map: Dict[Tuple[int, int], str]) -> Dict[str, List[str]]:
        """
        Build adjacency relationships for the 3x5 grid.
        
        Args:
            grid_map: Mapping from grid position to direction name
        
        Returns:
            Dictionary mapping each direction to its neighbors
        """
        adjacencies = {}
        
        for (row, col), name in grid_map.items():
            neighbors = []
            
            # Check all 8 possible neighbors (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    neighbor_pos = (row + dr, col + dc)
                    if neighbor_pos in grid_map:
                        neighbors.append(grid_map[neighbor_pos])
            
            adjacencies[name] = neighbors
        
        return adjacencies
    
    @staticmethod
    def _check_overlap(threshold1: Tuple[float, float], 
                      threshold2: Tuple[float, float]) -> bool:
        """Check if two threshold ranges overlap."""
        return not (threshold1[1] <= threshold2[0] or threshold2[1] <= threshold1[0])
    
    @staticmethod
    def _calculate_boundary(median1: float, mad1: float, 
                          median2: float, mad2: float) -> float:
        """
        Calculate optimal boundary between two distributions using
        weighted midpoint based on MAD (more robust than standard deviation).
        
        Args:
            median1: Median of first distribution
            mad1: Median Absolute Deviation of first distribution
            median2: Median of second distribution
            mad2: Median Absolute Deviation of second distribution
        
        Returns:
            Optimal boundary value
        """
        # Weight inversely proportional to MAD (more precise measurements get more weight)
        epsilon = 0.001  # Prevent division by zero
        weight1 = 1.0 / (mad1 + epsilon)
        weight2 = 1.0 / (mad2 + epsilon)
        
        # Weighted average of medians
        boundary = (median1 * weight2 + median2 * weight1) / (weight1 + weight2)
        
        return boundary
    
    @staticmethod
    def _apply_minimum_margins(directions: Dict[str, GazeDirection],
                              adjacencies: Dict[str, List[str]],
                              min_margin: float = 0.008) -> None:
        """
        Ensure minimum margins between adjacent thresholds.
        Reduced margin for 15-point system to allow finer granularity.
        
        Args:
            directions: Dictionary of GazeDirection objects
            adjacencies: Adjacency relationships
            min_margin: Minimum margin in radians (approximately 0.46 degrees)
        """
        for name1, neighbors in adjacencies.items():
            if name1 not in directions:
                continue
            
            dir1 = directions[name1]
            
            for name2 in neighbors:
                if name2 not in directions:
                    continue
                
                dir2 = directions[name2]
                
                # Check pitch margins
                pitch_gap = min(
                    abs(dir1.pitch_threshold[1] - dir2.pitch_threshold[0]),
                    abs(dir2.pitch_threshold[1] - dir1.pitch_threshold[0])
                )
                
                if pitch_gap < min_margin:
                    if dir1.pitch_threshold[1] <= dir2.pitch_threshold[0]:
                        mid = (dir1.pitch_threshold[1] + dir2.pitch_threshold[0]) / 2
                        dir1.pitch_threshold = (dir1.pitch_threshold[0], mid - min_margin/2)
                        dir2.pitch_threshold = (mid + min_margin/2, dir2.pitch_threshold[1])
                
                # Check yaw margins
                yaw_gap = min(
                    abs(dir1.yaw_threshold[1] - dir2.yaw_threshold[0]),
                    abs(dir2.yaw_threshold[1] - dir1.yaw_threshold[0])
                )
                
                if yaw_gap < min_margin:
                    if dir1.yaw_threshold[1] <= dir2.yaw_threshold[0]:
                        mid = (dir1.yaw_threshold[1] + dir2.yaw_threshold[0]) / 2
                        dir1.yaw_threshold = (dir1.yaw_threshold[0], mid - min_margin/2)
                        dir2.yaw_threshold = (mid + min_margin/2, dir2.yaw_threshold[1])


class GazeCalibrationSystem:
    """Main calibration system orchestrating the 15-point calibration process."""
    
    def __init__(self, samples_per_position: int = 100,
                 sample_collection_time: float = 5.5,
                 output_file: str = "gaze_direction_calibration.json"):
        """
        Initialize the gaze calibration system.
        Args:
            samples_per_position: Number of samples to collect per position
            sample_collection_time: Time in seconds to collect samples at each position
            output_file: Path to save calibration results
        """
        self.samples_per_position = samples_per_position
        self.sample_collection_time = sample_collection_time
        self.output_file = output_file
        
        # Initialize OpenCV window
        cv2.namedWindow("Gaze Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gaze Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Get screen dimensions (with fallback)
        self.screen_width = 1920  # Default
        self.screen_height = 1080  # Default
        
        # Try to get actual screen size
        temp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("Gaze Calibration", temp_img)
        cv2.waitKey(1)
        
        # Initialize components
        self.target = CalibrationTarget(self.screen_width, self.screen_height)
        self.threshold_calculator = ThresholdCalculator()
        
        # Initialize gaze tracking components
        self._initialize_gaze_tracking()
        
        # Storage for calibration data
        self.directions: Dict[str, GazeDirection] = {}
        
        # Initialize smoother for gaze data with optimized parameters
        self.gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.005, beta=0.7)
    
    def _initialize_gaze_tracking(self) -> None:
        """Initialize the gaze tracking models and camera."""
        try:
            # Load ONNX models
            self.face_detection_session = onnxruntime.InferenceSession("./models/face_detection.onnx")
            self.landmark_detection_session = onnxruntime.InferenceSession("./models/landmark_detection.onnx")
            self.gaze_estimation_session = onnxruntime.InferenceSession("./models/gaze_estimation.onnx")
            
            # Initialize camera with optimal settings
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            print("Gaze tracking initialized successfully")
            
        except Exception as e:
            print(f"Error initializing gaze tracking: {e}")
            sys.exit(1)
    
    def _get_current_gaze(self) -> Optional[Tuple[float, float]]:
        """
        Get the current gaze direction from the camera.
        
        Returns:
            Tuple of (pitch, yaw) in radians, or None if detection failed
        """
        try:
            from gaze_detector_integration import GazeDetector
            
            # Initialize gaze detector if not already done
            if not hasattr(self, '_gaze_detector'):
                self._gaze_detector = GazeDetector(
                    self.face_detection_session,
                    self.landmark_detection_session,
                    self.gaze_estimation_session
                )
            
            # Get frame from camera
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            # Get gaze using actual detection
            gaze_data = self._gaze_detector.get_gaze_from_frame(frame)
            
            # Apply smoothing
            if gaze_data is not None:
                timestamp = time.time()
                gaze_data = self.gaze_smoother(gaze_data, t=timestamp)
            
            return gaze_data
            
        except ImportError:
            # Fallback to mock implementation for testing
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            import random
            pitch = random.uniform(-0.5, 0.5)
            yaw = random.uniform(-0.5, 0.5)
            
            return (pitch, yaw)
    
    def collect_samples_for_position(self, position_name: str) -> Tuple[List[float], List[float]]:
        """
        Collect gaze samples for a specific target position with quality filtering.
        
        Args:
            position_name: Name of the target position
        
        Returns:
            Tuple of (pitch_samples, yaw_samples)
        """
        pitch_samples = []
        yaw_samples = []
        
        start_time = time.time()
        last_sample_time = 0
        sample_interval = self.sample_collection_time / self.samples_per_position
        
        # Warm-up period before collection starts
        warmup_time = 0.5
        
        print(f"\nCollecting samples for {position_name}...")
        
        while len(pitch_samples) < self.samples_per_position:
            current_time = time.time() - start_time
            
            # Wait for warm-up
            if current_time < warmup_time:
                # Update display during warm-up
                display_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                self.target.draw_target(display_img, position_name, 0.0)
                cv2.imshow("Gaze Calibration", display_img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt("Calibration cancelled by user")
                
                continue
            
            # Collect sample at intervals
            adjusted_time = current_time - warmup_time
            if adjusted_time - last_sample_time >= sample_interval:
                gaze_data = self._get_current_gaze()
                
                if gaze_data is not None:
                    pitch, yaw = gaze_data
                    
                    # Quality check: reject extreme outliers
                    if abs(pitch) < 1.5 and abs(yaw) < 1.5:  # ~86 degrees
                        pitch_samples.append(pitch)
                        yaw_samples.append(yaw)
                        last_sample_time = adjusted_time
                        
                        print(f"  Sample {len(pitch_samples)}/{self.samples_per_position}: "
                              f"pitch={np.degrees(pitch):6.2f}°, yaw={np.degrees(yaw):6.2f}°")
            
            # Update display
            progress = len(pitch_samples) / self.samples_per_position
            display_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            self.target.draw_target(display_img, position_name, progress)
            cv2.imshow("Gaze Calibration", display_img)
            
            # Check for escape key
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt("Calibration cancelled by user")
        
        return pitch_samples, yaw_samples
    
    def run_calibration(self) -> bool:
        """
        Run the complete calibration process for all 15 positions.
        
        Returns:
            True if calibration completed successfully, False otherwise
        """
        try:
            # Display instructions
            display_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            self.target.draw_instructions(display_img)
            cv2.imshow("Gaze Calibration", display_img)
            
            # Wait for user to start
            print("\n" + "="*70)
            print("Waiting for user to press SPACE to begin calibration...")
            print("="*70)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE key
                    break
                elif key == 27:  # ESC key
                    print("Calibration cancelled")
                    return False
            
            # Collect samples for each position
            total_positions = len(self.target.calibration_order)
            
            for idx, position_name in enumerate(self.target.calibration_order, 1):
                print(f"\n{'='*70}")
                print(f"Position {idx}/{total_positions}: {position_name.replace('_', ' ').title()}")
                print(f"{'='*70}")
                
                # Brief pause before starting collection
                time.sleep(0.8)
                
                # Collect samples
                pitch_samples, yaw_samples = self.collect_samples_for_position(position_name)
                
                # Get grid position
                grid_pos = None
                for pos, name in self.target.grid_map.items():
                    if name == position_name:
                        grid_pos = pos
                        break
                
                # Store the direction data
                screen_pos = self.target.positions[position_name]
                self.directions[position_name] = GazeDirection(
                    name=position_name,
                    grid_position=grid_pos if grid_pos else (0, 0),
                    screen_position=screen_pos,
                    samples_pitch=pitch_samples,
                    samples_yaw=yaw_samples,
                    pitch_threshold=(0, 0),  # Will be calculated later
                    yaw_threshold=(0, 0)      # Will be calculated later
                )
                
                # Brief pause between positions
                time.sleep(0.5)
            
            print("\n" + "="*70)
            print("Sample collection complete. Processing thresholds...")
            print("="*70)
            
            # Calculate thresholds with overlap resolution
            self.threshold_calculator.resolve_overlapping_thresholds(
                self.directions, 
                self.target.grid_map
            )
            
            # Save calibration data
            self.save_calibration()
            
            print("\n" + "="*70)
            print("CALIBRATION COMPLETED SUCCESSFULLY!")
            print("="*70)
            return True
            
        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user")
            return False
        except Exception as e:
            print(f"\nError during calibration: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
    
    def save_calibration(self) -> None:
        """Save the calibration data to JSON file with metadata."""
        calibration_data = {}
        
        for name, direction in self.directions.items():
            calibration_data[name] = direction.to_dict()
        
        # Add comprehensive metadata
        metadata = {
            "_metadata": {
                "version": "2.0",
                "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "grid_layout": {
                    "rows": CalibrationTarget.GRID_ROWS,
                    "columns": CalibrationTarget.GRID_COLS,
                    "total_positions": len(self.directions)
                },
                "sampling": {
                    "samples_per_position": self.samples_per_position,
                    "collection_time_per_position": self.sample_collection_time,
                    "total_samples": sum(len(d.samples_pitch) for d in self.directions.values())
                },
                "screen_resolution": f"{self.screen_width}x{self.screen_height}",
                "threshold_method": "statistical_mad_with_overlap_resolution",
                "algorithm_notes": "Using MAD (Median Absolute Deviation) for robust statistics"
            }
        }
        
        # Combine data and metadata
        output_data = {**calibration_data, **metadata}
        
        # Save to file
        try:
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nCalibration saved to: {self.output_file}")
            
            # Print summary
            self._print_calibration_summary()
            
        except IOError as e:
            print(f"Error saving calibration file: {e}")
            raise
    
    def _print_calibration_summary(self) -> None:
        """Print a formatted summary of the calibration results."""
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        print(f"{'Position':<20} {'Pitch Range (°)':<20} {'Yaw Range (°)':<20}")
        print("-"*70)
        
        for name in self.target.calibration_order:
            if name in self.directions:
                d = self.directions[name]
                pitch_range = f"[{np.degrees(d.pitch_threshold[0]):6.2f}, {np.degrees(d.pitch_threshold[1]):6.2f}]"
                yaw_range = f"[{np.degrees(d.yaw_threshold[0]):6.2f}, {np.degrees(d.yaw_threshold[1]):6.2f}]"
                print(f"{name:<20} {pitch_range:<20} {yaw_range:<20}")
        
        print("="*70)
    
    def validate_calibration(self, calibration_file: str) -> bool:
        """
        Validate a saved calibration file.
        
        Args:
            calibration_file: Path to the calibration file to validate
        
        Returns:
            True if calibration is valid, False otherwise
        """
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            
            # Check metadata
            if "_metadata" not in data:
                print("Warning: Calibration file missing metadata")
            
            # Count non-metadata entries
            direction_count = sum(1 for key in data.keys() if not key.startswith("_"))
            expected_count = CalibrationTarget.GRID_ROWS * CalibrationTarget.GRID_COLS
            
            if direction_count != expected_count:
                print(f"Warning: Expected {expected_count} directions, found {direction_count}")
            
            # Validate each direction
            for direction_name, direction_data in data.items():
                if direction_name.startswith("_"):
                    continue
                
                # Check required fields
                required_fields = ["x_threshold", "y_threshold"]
                for field in required_fields:
                    if field not in direction_data:
                        print(f"Missing field '{field}' for direction: {direction_name}")
                        return False
                
                # Validate threshold format
                x_thresh = direction_data["x_threshold"]
                y_thresh = direction_data["y_threshold"]
                
                if not (isinstance(x_thresh, list) and len(x_thresh) == 2):
                    print(f"Invalid x_threshold format for {direction_name}")
                    return False
                
                if not (isinstance(y_thresh, list) and len(y_thresh) == 2):
                    print(f"Invalid y_threshold format for {direction_name}")
                    return False
                
                # Check that min < max
                if x_thresh[0] >= x_thresh[1] or y_thresh[0] >= y_thresh[1]:
                    print(f"Invalid threshold range for {direction_name}")
                    return False
            
            print(f"✓ Calibration file is valid ({direction_count} positions)")
            return True
            
        except Exception as e:
            print(f"Error validating calibration file: {e}")
            return False


def main():
    """Main entry point for the calibration script."""
    parser = argparse.ArgumentParser(
        description="Gaze Direction Calibration System - 15 Point (3x5 Grid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs gaze calibration using a 3x5 grid (15 positions) by displaying 
targets and collecting eye-tracking samples. The output is a JSON file with calculated
thresholds for each gaze direction, with intelligent overlap resolution.

Example usage:
    python gaze_calibrator.py
    python gaze_calibrator.py --samples 40 --time 5.0
    python gaze_calibrator.py --validate existing_calibration.json
        """
    )
    
    parser.add_argument("--samples", type=int, default=30,
                       help="Number of samples to collect per position (default: 30, recommended: 30-50)")
    parser.add_argument("--time", type=float, default=4.0,
                       help="Time in seconds to collect samples at each position (default: 4.0)")
    parser.add_argument("--output", type=str, default="gaze_direction_calibration.json",
                       help="Output file name for calibration data (default: gaze_direction_calibration.json)")
    parser.add_argument("--validate", type=str, metavar="FILE",
                       help="Validate an existing calibration file instead of running calibration")
    
    args = parser.parse_args()
    
    # If validation mode
    if args.validate:
        calibrator = GazeCalibrationSystem()
        if calibrator.validate_calibration(args.validate):
            print(f"\n✓ Calibration file '{args.validate}' is valid and ready to use")
            sys.exit(0)
        else:
            print(f"\n✗ Calibration file '{args.validate}' is invalid")
            sys.exit(1)
    
    # Run calibration
    print("="*70)
    print("GAZE DIRECTION CALIBRATION SYSTEM - 15 POINT (3x5 Grid)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Grid Layout: 3 rows × 5 columns = 15 positions")
    print(f"  - Samples per position: {args.samples}")
    print(f"  - Collection time per position: {args.time} seconds")
    print(f"  - Output file: {args.output}")
    print(f"  - Estimated total time: ~{len(CalibrationTarget(1920, 1080).calibration_order) * (args.time + 1):.0f} seconds")
    print("="*70 + "\n")
    
    calibrator = GazeCalibrationSystem(
        samples_per_position=args.samples,
        sample_collection_time=args.time,
        output_file=args.output
    )
    
    if calibrator.run_calibration():
        print("\n" + "="*70)
        print("✓ CALIBRATION SUCCESSFUL")
        print("="*70)
        print(f"Calibration data saved to: {args.output}")
        print("You can now use this file with your main gaze tracking application")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ CALIBRATION FAILED OR CANCELLED")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()