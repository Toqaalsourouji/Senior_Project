import tkinter as tk
import ctypes
from typing import Callable, Tuple, List, Dict

class CalibrationDisplay:
    """
    Calibration module that displays points on screen for camera/sensor calibration.
    Waits for Pi signal before moving to next point.
    """
    
    def __init__(self, point_size: int = 30, margin: int = 50):
        """
        Args:
            point_size: Size of calibration point in pixels
            margin: Distance from screen edges for corner points in pixels
        """
        self.point_size = point_size
        self.margin = margin
        
        # Get screen resolution
        self.screen_w, self.screen_h = self._get_screen_resolution()
        
        # Define calibration points (8 total: 4 corners + 4 edges + 1 center)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_index = 0
        self.calibration_data = {}
        
        # Setup UI
        self.root = None
        self.canvas = None
        self.status_label = None
        
    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get screen resolution accounting for DPI scaling."""
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        return w, h
    
    def _generate_calibration_points(self) -> List[Dict]:
        """Generate 9 calibration points: 4 corners, 4 edges, 1 center."""
        w, h = self.screen_w, self.screen_h
        m = self.margin
        
        points = [
            # Corners
            {"name": "Top-Left Corner", "x": m, "y": m},
            {"name": "Top-Right Corner", "x": w - m, "y": m},
            {"name": "Bottom-Right Corner", "x": w - m, "y": h - m},
            {"name": "Bottom-Left Corner", "x": m, "y": h - m},
            
            # Edges
            {"name": "Top-Center", "x": w // 2, "y": m},
            {"name": "Right-Center", "x": w - m, "y": h // 2},
            {"name": "Bottom-Center", "x": w // 2, "y": h - m},
            {"name": "Left-Center", "x": m, "y": h // 2},
            
            # Center
            {"name": "Center", "x": w // 2, "y": h // 2},
        ]
        
        return points
    
    def _setup_ui(self):
        """Create fullscreen window with calibration point."""
        self.root = tk.Tk()
        self.root.title("Calibration")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        
        # Canvas for drawing calibration point
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Status label at top
        self.status_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 16),
            fg='white',
            bg='black'
        )
        self.status_label.place(x=self.screen_w // 2, y=30, anchor='center')
        
        # Instructions at bottom
        instructions = tk.Label(
            self.root,
            text="Point your Pi camera/sensor at the highlighted point\nPress ESC to cancel",
            font=('Arial', 12),
            fg='gray',
            bg='black'
        )
        instructions.place(x=self.screen_w // 2, y=self.screen_h - 50, anchor='center')
        
        # ESC to exit
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
    def _draw_calibration_point(self, x: int, y: int, name: str):
        """Draw a calibration point on the canvas."""
        self.canvas.delete('all')  # Clear previous point
        
        # Draw outer circle (pulsing effect)
        self.canvas.create_oval(
            x - self.point_size * 1.5,
            y - self.point_size * 1.5,
            x + self.point_size * 1.5,
            y + self.point_size * 1.5,
            outline='red',
            width=2
        )
        
        # Draw inner filled circle
        self.canvas.create_oval(
            x - self.point_size,
            y - self.point_size,
            x + self.point_size,
            y + self.point_size,
            fill='red',
            outline='white',
            width=3
        )
        
        # Draw crosshair
        line_length = self.point_size * 2
        self.canvas.create_line(
            x - line_length, y,
            x + line_length, y,
            fill='white',
            width=2
        )
        self.canvas.create_line(
            x, y - line_length,
            x, y + line_length,
            fill='white',
            width=2
        )
        
        # Update status
        progress = f"Point {self.current_point_index + 1}/{len(self.calibration_points)}"
        self.status_label.config(text=f"{progress}: {name}")
        
    def _show_current_point(self):
        """Display the current calibration point."""
        if self.current_point_index < len(self.calibration_points):
            point = self.calibration_points[self.current_point_index]
            self._draw_calibration_point(point['x'], point['y'], point['name'])
        else:
            self._show_completion()
    
    def _show_completion(self):
        """Show calibration complete message."""
        self.canvas.delete('all')
        self.canvas.create_text(
            self.screen_w // 2,
            self.screen_h // 2,
            text="Calibration Complete!",
            font=('Arial', 36, 'bold'),
            fill='green'
        )
        self.status_label.config(text="Calibration data saved")
        # Auto-close after 2 seconds
        self.root.after(2000, self.root.quit)
    
    def on_pi_signal_received(self, pi_data: Dict = None):
        """
        Called when Pi sends signal that it's done reading current point.
        
        Args:
            pi_data: Optional data from Pi (e.g., sensor readings, coordinates)
        """
        current_point = self.calibration_points[self.current_point_index]
        
        # Store calibration data
        self.calibration_data[current_point['name']] = {
            'screen_x': current_point['x'],
            'screen_y': current_point['y'],
            'pi_data': pi_data
        }
        
        print(f"✓ Calibration point '{current_point['name']}' recorded")
        if pi_data:
            print(f"  Pi data: {pi_data}")
        
        # Move to next point
        self.current_point_index += 1
        
        # Update display
        if self.current_point_index < len(self.calibration_points):
            self.root.after(500, self._show_current_point)  # Small delay before next point
        else:
            self._show_completion()
    
    def start_calibration(self, signal_callback: Callable = None):
        """
        Start the calibration process.
        
        Args:
            signal_callback: Function to call to wait for Pi signal.
                            Should call on_pi_signal_received() when signal arrives.
        """
        self._setup_ui()
        self._show_current_point()
        
        # If callback provided, set it up to listen for Pi signals
        if signal_callback:
            signal_callback(self)
        
        # Start UI loop
        self.root.mainloop()
        
        return self.calibration_data
    
    def get_calibration_data(self) -> Dict:
        """Return the collected calibration data."""
        return self.calibration_data


# Example usage functions:

def example_manual_mode():
    """Example: Manual calibration (press Space to advance)."""
    calibrator = CalibrationDisplay(point_size=25, margin=50)
    
    def setup_keyboard_listener(cal_display):
        """Use keyboard space as stand-in for Pi signal."""
        def on_key(event):
            if event.keysym == 'space':
                cal_display.on_pi_signal_received()
        
        cal_display.root.bind('<space>', on_key)
    
    data = calibrator.start_calibration(setup_keyboard_listener)
    print("\nCalibration Data:")
    for point_name, point_data in data.items():
        print(f"{point_name}: {point_data}")


def example_with_pi_listener(receive_signal_from_pi):
    """
    Example: Integration with your Pi listener.
    
    Args:
        receive_signal_from_pi: Your function that listens for Pi signals
    """
    calibrator = CalibrationDisplay(point_size=25, margin=50)
    
    def pi_signal_listener(cal_display):
        """Listen for Pi signals in separate thread."""
        import threading
        
        def listen():
            while cal_display.current_point_index < len(cal_display.calibration_points):
                # Wait for signal from Pi
                pi_data = receive_signal_from_pi()  # Your function here
                
                # Notify calibrator (thread-safe UI update)
                cal_display.root.after(0, lambda: cal_display.on_pi_signal_received(pi_data))
        
        # Start listener thread
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
    
    data = calibrator.start_calibration(pi_signal_listener)
    return data


if __name__ == "__main__":
    # Run manual mode for testing
    print("Starting calibration (press SPACE to move to next point, ESC to cancel)")
    example_manual_mode()