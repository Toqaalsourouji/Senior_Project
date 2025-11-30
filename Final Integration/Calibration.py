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
        
        # Define calibration points (15 total: 3x5 grid)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_index = 0
        self.calibration_data = {}
        
        # Setup UI
        self.root = None
        self.canvas = None
        self.status_label = None
        
    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get screen resolution accounting for DPI scaling."""
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            return w, h
        except:
            # Fallback for non-Windows systems
            root = tk.Tk()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return w, h
    
    def _generate_calibration_points(self) -> List[Dict]:
        """Generate 15 calibration points: 3x5 grid."""
        w, h = self.screen_w, self.screen_h
        
        # Calculate margins as 10% of screen dimensions
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        
        # Calculate usable area
        usable_width = w - 2 * margin_x
        usable_height = h - 2 * margin_y
        
        # Calculate spacing between points
        col_spacing = usable_width / (5 - 1)  # 5 columns
        row_spacing = usable_height / (3 - 1)  # 3 rows
        
        # Row and column names
        row_names = ["top", "middle", "bottom"]
        col_names = ["far_left", "left", "center", "right", "far_right"]
        
        points = []
        
        # Generate grid in snake pattern (left-to-right on even rows, right-to-left on odd rows)
        for row in range(3):
            if row % 2 == 0:
                # Left to right
                cols = range(5)
            else:
                # Right to left
                cols = range(4, -1, -1)
            
            for col in cols:
                # Calculate pixel position
                x = margin_x + int(col * col_spacing)
                y = margin_y + int(row * row_spacing)
                
                # Generate name
                if row == 1 and col == 2:
                    name = "center"
                else:
                    name = f"{row_names[row]}_{col_names[col]}"
                
                points.append({
                    "name": name.replace('_', ' ').title(),
                    "x": x,
                    "y": y
                })
        
        return points
    
    def _setup_ui(self):
        """Create fullscreen window with calibration point."""
        self.root = tk.Tk()
        self.root.title("Calibration")
        
        # Remove window decorations and go fullscreen
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        
        # Update to ensure window is rendered
        self.root.update_idletasks()
        
        # Canvas for drawing calibration point - fill entire screen
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg='black',
            highlightthickness=0,
            borderwidth=0
        )
        self.canvas.place(x=0, y=0, width=self.screen_w, height=self.screen_h)
        
        # Status label at top
        self.status_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 16, 'bold'),
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
        
        # Force window to front
        self.root.lift()
        self.root.focus_force()
        
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
            width=3
        )
        
        # Draw middle circle
        self.canvas.create_oval(
            x - self.point_size * 1.2,
            y - self.point_size * 1.2,
            x + self.point_size * 1.2,
            y + self.point_size * 1.2,
            outline='orange',
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
        line_length = self.point_size * 2.5
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
        
        # Draw small center dot
        self.canvas.create_oval(
            x - 3, y - 3,
            x + 3, y + 3,
            fill='white',
            outline=''
        )
        
        # Update status
        progress = f"Point {self.current_point_index + 1}/{len(self.calibration_points)}"
        self.status_label.config(text=f"{progress}: {name}")
        
        # Add coordinate display
        coord_text = f"Position: ({x}, {y})"
        self.canvas.create_text(
            self.screen_w // 2,
            self.screen_h - 100,
            text=coord_text,
            font=('Arial', 14),
            fill='white'
        )
        
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
        
        # Show summary
        summary_text = f"{len(self.calibration_data)} points calibrated successfully"
        self.canvas.create_text(
            self.screen_w // 2,
            self.screen_h // 2 + 60,
            text=summary_text,
            font=('Arial', 16),
            fill='white'
        )
        
        # Auto-close after 2 seconds - destroy window, don't quit
        self.root.after(2000, self._close_window)
    
    def _close_window(self):
        """Close the calibration window without quitting the application."""
        if self.root:
            self.root.destroy()
            self.root = None
    
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
        #if signal_callback:
        #    signal_callback(self)
        
        # Start UI loop
        self.root.mainloop()
        
        return self.calibration_data
    
    def get_calibration_data(self) -> Dict:
        """Return the collected calibration data."""
        return self.calibration_data


# Example usage functions:

def example_manual_mode():
    """Example: Manual calibration (press Space to advance)."""
    print("Starting calibration in manual mode...")
    print("Press SPACE to advance to next point, ESC to cancel")
    
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
        print("oga")
        def listen():
            while cal_display.current_point_index < len(cal_display.calibration_points):
                # Wait for signal from Pi
                pi_data = receive_signal_from_pi()  # Your function here
                
                # Notify calibrator (thread-safe UI update)
                cal_display.root.after(0, lambda: cal_display.on_pi_signal_received(pi_data))
        
        # Start listener thread
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        print("o2ga")
    data = calibrator.start_calibration(pi_signal_listener)
    return data


# Run example if executed directly
#if __name__ == "__main__":
   # example_manual_mode()