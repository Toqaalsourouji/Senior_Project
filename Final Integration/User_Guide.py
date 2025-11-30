import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

class SightSyncWelcome:
    """
    Modern, professional welcome screen for SightSync
    Following Material Design and modern UI/UX principles
    """
    
    def __init__(self, timeout=None):
        self.timeout = timeout
        # Color palette - Calm blue with vibrant accents
        self.colors = {
            'bg_gradient_top': '#F0F4F8',
            'bg_gradient_bottom': '#E1E8ED',
            'primary': '#2D7DD2',           # Professional blue
            'primary_dark': '#1E5A9A',      # Darker blue for depth
            'accent': '#4ECDC4',            # Calm turquoise
            'success': '#51CF66',           # Fresh green (unified)
            'text_dark': '#2C3E50',         # Near black
            'text_medium': '#5F6C7B',       # Medium gray
            'text_light': '#8C98A4',        # Light gray
            'card_white': '#FFFFFF',        # Pure white
            'card_subtle': '#F8FAFB',       # Subtle gray
            'shadow': '#B8C5D0'             # Soft shadow
        }
        
        # Create main window
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)

        self.root.title("SightSync Welcome")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        self.root.configure(bg=self.colors['bg_gradient_top'])
        
        # Custom fonts
        self.font_title = tkfont.Font(family="Segoe UI", size=42, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=16)
        self.font_heading = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.font_body = tkfont.Font(family="Segoe UI", size=11)
        self.font_small = tkfont.Font(family="Segoe UI", size=10)
        self.font_button = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        
        # Build UI
        self.create_ui()
        
    def create_gradient_frame(self, parent, height):
        """Create a frame with gradient-like appearance"""
        frame = tk.Frame(parent, bg=self.colors['bg_gradient_top'], height=height)
        frame.pack(fill=tk.X)
        frame.pack_propagate(False)
        return frame
    
    def create_card(self, parent, **kwargs):
        """Create a modern card with shadow effect"""
        # Shadow layer
        shadow = tk.Frame(
            parent,
            bg=self.colors['shadow'],
            **kwargs
        )
        
        # Main card
        card = tk.Frame(
            parent,
            bg=self.colors['card_white'],
            highlightthickness=0,
            **kwargs
        )
        
        return card
    
    def create_eye_icon(self, parent, closed=False):
        """Create eye icon using text/unicode"""
        if closed:
            return tk.Label(parent, text="◡", font=("Arial", 16), 
                          fg=self.colors['text_medium'], bg=parent['bg'])
        else:
            return tk.Label(parent, text="◉", font=("Arial", 14), 
                          fg=self.colors['text_medium'], bg=parent['bg'])
    
    def create_ui(self):
        """Build the complete UI"""
        
        # Main container with gradient background
        main_container = tk.Frame(self.root, bg=self.colors['bg_gradient_top'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # ===== HEADER SECTION =====
        header_frame = tk.Frame(main_container, bg=self.colors['bg_gradient_top'])
        header_frame.pack(pady=(0, 20))
        
        # Optional: Load eye image if available
        try:
            eye_img = Image.open("eye.png")
            eye_img = eye_img.resize((100, 100))
            self.eye_icon_top = ImageTk.PhotoImage(eye_img)
            
            eye_label = tk.Label(
                header_frame,
                image=self.eye_icon_top,
                bg=self.colors['bg_gradient_top']
            )
            eye_label.pack(pady=(0, 10))
        except:
            # Skip if image not found
            pass
        
        # Eye icons row
        eye_row = tk.Frame(header_frame, bg=self.colors['bg_gradient_top'])
        eye_row.pack()
        
        # Left eye
        left_eye = tk.Label(eye_row, text="◉", font=("Arial", 20), 
                           fg=self.colors['accent'], bg=self.colors['bg_gradient_top'])
        left_eye.pack(side=tk.LEFT, padx=15)
        
        # Title
        title = tk.Label(
            eye_row,
            text="SightSync",
            font=self.font_title,
            fg=self.colors['primary'],
            bg=self.colors['bg_gradient_top']
        )
        title.pack(side=tk.LEFT)
        
        # Right eye
        right_eye = tk.Label(eye_row, text="◉", font=("Arial", 20), 
                            fg=self.colors['accent'], bg=self.colors['bg_gradient_top'])
        right_eye.pack(side=tk.LEFT, padx=15)
        
        # Subtitle
        subtitle = tk.Label(
            header_frame,
            text="Eye-Controlled Interface",
            font=self.font_subtitle,
            fg=self.colors['text_medium'],
            bg=self.colors['bg_gradient_top']
        )
        subtitle.pack(pady=(5, 0))
        
        # ===== INFO CARD =====
        info_card = self.create_card(main_container)
        info_card.pack(fill=tk.X, pady=(0, 25))
        
        info_text = tk.Label(
            info_card,
            text="Move your gaze to control the cursor",
            font=self.font_body,
            fg=self.colors['text_dark'],
            bg=self.colors['card_white'],
            pady=15
        )
        info_text.pack()
        
        # ===== GESTURE CONTROLS SECTION =====
        gesture_header = tk.Frame(main_container, bg=self.colors['bg_gradient_top'])
        gesture_header.pack(fill=tk.X, pady=(0, 15))
        
        gesture_title = tk.Label(
            gesture_header,
            text="GESTURE CONTROLS",
            font=self.font_heading,
            fg=self.colors['primary'],
            bg=self.colors['bg_gradient_top'],
            anchor=tk.W
        )
        gesture_title.pack(side=tk.LEFT)
        
        gesture_subtitle = tk.Label(
            gesture_header,
            text="Hold each gesture for 2 seconds",
            font=self.font_small,
            fg=self.colors['text_light'],
            bg=self.colors['bg_gradient_top']
        )
        gesture_subtitle.pack(side=tk.RIGHT)
        
        # ===== GESTURE CARDS =====
        gestures = [
            ("Double Blink", "Left Click", "◡ ◡"),
            ("Right Eye Closed", "Right Click", "◉ ◡"),
            ("Left Eye Closed", "Open Keyboard", "◡ ◉"),
            ("Raise Eyebrows", "Scroll Mode", "◉̲ ◉̲"),
            ("Open Mouth Wide", "Pause Mouse", "◠"),
        ]
        
        for i, (gesture, action, icon) in enumerate(gestures):
            # Alternate card colors
            card_bg = self.colors['card_white'] if i % 2 == 0 else self.colors['card_subtle']
            
            card = tk.Frame(
                main_container,
                bg=card_bg,
                highlightbackground=self.colors['shadow'],
                highlightthickness=1
            )
            card.pack(fill=tk.X, pady=4)
            
            # Card content container
            card_content = tk.Frame(card, bg=card_bg)
            card_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
            
            # Gesture name
            gesture_label = tk.Label(
                card_content,
                text=gesture,
                font=self.font_body,
                fg=self.colors['text_dark'],
                bg=card_bg,
                width=20,
                anchor=tk.W
            )
            gesture_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Arrow
            arrow_label = tk.Label(
                card_content,
                text="→",
                font=("Arial", 16),
                fg=self.colors['success'],
                bg=card_bg
            )
            arrow_label.pack(side=tk.LEFT, padx=10)
            
            # Action
            action_label = tk.Label(
                card_content,
                text=action,
                font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
                fg=self.colors['success'],
                bg=card_bg,
                width=15,
                anchor=tk.W
            )
            action_label.pack(side=tk.LEFT, padx=10)
            
            # Icon
            icon_label = tk.Label(
                card_content,
                text=icon,
                font=("Arial", 14),
                fg=self.colors['text_medium'],
                bg=card_bg
            )
            icon_label.pack(side=tk.RIGHT, padx=10)
        
        # ===== START BUTTON =====
        button_frame = tk.Frame(main_container, bg=self.colors['bg_gradient_top'])
        button_frame.pack(pady=(25, 0))
        
        # Shadow effect for button
        button_shadow = tk.Frame(
            button_frame,
            bg=self.colors['shadow'],
            height=52,
            width=402
        )
        button_shadow.place(x=3, y=3)
        
        # Main button
        start_button = tk.Frame(
            button_frame,
            bg=self.colors['primary'],
            height=50,
            width=400,
            cursor="hand2"
        )
        start_button.pack()
        start_button.pack_propagate(False)
        
        # Button content
        button_content = tk.Frame(start_button, bg=self.colors['primary'])
        button_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Left blink icon
        left_blink = tk.Label(
            button_content,
            text="◡",
            font=("Arial", 14),
            fg='white',
            bg=self.colors['primary']
        )
        left_blink.pack(side=tk.LEFT, padx=8)
        
        # Button text
        button_text = tk.Label(
            button_content,
            text="DOUBLE BLINK TO START",
            font=self.font_button,
            fg='white',
            bg=self.colors['primary']
        )
        button_text.pack(side=tk.LEFT)
        
        # Right blink icon
        right_blink = tk.Label(
            button_content,
            text="◡",
            font=("Arial", 14),
            fg='white',
            bg=self.colors['primary']
        )
        right_blink.pack(side=tk.LEFT, padx=8)
        
        # Hover effects
        def on_enter(e):
            start_button.config(bg=self.colors['primary_dark'])
            button_content.config(bg=self.colors['primary_dark'])
            button_text.config(bg=self.colors['primary_dark'])
            left_blink.config(bg=self.colors['primary_dark'])
            right_blink.config(bg=self.colors['primary_dark'])
        
        def on_leave(e):
            start_button.config(bg=self.colors['primary'])
            button_content.config(bg=self.colors['primary'])
            button_text.config(bg=self.colors['primary'])
            left_blink.config(bg=self.colors['primary'])
            right_blink.config(bg=self.colors['primary'])
        
        def on_click(e):
            print("\n" + "="*60)
            print("✓ Starting SightSync...")
            print("="*60 + "\n")
            self.root.destroy()
        
        start_button.bind("<Enter>", on_enter)
        start_button.bind("<Leave>", on_leave)
        start_button.bind("<Button-1>", on_click)
        button_text.bind("<Button-1>", on_click)
        
    def run(self):
        """Start the application"""
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"+{x}+{y}")

        print("\n" + "="*60)
        print("SightSync - Modern Professional Interface")
        print("="*60)
        print("\nWelcome screen loaded successfully!")
        print("Waiting for interaction...")
        print("="*60 + "\n")

        # AUTO CLOSE TIMER
        if self.timeout is not None:
            # convert seconds → milliseconds
            ms = int(self.timeout * 1000)
            self.root.after(ms, self._auto_close)

        self.root.mainloop()
        return True

    def _auto_close(self):
        print("✓ Auto-close timer reached. Closing window.")
        self.root.destroy()



def main():
    """Main entry point"""
    app = SightSyncWelcome()
    return app.run()


if __name__ == "__main__":
    main()