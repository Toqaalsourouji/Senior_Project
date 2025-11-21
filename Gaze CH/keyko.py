from tkinter import *
import tkinter.font as font
from tkinter import messagebox
from sys import exit as end
from os import system
import time
import json


def time_calculator(start_inf, label=""):
    end_inf = time.time()
    calc_time = end_inf - start_inf
    if calc_time < 0.02:
        return
    print(f"{label} time: {calc_time*1000:.5f} ms")


has_keyboard = True

try:
    import keyboard
except (ModuleNotFoundError, ImportError):
    dummy = Tk()
    dummy.withdraw()
    messagebox.showwarning('Missing Module: keyboard', 'Your system is missing the module "keyboard" for this program to work correctly.\n\nPlease click OK to install the "keyboard" module automatically.\nIn case this fails, the keyboard will still open in a non functional state')
    kbmodulestatus = system('python -m pip install keyboard')
    if kbmodulestatus != 0:
        messagebox.showerror('Error', 'Couldn\'t install "keyboard" module automatically. Please try again manually in command prompt using command:\n\npip install keyboard')
        dummy.destroy()
        has_keyboard = False
    else:
        import keyboard
        dummy.destroy()
        has_keyboard = True

has_pynput = True

try:
    from pynput.mouse import Controller
    mouse = Controller()
except (ModuleNotFoundError, ImportError):
    print("pynput not found, installing...")
    pynputstatus = system('python -m pip install pynput --quiet')
    if pynputstatus != 0:
        print("Failed to install pynput. Mouse snapping disabled.")
        has_pynput = False
        mouse = None
    else:
        print("pynput installed successfully!")
        from pynput.mouse import Controller
        mouse = Controller()
        has_pynput = True

class VirtualKeyboard:

    def __init__(self, master=Tk()):
        self.master = master

        # Navigation state
        self.current_row = 2
        self.current_col = 6
        self.nav_enabled = True
        
        self.all_buttons = []
        self.mousemovey=True
        # Colors
        self.darkgray = "#242424"
        self.gray = "#383838"
        self.darkred = "#591717"
        self.red = "#822626"
        self.darkpurple = "#7151c4"
        self.purple = "#9369ff"
        self.darkblue = "#386cba"
        self.blue = "#488bf0"
        self.darkyellow = "#bfb967"
        self.yellow = "#ebe481"
        self.highlight_color = "#00BFFF"
        self.darkgreen = "#2d5016"
        self.green = "#4a7c2c"

        self.master.configure(bg=self.gray)
        self.unmap_bind = self.master.bind("<Unmap>", lambda e: [self.rel_win(), self.rel_alts(), self.rel_shifts(), self.rel_ctrls()])

        self.master.protocol("WM_DELETE_WINDOW", lambda: [self.master.destroy(), end()])
        self.master.title("Virtual Keyboard (NON FUNCTIONAL)")

        self.user_scr_width = int(self.master.winfo_screenwidth())
        self.user_scr_height = int(self.master.winfo_screenheight())

        self.load_settings()
        self.master.attributes('-alpha', self.trans_value)
        self.master.attributes('-topmost', True)

        self.size_value_map = [
            (int(0.56 * self.user_scr_width), int(0.34 * self.user_scr_height)),
            (int(0.63 * self.user_scr_width), int(0.37 * self.user_scr_height)),
            (int(0.70 * self.user_scr_width), int(0.42 * self.user_scr_height)),
            (int(0.78 * self.user_scr_width), int(0.46 * self.user_scr_height)),
            (int(0.86 * self.user_scr_width), int(0.51 * self.user_scr_height)),
            (int(0.94 * self.user_scr_width), int(0.56 * self.user_scr_height))
        ]

        self.size_value_names = ["Microscope", "Small", "Medium", "Large", "Very Large", "GIGA SIZE"]

        self.master.geometry(f"{self.size_value_map[self.size_current][0]}x{self.size_value_map[self.size_current][1]}")
        self.master.resizable(False, False)

        self.row1keys = ["esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
                         "f11", "f12", "print_screen", "scroll_lock", "numlock"]

        self.row2keys = ["`", "1", "2", "3", "4", "5", "6", "7",
                         "8", "9", "0", "-", "=", "backspace", "page_up"]

        self.row3keys = ["tab", "q", "w", "e", "r", "t", "y", 'u',
                         'i', 'o', 'p', '[', ']', 'enter', 'page_down']

        self.row4keys = ["capslock", 'a', 's', 'd', 'f', 'g', 'h', 'j',
                         'k', 'l', ';', "'", '\\', 'delete', 'home', 'end']

        self.row5keys = ["left shift", 'z', 'x', 'c', 'v', 'b', 'n', 'm',
                         ',', '.', '/', 'right shift', 'up', 'insert']

        self.row6keys = ["left ctrl", 'win', 'alt', 'spacebar', 'alt gr',
                         'right ctrl', 'Mouse: True', 'left', 'down', 'right']

        self.row1buttons = []
        self.row2buttons = []
        self.row3buttons = []
        self.row4buttons = []
        self.row5buttons = []
        self.row6buttons = []
        self.row7buttons = []

        appendrow1 = self.row1buttons.append
        appendrow2 = self.row2buttons.append
        appendrow3 = self.row3buttons.append
        appendrow4 = self.row4buttons.append
        appendrow5 = self.row5buttons.append
        appendrow6 = self.row6buttons.append
        appendrow7 = self.row7buttons.append

        self.master.columnconfigure(0, weight=1)
        for i in range(7):
            self.master.rowconfigure(i, weight=1)

        if self.user_scr_width < 1600:
            self.keyfont = font.Font(family="Calibri", size=10, weight='bold')
            self.bottomfont = font.Font(family='Calibri', size=11, weight='bold')
        else:
            self.keyfont = font.Font(family="Calibri", size=13, weight='bold')
            self.bottomfont = font.Font(family='Calibri', size=13, weight='bold')

        self.spl_key_pressed = False

        #   ROW 1
        keyframe1 = Frame(self.master, height=1)
        keyframe1.rowconfigure(0, weight=1)

        for key in self.row1keys:
            ind = self.row1keys.index(key)
            keyframe1.columnconfigure(ind, weight=1)
            appendrow1(Button(
                keyframe1,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=1,
                relief=RAISED
            ))
            if key == "print_screen":
                self.row1buttons[ind].config(text="PrtScr", width=3, height=2)
            elif key == "scroll_lock":
                self.row1buttons[ind].config(text="ScrLck", width=3)
            elif key == "numlock":
                self.row1buttons[ind].config(text="NumLck", width=3)
            else:
                self.row1buttons[ind].config(text=key.title())

            self.row1buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        #   ROW 2
        keyframe2 = Frame(self.master, height=1)
        keyframe2.rowconfigure(0, weight=1)

        for key in self.row2keys:
            ind = self.row2keys.index(key)
            if ind == 13:
                keyframe2.columnconfigure(ind, weight=2)
            else:
                keyframe2.columnconfigure(ind, weight=1)
            appendrow2(Button(
                keyframe2,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=1,
                relief=RAISED
            ))
            if key == "page_up":
                self.row2buttons[ind].config(text="Pg Up", width=2)
            elif key == "backspace":
                self.row2buttons[ind].config(text=key.title(), width=4)

            self.row2buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        self.row2buttons[0].config(text="~\n`")
        self.row2buttons[1].config(text="!\n1")
        self.row2buttons[2].config(text="@\n2")
        self.row2buttons[3].config(text="#\n3")
        self.row2buttons[4].config(text="$\n4")
        self.row2buttons[5].config(text="%\n5")
        self.row2buttons[6].config(text="^\n6")
        self.row2buttons[7].config(text="&\n7")
        self.row2buttons[8].config(text="*\n8")
        self.row2buttons[9].config(text="(\n9")
        self.row2buttons[10].config(text=")\n0")
        self.row2buttons[11].config(text="_\n-")
        self.row2buttons[12].config(text="+\n=")

        #   ROW 3
        keyframe3 = Frame(self.master, width=1)
        keyframe3.rowconfigure(0, weight=1)

        for key in self.row3keys:
            ind = self.row3keys.index(key)
            if ind == 13:
                keyframe3.columnconfigure(ind, weight=2)
            else:
                keyframe3.columnconfigure(ind, weight=1)
            appendrow3(Button(
                keyframe3,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=1,
                relief=RAISED
            ))
            if key == "page_down":
                self.row3buttons[ind].config(text="Pg Dn", width=2)
            elif key == "[":
                self.row3buttons[ind].config(text="{\n[", width=1)
            elif key == "]":
                self.row3buttons[ind].config(text="}\n]", width=1)
            elif key == "tab":
                self.row3buttons[ind].config(text="Tab", width=3)
            elif key == "enter":
                self.row3buttons[ind].config(text="Enter", width=3)
            else:
                self.row3buttons[ind].config(text=key.title())

            self.row3buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        #   ROW 4
        keyframe4 = Frame(self.master, height=1)
        keyframe4.rowconfigure(0, weight=1)

        for key in self.row4keys:
            ind = self.row4keys.index(key)
            keyframe4.columnconfigure(ind, weight=1)
            appendrow4(Button(
                keyframe4,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=2,
                relief=RAISED
            ))
            if key == ";":
                self.row4buttons[ind].config(text=":\n;")
            elif key == "'":
                self.row4buttons[ind].config(text='"\n\'')
            elif key == "\\":
                self.row4buttons[ind].config(text="|\n\\")
            elif key == "capslock":
                self.row4buttons[ind].config(text="CapsLck", width=5)
            else:
                self.row4buttons[ind].config(text=key.title())

            self.row4buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        #   ROW 5
        keyframe5 = Frame(self.master, height=1)
        keyframe5.rowconfigure(0, weight=1)

        for key in self.row5keys:
            ind = self.row5keys.index(key)
            if ind == 0 or ind == 11:
                keyframe5.columnconfigure(ind, weight=3)
            else:
                keyframe5.columnconfigure(ind, weight=1)
            appendrow5(Button(
                keyframe5,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=1,
                relief=RAISED
            ))
            if key == ",":
                self.row5buttons[ind].config(text="<\n,")
            elif key == ".":
                self.row5buttons[ind].config(text=">\n.")
            elif key == "/":
                self.row5buttons[ind].config(text="?\n/")
            elif key == "up":
                self.row5buttons[ind].config(text="↑")
            elif key == "insert":
                self.row5buttons[ind].config(text="Insert", width=1)
            elif key == "left shift":
                self.row5buttons[ind].config(text="Shift", width=6)
            elif key == "right shift":
                self.row5buttons[ind].config(text="Shift", width=6)
            else:
                self.row5buttons[ind].config(text=key.title())

            self.row5buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        #   ROW 6
        keyframe6 = Frame(self.master, height=1)
        keyframe6.rowconfigure(0, weight=1)

        for key in self.row6keys:
            ind = self.row6keys.index(key)
            if ind == 3:
                keyframe6.columnconfigure(ind, weight=12)
            else:
                keyframe6.columnconfigure(ind, weight=1)
            appendrow6(Button(
                keyframe6,
                font=self.keyfont,
                border=7,
                bg=self.gray,
                activebackground=self.darkgray,
                activeforeground="#bababa",
                fg="white",
                width=1,
                relief=RAISED
            ))

            if key == "left":
                self.row6buttons[ind].config(text="←")
            elif key == "down":
                self.row6buttons[ind].config(text="↓")
            elif key == "right":
                self.row6buttons[ind].config(text="→")
            elif key == "spacebar":
                self.row6buttons[ind].config(text="\n")
            elif key == "win":
                self.row6buttons[ind].config(text="Win")
            elif key == "left ctrl":
                self.row6buttons[ind].config(text="Ctrl")
            elif key == "right ctrl":
                self.row6buttons[ind].config(text="Ctrl")
            elif key == "alt":
                self.row6buttons[ind].config(text="Alt")
            elif key == "alt gr":
                self.row6buttons[ind].config(text="Alt")
            elif key == "Mouse: True":
                self.row6buttons[ind].config(text=key, width=4, bg=self.red, activebackground=self.darkred, command=self.donothing)
            else:
                self.row6buttons[ind].config(text=key.title())

            self.row6buttons[ind].grid(row=0, column=ind, sticky="NSEW")

        self.update_mouse_button_text()
        #   ROW 7 - Control buttons
        infoframe7 = Frame(self.master, height=1, bg=self.gray)
        infoframe7.rowconfigure(0, weight=1)

        # Copy button
        infoframe7.columnconfigure(0, weight=1)
        self.copy_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.purple,
            text="COPY",
            activebackground=self.darkpurple,
            activeforeground="black",
            fg="black",
            relief=RAISED
        )
        self.copy_button.grid(row=0, column=0, padx=2, sticky="NSEW")
        appendrow7(self.copy_button)

        # Cut button
        infoframe7.columnconfigure(1, weight=1)
        self.cut_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.purple,
            text="CUT",
            activebackground=self.darkpurple,
            activeforeground="black",
            fg="black",
            relief=RAISED
        )
        self.cut_button.grid(row=0, column=1, padx=2, sticky="NSEW")
        appendrow7(self.cut_button)

        # Paste button
        infoframe7.columnconfigure(2, weight=1)
        self.paste_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.purple,
            text="PASTE",
            activebackground=self.darkpurple,
            activeforeground="black",
            fg="black",
            relief=RAISED
        )
        self.paste_button.grid(row=0, column=2, padx=2, sticky="NSEW")
        appendrow7(self.paste_button)

        # Select all button
        infoframe7.columnconfigure(3, weight=1)
        self.selall_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.purple,
            text="SELECT ALL",
            activebackground=self.darkpurple,
            activeforeground="black",
            fg="black",
            relief=RAISED
        )
        self.selall_button.grid(row=0, column=3, padx=2, sticky="NSEW")
        appendrow7(self.selall_button)

        # Opacity cycle button
        infoframe7.columnconfigure(4, weight=1)
        self.opacity_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.yellow,
            text=f"Opacity\n{int(self.trans_value * 100)}%",
            activebackground=self.darkyellow,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=self.cycle_opacity
        )
        self.opacity_button.grid(row=0, column=4, padx=2, sticky="NSEW")
        appendrow7(self.opacity_button)

        # Size cycle button
        infoframe7.columnconfigure(5, weight=1)
        self.size_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.yellow,
            text=f"Size\n{self.size_value_names[self.size_current]}",
            activebackground=self.darkyellow,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=self.cycle_size
        )
        self.size_button.grid(row=0, column=5, padx=2, sticky="NSEW")
        appendrow7(self.size_button)

        # Task manager button
        infoframe7.columnconfigure(6, weight=1)
        self.taskmnger_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.blue,
            text="Task Manager",
            activebackground=self.darkblue,
            activeforeground="black",
            fg="black",
            relief=RAISED
        )
        self.taskmnger_button.grid(row=0, column=6, padx=2, sticky="NSEW")
        appendrow7(self.taskmnger_button)

        # Pin keyboard button
        infoframe7.columnconfigure(7, weight=1)
        self.pinkb_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.darkblue,
            text="Unpin 📌",
            activebackground=self.blue,
            activeforeground="black",
            fg="black",
            relief=SUNKEN,
            command=self.keyboard_top)
        self.pinkb_button.grid(row=0, column=7, padx=2, sticky="NSEW")
        appendrow7(self.pinkb_button)

        # Move keyboard UP button
        infoframe7.columnconfigure(8, weight=1)
        self.move_up_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.green,
            text="↑",
            activebackground=self.darkgreen,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=lambda: self.move_keyboard(0, -30)
        )
        self.move_up_button.grid(row=0, column=8, padx=2, sticky="NSEW")
        appendrow7(self.move_up_button)

        # Move keyboard DOWN button
        infoframe7.columnconfigure(9, weight=1)
        self.move_down_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.green,
            text="↓",
            activebackground=self.darkgreen,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=lambda: self.move_keyboard(0, 30)
        )
        self.move_down_button.grid(row=0, column=9, padx=2, sticky="NSEW")
        appendrow7(self.move_down_button)

        # Move keyboard LEFT button
        infoframe7.columnconfigure(10, weight=1)
        self.move_left_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.green,
            text="←",
            activebackground=self.darkgreen,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=lambda: self.move_keyboard(-30, 0)
        )
        self.move_left_button.grid(row=0, column=10, padx=2, sticky="NSEW")
        appendrow7(self.move_left_button)

        # Move keyboard RIGHT button
        infoframe7.columnconfigure(11, weight=1)
        self.move_right_button = Button(
            infoframe7,
            font=self.bottomfont,
            border=5,
            bg=self.green,
            text="→",
            activebackground=self.darkgreen,
            activeforeground="black",
            fg="black",
            relief=RAISED,
            command=lambda: self.move_keyboard(30, 0)
        )
        self.move_right_button.grid(row=0, column=11, padx=2, sticky="NSEW")
        appendrow7(self.move_right_button)

        # Store button rows for navigation
        self.all_buttons = [
            self.row1buttons,
            self.row2buttons,
            self.row3buttons,
            self.row4buttons,
            self.row5buttons,
            self.row6buttons,
            self.row7buttons
        ]

        # Add frames to window
        keyframe1.grid(row=0, sticky="NSEW", padx=9, pady=6)
        keyframe2.grid(row=1, sticky="NSEW", padx=9)
        keyframe3.grid(row=2, sticky="NSEW", padx=9)
        keyframe4.grid(row=3, sticky="NSEW", padx=9)
        keyframe5.grid(row=4, sticky="NSEW", padx=9)
        keyframe6.grid(row=5, padx=9, sticky="NSEW")
        infoframe7.grid(row=6, padx=9, pady=5, sticky="NSEW")

        # Bind arrow keys for navigation
        self.master.bind('<Up>', lambda e: self.navigate('up'))
        self.master.bind('<Down>', lambda e: self.navigate('down'))
        self.master.bind('<Left>', lambda e: self.navigate('left'))
        self.master.bind('<Right>', lambda e: self.navigate('right'))
        self.master.bind('<Return>', lambda e: self.activate_current_key())
        self.master.bind('<space>', lambda e: self.activate_current_key())

        # Highlight starting position
        self.master.after(100, self.highlight_current_key)

    def move_keyboard(self, dx, dy):
        """Move the keyboard window by dx, dy pixels"""
        current_x = self.master.winfo_x()
        current_y = self.master.winfo_y()
        new_x = current_x + dx
        new_y = current_y + dy
        self.master.geometry(f"+{new_x}+{new_y}")
        self.master.after(50, self.highlight_current_key)

    def cycle_opacity(self):
        """Cycle through opacity values"""
        opacity_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        current_index = min(range(len(opacity_values)), key=lambda i: abs(opacity_values[i] - self.trans_value))
        next_index = (current_index + 1) % len(opacity_values)
        self.trans_value = opacity_values[next_index]
        self.master.attributes('-alpha', self.trans_value)
        self.opacity_button.config(text=f"Opacity\n{int(self.trans_value * 100)}%")
        self.save_settings()
        self.master.after(50, self.highlight_current_key)

    def cycle_size(self):
        """Cycle through size values"""
        self.size_current = (self.size_current + 1) % len(self.size_value_names)
        new_width = self.size_value_map[self.size_current][0]
        new_height = self.size_value_map[self.size_current][1]
        self.master.geometry(f"{new_width}x{new_height}")
        self.size_button.config(text=f"Size\n{self.size_value_names[self.size_current]}")
        self.save_settings()
        self.master.after(100, self.highlight_current_key)

    def load_settings(self):
        """Load saved settings from file"""
        try:
            with open('vkb_settings.json', 'r') as f:
                settings = json.load(f)
                self.trans_value = settings.get('transparency', 0.7)
                self.size_current = settings.get('size', 2)
                self.mousemovey = settings.get('mousemove', True)  # Add this
        except (FileNotFoundError, json.JSONDecodeError):
            self.trans_value = 0.7
            self.size_current = 2
            self.mousemovey = True  # Add this

    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'transparency': self.trans_value,
            'size': self.size_current,
            'mousemove': self.mousemovey  # Add this
        }
        try:
            with open('vkb_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
        except:
            pass
        
    def update_mouse_button_text(self):
          """Update the button text to show mouse snap status"""
          status = "ON" if self.mousemovey else "OFF"
          # Find the :) button (it's at row6buttons[6])
          self.row6buttons[6].config(text=f"Mouse\n{status}")


    def on_button_click(self, row, col):
        """Update navigation position when button is clicked with mouse"""
        self.current_row = row
        self.current_col = col
        self.highlight_current_key()        



    def get_button_original_colors(self, button):
        """Store original colors for a button"""
        if not hasattr(button, '_original_bg'):
            button._original_bg = button.cget('bg')
            button._original_fg = button.cget('fg')
        return button._original_bg, button._original_fg

    def highlight_current_key(self):
        """Highlight the currently selected key"""
        for row in self.all_buttons:
            for button in row:
                orig_bg, orig_fg = self.get_button_original_colors(button)
                if button.cget('relief') != SUNKEN:
                    button.config(bg=orig_bg, fg=orig_fg)
        
        if 0 <= self.current_row < len(self.all_buttons):
            row = self.all_buttons[self.current_row]
            if 0 <= self.current_col < len(row):
                current_button = row[self.current_col]
                current_button.config(bg=self.highlight_color, fg="black")
                self.snap_mouse_to_button(current_button)

    def snap_mouse_to_button(self, button):
        """Move mouse cursor to the center of the button"""
        if not has_pynput or mouse is None or self.mousemovey is False:
            return
        try:
            x = button.winfo_rootx() + button.winfo_width() // 2
            y = button.winfo_rooty() + button.winfo_height() // 2
            mouse.position = (x, y)
        except:
            pass

    def navigate(self, direction):
        """Navigate between keys using arrow keys with smart column alignment"""
        if not self.nav_enabled:
            return           
        

        #dis is dah place holder, Finds which button in the new row has its center closest to that x-position instea fo quikc siwtchin
        if direction == 'left':
            self.current_col = max(0, self.current_col - 1)
        elif direction == 'right':
            row = self.all_buttons[self.current_row]
            self.current_col = min(len(row) - 1, self.current_col + 1)
        elif direction == 'up' or direction == 'down':
            # Store the current button's horizontal position before moving
            current_button = self.all_buttons[self.current_row][self.current_col]
            try:
                current_x = current_button.winfo_rootx() + current_button.winfo_width() // 2
            except:
                current_x = 0
            
            # Move to new row
            if direction == 'up':
                self.current_row = max(0, self.current_row - 1)
            else:  # down
                self.current_row = min(len(self.all_buttons) - 1, self.current_row + 1)
            
            # Find the button in the new row that's closest to our horizontal position
            new_row = self.all_buttons[self.current_row]
            closest_col = 0
            min_distance = float('inf')
            
            for col_idx, button in enumerate(new_row):
                try:
                    button_x = button.winfo_rootx() + button.winfo_width() // 2
                    distance = abs(button_x - current_x)
                    if distance < min_distance:
                        min_distance = distance
                        closest_col = col_idx
                except:
                    pass
            
            self.current_col = closest_col
        
        self.highlight_current_key()

    def activate_current_key(self):
        """Activate/click the currently highlighted key"""
        if 0 <= self.current_row < len(self.all_buttons):
            row = self.all_buttons[self.current_row]
            if 0 <= self.current_col < len(row):
                current_button = row[self.current_col]
                current_button.invoke()

    def donothing(self):
        self.mousemovey = not self.mousemovey
        self.update_mouse_button_text()
        self.save_settings()

    def quest_press(self, x):
        if self.row5buttons[0].cget('relief') == SUNKEN:
            if x == "-":
                self.vpresskey("shift+_")
            elif x == "/":
                self.vpresskey("shift+?")
        else:
            self.vpresskey(x)

        if self.spl_key_pressed:
            keyboard.press('shift')

    def rel_shifts(self):
        keyboard.release('shift')
        self.row5buttons[0].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")
        self.row5buttons[11].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")

    def prs_shifts(self):
        keyboard.press('shift')
        self.row5buttons[0].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")
        self.row5buttons[11].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")

    def rel_ctrls(self):
        keyboard.release('ctrl')
        self.row6buttons[0].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")
        self.row6buttons[5].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")

    def prs_ctrls(self):
        keyboard.press('ctrl')
        self.row6buttons[0].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")
        self.row6buttons[5].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")

    def rel_alts(self):
        keyboard.release('alt')
        self.row6buttons[2].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")
        self.row6buttons[4].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")

    def prs_alts(self):
        keyboard.press('alt')
        self.row6buttons[2].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")
        self.row6buttons[4].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")

    def rel_win(self):
        keyboard.release('win')
        self.row6buttons[1].config(relief=RAISED, bg=self.gray, activebackground=self.darkgray, activeforeground="#bababa", fg="white")

    def prs_win(self):
        keyboard.press('win')
        self.row6buttons[1].config(relief=SUNKEN, activebackground=self.gray, bg=self.darkgray, fg="#bababa", activeforeground="white")

    def vpresskey(self, x):
        self.master.unbind("<Unmap>", self.unmap_bind)
        self.master.withdraw()
        self.master.after(5, keyboard.send(x))
        self.master.after(20, self.master.wm_deiconify)
        if not self.spl_key_pressed:
            self.rel_shifts()
            self.rel_ctrls()
            self.rel_alts()
            self.rel_win()

        if self.pinkb_button.cget('relief') == RAISED:
            self.addkbtotop()
        self.unmap_bind = self.master.bind("<Unmap>", lambda e: [self.rel_win(), self.rel_alts(), self.rel_shifts(), self.rel_ctrls()])
        self.master.after(50, self.highlight_current_key)

    def vupdownkey(self, event, y, a):
        self.master.after(1, self.donothing())

        if y == "shift":
            if self.row5buttons[0].cget('relief') == SUNKEN or self.row5buttons[11].cget('relief') == SUNKEN:
                self.rel_shifts()
            else:
                self.prs_shifts()
        elif y == "ctrl":
            if self.row6buttons[0].cget('relief') == SUNKEN or self.row6buttons[5].cget('relief') == SUNKEN:
                self.rel_ctrls()
            else:
                self.prs_ctrls()
        elif y == "alt":
            if self.row6buttons[2].cget('relief') == SUNKEN or self.row6buttons[4].cget('relief') == SUNKEN:
                self.rel_alts()
            else:
                self.prs_alts()
        elif y == "win":
            if self.row6buttons[1].cget('relief') == SUNKEN:
                self.rel_win()
            else:
                self.prs_win()

        if a == "L":
            self.spl_key_pressed = False
        elif a == "R":
            self.spl_key_pressed = True
        
        self.master.after(50, self.highlight_current_key)

    def removekbfromtop(self):
        self.master.attributes('-topmost', False)
        self.pinkb_button.config(bg=self.blue, activebackground=self.darkblue, relief=RAISED, text="Pin 📌")
        self.master.update()

    def addkbtotop(self):
        self.master.attributes('-topmost', True)
        self.pinkb_button.config(relief=SUNKEN, bg=self.darkblue, activebackground=self.blue, text="Unpin 📌")
        self.master.update()

    def keyboard_top(self):
        if self.pinkb_button.cget("relief") == RAISED:
            self.addkbtotop()
        elif self.pinkb_button.cget("relief") == SUNKEN:
            self.removekbfromtop()
        else:
            self.removekbfromtop()

    def start(self):
        running = True
        
        def on_closing():
            nonlocal running
            running = False
            if has_keyboard:
                keyboard.release('shift')
                keyboard.release('ctrl')
                keyboard.release('alt')
                keyboard.release('win')
            self.master.destroy()
            end()
        
        self.master.protocol("WM_DELETE_WINDOW", on_closing)
        
        while running:
            start_time = time.time()
            try:
                self.master.update()
            except:
                break
            time_calculator(start_time, "Main loop")

    def engine(self):
        self.master.title("Virtual Keyboard")
        self.master.protocol("WM_DELETE_WINDOW", lambda: [keyboard.release('shift'), keyboard.release('ctrl'), keyboard.release('alt'), keyboard.release('win'), self.master.destroy(), end()])
        
        for key in self.row1keys:
            ind = self.row1keys.index(key)
            self.row1buttons[ind].config(command=lambda x=key: self.vpresskey(x))

        for key in self.row2keys:
            ind = self.row2keys.index(key)
            self.row2buttons[ind].config(command=lambda x=key: self.vpresskey(x))
        self.row2buttons[11].config(command=lambda x='-': self.quest_press(x))

        for key in self.row3keys:
            ind = self.row3keys.index(key)
            self.row3buttons[ind].config(command=lambda x=key: self.vpresskey(x))

        for key in self.row4keys:
            ind = self.row4keys.index(key)
            self.row4buttons[ind].config(command=lambda x=key: self.vpresskey(x))

        for key in self.row5keys:
            ind = self.row5keys.index(key)
            self.row5buttons[ind].config(command=lambda x=key: self.vpresskey(x))
            if key == "/":
                self.row5buttons[ind].config(command=lambda x='/': self.quest_press(x))
            elif key == "left shift":
                self.row5buttons[ind].config(command=lambda: self.vupdownkey(event="<Button-1>", y='shift', a="L"))
                self.row5buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='shift', a="R": self.vupdownkey(event, y, a))
            elif key == "right shift":
                self.row5buttons[ind].config(command=lambda: self.vupdownkey(event="<Button-1>", y='shift', a="L"))
                self.row5buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='shift', a="R": self.vupdownkey(event, y, a))

        for key in self.row6keys:
            ind = self.row6keys.index(key)
            self.row6buttons[ind].config(command=lambda x=key: self.vpresskey(x))
            if key == "win":
                self.row6buttons[ind].config(command=lambda: self.vupdownkey("<Button-1>", 'win', "L"))
                self.row6buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='win', a="R": self.vupdownkey(event, y, a))
            elif key == "Mouse: True":
                self.row6buttons[ind].config(command=self.donothing)
            elif key == "left ctrl":
                self.row6buttons[ind].config(command=lambda: self.vupdownkey("<Button-1>", 'ctrl', "L"))
                self.row6buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='ctrl', a="R": self.vupdownkey(event, y, a))
            elif key == "right ctrl":
                self.row6buttons[ind].config(command=lambda: self.vupdownkey("<Button-1>", 'ctrl', "L"))
                self.row6buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='ctrl', a="R": self.vupdownkey(event, y, a))
            elif key == "alt":
                self.row6buttons[ind].config(command=lambda: self.vupdownkey("<Button-1>", 'alt', "L"))
                self.row6buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='alt', a="R": self.vupdownkey(event, y, a))
            elif key == "alt gr":
                self.row6buttons[ind].config(command=lambda: self.vupdownkey("<Button-1>", 'alt', "L"))
                self.row6buttons[ind].bind('<Button-3>', lambda event="<Button-3>", y='alt', a="R": self.vupdownkey(event, y, a))
        for row_idx, row in enumerate(self.all_buttons):
          for col_idx, button in enumerate(row):
              button.bind('<Button-1>', lambda e, r=row_idx, c=col_idx: self.on_button_click(r, c), add='+')


        self.copy_button.config(command=lambda: self.vpresskey('ctrl+c'))
        self.cut_button.config(command=lambda: self.vpresskey('ctrl+x'))
        self.paste_button.config(command=lambda: self.vpresskey('ctrl+v'))
        self.selall_button.config(command=lambda: self.vpresskey('ctrl+a'))
        self.taskmnger_button.config(command=lambda: [self.removekbfromtop(), self.vpresskey('ctrl+shift+esc')])


if __name__ == '__main__':
    keyboard1 = VirtualKeyboard()

    if has_keyboard:
        keyboard1.engine()

    keyboard1.start()