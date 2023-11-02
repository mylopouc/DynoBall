import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence


class Interface:

    def __init__(self):
        self.root = tk.Tk()
        self.radius_entry = ttk.Entry()
        self.volume_label = ttk.Label()
        self.frames = []
        self.frame_idx = 0
        self.gif_label = ttk.Label()

    def calculate_volume(self):
        try:
            radius = float(self.radius_entry.get())
            volume = (4 / 3) * 3.14 * (radius ** 3)
            self.volume_label.config(text=f"Volume: {volume:.2f} units³")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the radius.")

    def update_frame(self):
        frame = self.frames[self.frame_idx]
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        self.gif_label.configure(image=frame)
        self.gif_label.after(100, self.update_frame)

    def display_animation(self):
        try:
            radius = float(self.radius_entry.get())
            img = Image.open("animation.gif")
            img = img.resize((500, 500), Image.ANTIALIAS)

            gif = ImageTk.PhotoImage(img)
            self.gif_label = ttk.Label(self.root, image=gif)
            self.gif_label.image = gif
            self.gif_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))

            # Load the GIF
            self.frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(img)]
            self.frame_idx = 0

            self.update_frame()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the radius.")

    def create(self):

        # self.root = tk.Tk()
        self.root.title("Sphere Volume Calculator")

        radius_label = ttk.Label(self.root, text="Enter Radius:")
        radius_label.grid(row=0, column=0, padx=(10, 0), pady=(10, 0))

        self.radius_entry = ttk.Entry(self.root)
        self.radius_entry.grid(row=0, column=1, padx=(0, 10), pady=(10, 0))

        calculate_button = ttk.Button(self.root, text="Calculate Volume", command=self.calculate_volume)
        calculate_button.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.volume_label = ttk.Label(self.root, text="")
        self.volume_label.grid(row=2, column=0, columnspan=2)

        display_button = ttk.Button(self.root, text="Display Animation", command=self.display_animation)
        display_button.grid(row=3, column=0, columnspan=2, pady=(10, 0))


        self.root.mainloop()
