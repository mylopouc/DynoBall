import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence

class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("Sphere Volume Calculator")

        self.radius_label = ttk.Label(master, text="Enter Radius:")
        self.radius_label.grid(row=0, column=0, padx=(10, 0), pady=(10, 0))

        self.radius_entry = ttk.Entry(master)
        self.radius_entry.grid(row=0, column=1, padx=(0, 10), pady=(10, 0))

        self.calculate_button = ttk.Button(master, text="Calculate Volume", command=self.calculate_volume)
        self.calculate_button.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.volume_label = ttk.Label(master, text="")
        self.volume_label.grid(row=2, column=0, columnspan=2)

        self.display_button = ttk.Button(master, text="Display Animation", command=self.display_animation)
        self.display_button.grid(row=3, column=0, columnspan=2, pady=(10, 0))

    def calculate_volume(self):
        try:
            radius = float(self.radius_entry.get())
            volume = (4/3) * 3.14 * (radius ** 3)
            self.volume_label.config(text=f"Volume: {volume:.2f} units³")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the radius.")

    def update_frame(self):
        frame = frames[self.frame_idx]
        self.frame_idx = (self.frame_idx + 1) % len(frames)
        self.gif_label.configure(image=frame)
        self.master.after(100, self.update_frame)

    def display_animation(self):
        try:
            radius = float(self.radius_entry.get())
            img = Image.open("animation.gif")
            img = img.resize((500, 500), Image.ANTIALIAS)
            self.gif = ImageTk.PhotoImage(img)
            self.gif_label = ttk.Label(self.master, image=self.gif)
            self.gif_label.grid(row=4, column=0, columnspan=2, pady=(10, 0))

            global frames
            frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(img)]

            self.frame_idx = 0

            self.update_frame()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the radius")

if __name__ == "__main__":
    root = tk.Tk()
    interface = Interface(root)
    root.mainloop()