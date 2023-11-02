import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import vlc

class VideoPlayerApp:
    def __init__(self, master, video_path):
        self.master = master
        self.master.title("Video Player")

        self.button = ttk.Button(self.master, text="Play Video", command=self.play_video)
        self.button.pack(pady=20)

        self.video_path = video_path

        messagebox.showinfo("Info", "Click 'OK' to start playing the video.")

    def play_video(self):
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new(self.video_path)

        self.player.set_media(self.media)
        self.player.play()

        self.master.after(100, self.check_playing)

    def check_playing(self):
        if self.player.get_state() == vlc.State.Ended:
            self.player.stop()
            self.player.release()
            self.master.destroy()
        else:
            self.master.after(100, self.check_playing)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root, "path_to_your_video.mp4")
    root.mainloop()
