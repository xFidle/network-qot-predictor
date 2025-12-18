import tkinter as tk
from pathlib import Path
from tkinter import font

from PIL import Image, ImageTk


class LabelingWindow(tk.Tk):
    def __init__(self, class_names: tuple[str, str]) -> None:
        super().__init__()
        self.font = font.Font(family="Arial", size=20, weight="bold")
        self.current_image: ImageTk.PhotoImage | None = None

        self.image_frame = tk.Frame(self, width=512, height=512)
        self.image_frame.pack(padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True)

        minority_class, majority_class = class_names

        self.selected_class = tk.StringVar(value="")
        self.radio_button_0 = tk.Radiobutton(
            self,
            text=f"{minority_class} (0)",
            variable=self.selected_class,
            value="0",
            font=self.font,
        )
        self.radio_button_0.pack(side=tk.LEFT, padx=10)

        self.radio_button_1 = tk.Radiobutton(
            self,
            text=f"{majority_class} (1)",
            variable=self.selected_class,
            value="1",
            font=self.font,
        )
        self.radio_button_1.pack(side=tk.LEFT, padx=10)

        self.confirmed = tk.IntVar(value=0)
        self.confirm_button = tk.Button(
            self, text="Confirm", command=self._on_confirm, font=self.font
        )
        self.quitted = False
        self.confirm_button.pack(side=tk.RIGHT)
        self.confirm_button = tk.Button(self, text="Quit", command=self._on_quit, font=self.font)
        self.confirm_button.pack(side=tk.RIGHT)

    def set_sample(self, image_path: Path) -> None:
        img = Image.open(image_path)
        img = img.resize((512, 512))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.current_image = photo
        self.selected_class.set("")

    def wait_for_label(self) -> str:
        self.confirmed.set(0)
        self.wait_variable(self.confirmed)
        if self.quitted:
            return "q"
        return self.selected_class.get()

    def quit(self) -> None:
        self.destroy()

    def _on_confirm(self) -> None:
        if self.selected_class.get():
            self.confirmed.set(1)

    def _on_quit(self) -> None:
        self.quitted = True
        self.selected_class.set("q")
        self.confirmed.set(0)
