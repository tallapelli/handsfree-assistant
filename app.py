# app.py
import threading
import customtkinter
from virtual_mouse import run_virtual_mouse
from speech_commander import run_speech_commander

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hands-Free Assistant")
        self.geometry("600x450")

        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.is_mouse_running = False
        self.is_speech_running = False

        # Stop events to gracefully stop threads
        self.mouse_stop_event = threading.Event()
        self.speech_stop_event = threading.Event()

        # --- UI Layout ---
        self.title_label = customtkinter.CTkLabel(
            self, text="Hands-Free Control Center",
            font=customtkinter.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)

        # Virtual Mouse Section
        self.mouse_frame = customtkinter.CTkFrame(self)
        self.mouse_frame.pack(pady=10, padx=20, fill="x")
        self.mouse_label = customtkinter.CTkLabel(self.mouse_frame, text="Virtual Mouse (Gesture Control)", font=customtkinter.CTkFont(size=16))
        self.mouse_label.pack(side="left", padx=10, expand=True)
        self.mouse_switch = customtkinter.CTkSwitch(self.mouse_frame, text="", command=self.toggle_mouse)
        self.mouse_switch.pack(side="right", padx=10)

        # Speech Commander Section
        self.speech_frame = customtkinter.CTkFrame(self)
        self.speech_frame.pack(pady=10, padx=20, fill="x")
        self.speech_label = customtkinter.CTkLabel(self.speech_frame, text="Voice Commands (Speech-to-Text)", font=customtkinter.CTkFont(size=16))
        self.speech_label.pack(side="left", padx=10, expand=True)
        self.speech_switch = customtkinter.CTkSwitch(self.speech_frame, text="", command=self.toggle_speech)
        self.speech_switch.pack(side="right", padx=10)

        # Status
        self.status_label = customtkinter.CTkLabel(self, text="Status: Idle", font=customtkinter.CTkFont(size=12))
        self.status_label.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_status(self):
        if self.is_mouse_running and self.is_speech_running:
            self.status_label.configure(text="Status: Mouse and Voice Active")
        elif self.is_mouse_running:
            self.status_label.configure(text="Status: Virtual Mouse Running")
        elif self.is_speech_running:
            self.status_label.configure(text="Status: Voice Commands Active")
        else:
            self.status_label.configure(text="Status: Idle")

    def toggle_mouse(self):
        if self.mouse_switch.get() == 1:
            if not self.is_mouse_running:
                self.is_mouse_running = True
                self.mouse_stop_event.clear()
                self.mouse_thread = threading.Thread(target=run_virtual_mouse, args=(self.mouse_stop_event,), daemon=True)
                self.mouse_thread.start()
        else:
            if self.is_mouse_running:
                self.is_mouse_running = False
                self.mouse_stop_event.set()
        self.update_status()

    def toggle_speech(self):
        if self.speech_switch.get() == 1:
            if not self.is_speech_running:
                self.is_speech_running = True
                self.speech_stop_event.clear()
                # Pass the status label update function as a callback
                self.speech_thread = threading.Thread(target=run_speech_commander, args=(self.speech_stop_event, self.update_speech_status), daemon=True)
                self.speech_thread.start()
        else:
            if self.is_speech_running:
                self.is_speech_running = False
                self.speech_stop_event.set()
        self.update_status()

    def update_speech_status(self, text):
        self.status_label.configure(text=text)

    def on_close(self):
        self.mouse_stop_event.set()
        self.speech_stop_event.set()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()