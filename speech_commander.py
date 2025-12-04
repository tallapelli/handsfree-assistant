import queue
import sounddevice as sd
import whisper
import numpy as np
import threading
import pyautogui
import torch
import os
import json
import tkinter as tk
from tkinter import messagebox
from vosk import Model, KaldiRecognizer
import time
from collections import deque
import subprocess
import sys
import psutil

class EnhancedSpeechCommander:
    def __init__(self, stop_event, status_callback):
        self.stop_event = stop_event
        self.status_callback = status_callback
        
        # --- State Management ---
        self.mode = 'WAITING'  # WAITING, DICTATING, CONFIRMING
        
        # --- Wake Words ---
        self.wake_word = "start typing"
        self.stop_word = "stop typing"
        
        # --- ENHANCED: Voice Commands for Browser & App Control ---
        self.browser_commands = {
            # Tab Management
            "close tab": self._close_tab,
            "new tab": self._new_tab,
            "next tab": self._next_tab,
            "previous tab": self._previous_tab,
            "reopen tab": self._reopen_tab,
            "duplicate tab": self._duplicate_tab,
            
            # Window Management
            "new window": self._new_window,
            "close window": self._close_window,
            "minimize window": self._minimize_window,
            "maximize window": self._maximize_window,
            "switch window": self._switch_window,
            
            # Navigation
            "go back": self._go_back,
            "go forward": self._go_forward,
            "refresh page": self._refresh_page,
            "refresh": self._refresh_page,
            "reload": self._refresh_page,
            "home page": self._go_home,
            "open bookmarks": self._open_bookmarks,
            
            # Browser Specific
            "open incognito": self._open_incognito,
            "open private": self._open_incognito,
            "developer tools": self._open_dev_tools,
            "view source": self._view_source,
            "full screen": self._toggle_fullscreen,
            "zoom in": self._zoom_in,
            "zoom out": self._zoom_out,
            "zoom reset": self._zoom_reset,
            
            # Application Control
            "open chrome": lambda: self._open_application("chrome"),
            "open firefox": lambda: self._open_application("firefox"),
            "open edge": lambda: self._open_application("msedge"),
            "open notepad": lambda: self._open_application("notepad"),
            "open calculator": lambda: self._open_application("calc"),
            "open file explorer": lambda: self._open_application("explorer"),
            "open task manager": lambda: self._open_task_manager(),
            
            # System Control
            "alt tab": self._alt_tab,
            "show desktop": self._show_desktop,
            "lock screen": self._lock_screen,
            "take screenshot": self._take_screenshot,
            "open start menu": self._open_start_menu,
        }
        
        # Alternative wake phrases for better recognition
        self.wake_phrases = ["start typing", "begin typing", "start dictation"]
        self.stop_phrases = ["stop typing", "end typing", "stop", "stop dictation"]
        
        # --- Store transcribed text for confirmation ---
        self.pending_text = ""

        # --- VAD Parameters ---
        self.samplerate = 16000
        # --- OPTIMIZATION: Silence duration for auto-transcription ---
        self.silence_duration = 1.0 # Seconds of silence to trigger transcription
        
        # --- Control flags ---
        self.processing_confirmation = False
        
        # --- NEW: Audio buffer management with timestamps ---
        self.audio_segments = deque()  # Store (audio_data, timestamp) tuples
        self.buffer_start_time = None
        self.last_stop_detection_time = None
        
        # --- Vosk Model Setup (for fast wake-word detection) ---
        vosk_model_path = os.path.join("model", "vosk-model-small-en-us-0.15")
        if not os.path.exists(vosk_model_path):
            raise FileNotFoundError(f"Vosk model not found at {vosk_model_path}. Please follow setup instructions.")
        self.vosk_model = Model(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, self.samplerate)
        
        # --- NEW: Separate recognizer for real-time stop detection ---
        self.stop_recognizer = KaldiRecognizer(self.vosk_model, self.samplerate)
        
        # Enhanced grammar - include all commands
        all_phrases = (self.wake_phrases + self.stop_phrases + 
                      list(self.browser_commands.keys()))
        grammar = json.dumps(all_phrases)
        self.recognizer.SetWords(True)
        self.stop_recognizer.SetWords(True)
        
        # --- Whisper Model Setup (for high-accuracy dictation) ---
        self.status_callback("Status: Loading speech models...")
        print("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = self.device == "cuda"
        print(f"Using device: {self.device} (FP16: {self.fp16})")
        self.model = whisper.load_model("medium")
        print("Whisper model loaded.")
        self.status_callback("Status: Waiting for wake word or voice command...")

        # --- Audio Streaming Setup ---
        self.blocksize = 4096 
        self.q = queue.Queue()
        self.audio_buffer = []
        
        # --- Initialize tkinter for dialog boxes ---
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
        # Configure pyautogui for immediate typing and control
        pyautogui.PAUSE = 0  # Remove delay between keystrokes
        pyautogui.FAILSAFE = True  # Safety feature - move mouse to corner to stop
        
        # Print available commands on startup
        self._print_available_commands()
        
    def _print_available_commands(self):
        """Print all available voice commands."""
        print("\n" + "="*80)
        print("🎤 ENHANCED SPEECH COMMANDER - AVAILABLE VOICE COMMANDS")
        print("="*80)
        print("📝 DICTATION COMMANDS:")
        print("   • 'start typing' / 'begin typing' - Start dictation")
        print("   • 'stop typing' / 'end typing' - Stop dictation")
        print("\n🌐 BROWSER TAB MANAGEMENT:")
        print("   • 'close tab' - Close current tab")
        print("   • 'new tab' - Open new tab")
        print("   • 'next tab' / 'previous tab' - Navigate between tabs")
        print("   • 'reopen tab' - Reopen last closed tab")
        print("   • 'duplicate tab' - Duplicate current tab")
        print("\n🪟 WINDOW MANAGEMENT:")
        print("   • 'new window' / 'close window' - Manage windows")
        print("   • 'minimize window' / 'maximize window' - Window states")
        print("   • 'switch window' - Switch between windows")
        print("\n🧭 NAVIGATION:")
        print("   • 'go back' / 'go forward' - Browser navigation")
        print("   • 'refresh page' / 'reload' - Refresh current page")
        print("   • 'home page' - Go to home page")
        print("   • 'open bookmarks' - Open bookmarks")
        print("\n🔧 BROWSER TOOLS:")
        print("   • 'open incognito' / 'open private' - Private browsing")
        print("   • 'developer tools' - Open dev tools")
        print("   • 'full screen' - Toggle fullscreen")
        print("   • 'zoom in' / 'zoom out' / 'zoom reset' - Zoom controls")
        print("\n🚀 APPLICATION CONTROL:")
        print("   • 'open chrome' / 'open firefox' / 'open edge' - Open browsers")
        print("   • 'open notepad' / 'open calculator' - Open applications")
        print("   • 'open file explorer' / 'open task manager' - System apps")
        print("\n💻 SYSTEM CONTROL:")
        print("   • 'alt tab' - Switch between applications")
        print("   • 'show desktop' - Show/hide desktop")
        print("   • 'take screenshot' - Capture screen")
        print("   • 'lock screen' - Lock computer")
        print("="*80 + "\n")
        
    # === BROWSER CONTROL METHODS ===
    
    def _close_tab(self):
        """Close current browser tab."""
        print("🗂️ Closing current tab...")
        pyautogui.hotkey('ctrl', 'w')
        self.status_callback("✓ Closed tab")
        
    def _new_tab(self):
        """Open new browser tab."""
        print("📄 Opening new tab...")
        pyautogui.hotkey('ctrl', 't')
        self.status_callback("✓ Opened new tab")
        
    def _next_tab(self):
        """Switch to next tab."""
        print("➡️ Switching to next tab...")
        pyautogui.hotkey('ctrl', 'tab')
        self.status_callback("✓ Switched to next tab")
        
    def _previous_tab(self):
        """Switch to previous tab."""
        print("⬅️ Switching to previous tab...")
        pyautogui.hotkey('ctrl', 'shift', 'tab')
        self.status_callback("✓ Switched to previous tab")
        
    def _reopen_tab(self):
        """Reopen last closed tab."""
        print("🔄 Reopening last closed tab...")
        pyautogui.hotkey('ctrl', 'shift', 't')
        self.status_callback("✓ Reopened last closed tab")
        
    def _duplicate_tab(self):
        """Duplicate current tab."""
        print("📋 Duplicating current tab...")
        pyautogui.hotkey('ctrl', 'l')  # Select address bar
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'c')  # Copy URL
        pyautogui.hotkey('ctrl', 't')  # New tab
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')  # Paste URL
        pyautogui.press('enter')
        self.status_callback("✓ Duplicated tab")
        
    def _new_window(self):
        """Open new browser window."""
        print("🪟 Opening new window...")
        pyautogui.hotkey('ctrl', 'n')
        self.status_callback("✓ Opened new window")
        
    def _close_window(self):
        """Close current browser window."""
        print("❌ Closing current window...")
        pyautogui.hotkey('ctrl', 'shift', 'w')
        self.status_callback("✓ Closed window")
        
    def _minimize_window(self):
        """Minimize current window."""
        print("⬇️ Minimizing window...")
        pyautogui.hotkey('win', 'down')
        self.status_callback("✓ Minimized window")
        
    def _maximize_window(self):
        """Maximize current window."""
        print("⬆️ Maximizing window...")
        pyautogui.hotkey('win', 'up')
        self.status_callback("✓ Maximized window")
        
    def _switch_window(self):
        """Switch between windows of the same application."""
        print("🔄 Switching window...")
        pyautogui.hotkey('alt', 'tab')
        self.status_callback("✓ Switched window")
        
    # === NAVIGATION METHODS ===
    
    def _go_back(self):
        """Go back in browser history."""
        print("⬅️ Going back...")
        pyautogui.hotkey('alt', 'left')
        self.status_callback("✓ Went back")
        
    def _go_forward(self):
        """Go forward in browser history."""
        print("➡️ Going forward...")
        pyautogui.hotkey('alt', 'right')
        self.status_callback("✓ Went forward")
        
    def _refresh_page(self):
        """Refresh current page."""
        print("🔄 Refreshing page...")
        pyautogui.hotkey('ctrl', 'r')
        self.status_callback("✓ Page refreshed")
        
    def _go_home(self):
        """Go to home page."""
        print("🏠 Going to home page...")
        pyautogui.hotkey('alt', 'home')
        self.status_callback("✓ Went to home page")
        
    def _open_bookmarks(self):
        """Open bookmarks."""
        print("📚 Opening bookmarks...")
        pyautogui.hotkey('ctrl', 'shift', 'b')
        self.status_callback("✓ Toggled bookmarks")
        
    # === BROWSER TOOLS ===
    
    def _open_incognito(self):
        """Open incognito/private window."""
        print("🕵️ Opening incognito window...")
        pyautogui.hotkey('ctrl', 'shift', 'n')
        self.status_callback("✓ Opened incognito window")
        
    def _open_dev_tools(self):
        """Open developer tools."""
        print("🔧 Opening developer tools...")
        pyautogui.press('f12')
        self.status_callback("✓ Toggled developer tools")
        
    def _view_source(self):
        """View page source."""
        print("📄 Viewing page source...")
        pyautogui.hotkey('ctrl', 'u')
        self.status_callback("✓ Opened page source")
        
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        print("🖥️ Toggling fullscreen...")
        pyautogui.press('f11')
        self.status_callback("✓ Toggled fullscreen")
        
    def _zoom_in(self):
        """Zoom in."""
        print("🔍 Zooming in...")
        pyautogui.hotkey('ctrl', 'plus')
        self.status_callback("✓ Zoomed in")
        
    def _zoom_out(self):
        """Zoom out."""
        print("🔍 Zooming out...")
        pyautogui.hotkey('ctrl', 'minus')
        self.status_callback("✓ Zoomed out")
        
    def _zoom_reset(self):
        """Reset zoom to 100%."""
        print("🔍 Resetting zoom...")
        pyautogui.hotkey('ctrl', '0')
        self.status_callback("✓ Reset zoom")
        
    # === APPLICATION CONTROL ===
    
    def _open_application(self, app_name):
        """Open specified application."""
        app_commands = {
            'chrome': ['chrome', 'google-chrome', 'chrome.exe'],
            'firefox': ['firefox', 'firefox.exe'],
            'msedge': ['msedge', 'msedge.exe', 'edge'],
            'notepad': ['notepad', 'notepad.exe'],
            'calc': ['calc', 'calc.exe'],
            'explorer': ['explorer', 'explorer.exe']
        }
        
        print(f"🚀 Opening {app_name}...")
        
        try:
            if sys.platform == "win32":
                if app_name in app_commands:
                    for cmd in app_commands[app_name]:
                        try:
                            subprocess.Popen([cmd])
                            self.status_callback(f"✓ Opened {app_name}")
                            return
                        except FileNotFoundError:
                            continue
                # Fallback to Windows run dialog
                pyautogui.hotkey('win', 'r')
                time.sleep(0.2)
                pyautogui.typewrite(app_name)
                pyautogui.press('enter')
                
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(['open', '-a', app_name])
            else:  # Linux
                subprocess.Popen([app_name])
                
            self.status_callback(f"✓ Attempted to open {app_name}")
            
        except Exception as e:
            print(f"Error opening {app_name}: {e}")
            self.status_callback(f"❌ Failed to open {app_name}")
            
    def _open_task_manager(self):
        """Open Task Manager."""
        print("📊 Opening Task Manager...")
        if sys.platform == "win32":
            pyautogui.hotkey('ctrl', 'shift', 'esc')
        else:
            # For non-Windows systems, try to open system monitor
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(['open', '-a', 'Activity Monitor'])
                else:
                    subprocess.Popen(['gnome-system-monitor'])
            except:
                pass
        self.status_callback("✓ Opened Task Manager")
        
    # === SYSTEM CONTROL ===
    
    def _alt_tab(self):
        """Alt+Tab to switch applications."""
        print("🔄 Alt+Tab switching...")
        pyautogui.hotkey('alt', 'tab')
        self.status_callback("✓ Alt+Tab")
        
    def _show_desktop(self):
        """Show desktop."""
        print("🖥️ Showing desktop...")
        pyautogui.hotkey('win', 'd')
        self.status_callback("✓ Showed desktop")
        
    def _lock_screen(self):
        """Lock the screen."""
        print("🔒 Locking screen...")
        if sys.platform == "win32":
            pyautogui.hotkey('win', 'l')
        elif sys.platform == "darwin":
            pyautogui.hotkey('cmd', 'ctrl', 'q')
        else:
            # Linux - varies by desktop environment
            try:
                subprocess.Popen(['gnome-screensaver-command', '-l'])
            except:
                pass
        self.status_callback("✓ Screen locked")
        
    def _take_screenshot(self):
        """Take a screenshot."""
        print("📸 Taking screenshot...")
        if sys.platform == "win32":
            pyautogui.hotkey('win', 'shift', 's')  # Windows Snipping Tool
        elif sys.platform == "darwin":
            pyautogui.hotkey('cmd', 'shift', '3')  # macOS screenshot
        else:
            pyautogui.hotkey('prtsc')  # Linux
        self.status_callback("✓ Screenshot taken")
        
    def _open_start_menu(self):
        """Open start menu."""
        print("📋 Opening start menu...")
        pyautogui.press('win')
        self.status_callback("✓ Opened start menu")
        
    # === EXISTING DICTATION METHODS (Enhanced) ===
    
    def _contains_wake_phrase(self, text):
        """Check if text contains any wake phrase."""
        text_lower = text.lower().strip()
        for phrase in self.wake_phrases:
            if phrase in text_lower:
                return True
        return False
    
    def _contains_stop_phrase(self, text):
        """Check if text contains any stop phrase."""
        text_lower = text.lower().strip()
        for phrase in self.stop_phrases:
            if phrase in text_lower:
                return True
        return False
        
    def _check_browser_command(self, text):
        """Check if text contains a browser command and execute it."""
        text_lower = text.lower().strip()
        
        # Check for exact matches first
        if text_lower in self.browser_commands:
            print(f"🎯 Executing command: '{text_lower}'")
            self.browser_commands[text_lower]()
            return True
            
        # Check for partial matches
        for command, function in self.browser_commands.items():
            if command in text_lower:
                print(f"🎯 Executing command: '{command}' (matched from '{text_lower}')")
                function()
                return True
                
        return False
        
    def _audio_callback(self, indata, frames, time, status):
        """This is called for each audio block from the microphone."""
        if status:
            print(status, flush=True)
        self.q.put(bytes(indata))

    def _show_confirmation_dialog(self, text):
        """Shows a confirmation dialog and returns True if user wants to type the text."""
        # Bring the dialog to front and make it stay on top
        self.root.deiconify()
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        
        # Show the confirmation dialog with clear message
        result = messagebox.askyesno(
            "Confirm Text to Type", 
            f"Is this what you want to type where your cursor is?\n\n\"{text}\"\n\nClick YES to type it at cursor position\nClick NO to re-listen and try again",
            parent=self.root
        )
        
        # Hide the root window again but keep it available
        self.root.withdraw()
        self.root.attributes('-topmost', False)
        
        return result

    def _get_audio_up_to_stop_phrase(self):
        """Extract audio buffer up to the point where stop phrase was detected."""
        if not self.audio_segments:
            return b""
        
        if self.last_stop_detection_time is None:
            # No stop phrase detected, return all audio
            return b"".join([segment[0] for segment in self.audio_segments])
        
        # Calculate how much audio to include (up to stop phrase detection)
        valid_audio = []
        current_time = self.buffer_start_time
        bytes_per_second = self.samplerate * 2  # 16-bit = 2 bytes per sample
        bytes_per_block = self.blocksize * 2
        time_per_block = bytes_per_block / bytes_per_second
        
        for audio_data, timestamp in self.audio_segments:
            if current_time <= self.last_stop_detection_time:
                valid_audio.append(audio_data)
            else:
                # This audio block contains or comes after the stop phrase
                # Calculate how much of this block to include
                time_into_block = self.last_stop_detection_time - current_time
                if time_into_block > 0:
                    # Include partial block up to stop phrase
                    samples_to_include = int(time_into_block * self.samplerate)
                    bytes_to_include = samples_to_include * 2  # 2 bytes per sample
                    bytes_to_include = min(bytes_to_include, len(audio_data))
                    if bytes_to_include > 0:
                        valid_audio.append(audio_data[:bytes_to_include])
                break
            current_time += time_per_block
        
        return b"".join(valid_audio)

    def _process_whisper_buffer(self):
        """Processes the collected audio buffer with the Whisper model, excluding audio after stop phrase."""
        if not self.audio_segments:
            return

        # Get audio up to stop phrase detection point
        full_audio_bytes = self._get_audio_up_to_stop_phrase()
        
        # Clear the buffer
        self.audio_segments.clear()
        self.buffer_start_time = None
        self.last_stop_detection_time = None
        
        if not full_audio_bytes:
            print("No valid audio to transcribe (stop phrase detected immediately)")
            self.mode = 'WAITING'
            self.status_callback("Status: Waiting for wake word or voice command...")
            return
        
        try:
            audio_np = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if audio_np.size == 0:
                return

            print(f"Transcribing {len(audio_np)/self.samplerate:.2f} seconds of audio (up to stop phrase)...")
            result = self.model.transcribe(audio_np, language='en', fp16=self.fp16, without_timestamps=True)
            text = result.get('text', '').strip()

            # Clean up the transcribed text by removing stop phrases
            cleaned_text = text
            for stop_phrase in self.stop_phrases:
                cleaned_text = cleaned_text.lower().replace(stop_phrase.lower(), "").strip()
            
            # Remove any remaining common stop words that might have been picked up
            stop_words_to_remove = ["stop", "typing", "end"]
            words = cleaned_text.split()
            cleaned_words = [word for word in words if word.lower() not in stop_words_to_remove]
            cleaned_text = " ".join(cleaned_words).strip()

            if cleaned_text:
                print(f"Transcribed (cleaned): '{cleaned_text}'")
                self.status_callback(f"Transcribed: {cleaned_text}")
                
                if self._show_confirmation_dialog(cleaned_text):
                    # Give the user a clear instruction and a 3-second window to switch focus
                    print("\n!!! You have 3 seconds to click on the window where you want to type !!!\n")
                    self.status_callback("Click where you want to type...")
                    time.sleep(3) # Wait for 3 seconds
                    
                    # User clicked YES - type the text exactly where cursor is
                    print(f"✓ Typing at cursor: '{cleaned_text}'")
                    pyautogui.typewrite(cleaned_text, interval=0.01)
                    
                    self.status_callback(f"✓ Typed: {cleaned_text}")
                    self.mode = 'WAITING'
                    self.status_callback("Status: Waiting for wake word or voice command...")
                else:
                    # User clicked NO - go back to dictating mode for re-listening
                    print("✗ User chose to re-listen. Starting typing again...")
                    self.status_callback("Status: Re-listening... (say 'stop typing' when done)")
                    self.mode = 'DICTATING'
                    self.audio_segments.clear()
                    self.buffer_start_time = None
                    self.last_stop_detection_time = None
            else:
                print("No valid speech detected after cleaning")
                self.status_callback("No speech detected, waiting for wake word or voice command...")
                self.mode = 'WAITING'
                self.status_callback("Status: Waiting for wake word or voice command...")
                
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            self.status_callback(f"Error: {e}")
            self.mode = 'WAITING'
            self.status_callback("Status: Waiting for wake word or voice command...")

    def run(self):
        """Main loop for the enhanced speech commander."""
        print("Enhanced Speech Commander thread started.")
        print("🎯 Say 'START TYPING' clearly to begin dictation!")
        print("🎯 Or use any of the voice commands listed above!")
        print("🛑 Say 'STOP TYPING' to end recording")
        print("=" * 80)
        
        mic_config = {'device': 1, 'channels': 1, 'dtype': 'int16'}
        
        try:
            with sd.RawInputStream(samplerate=self.samplerate,
                                   device=mic_config['device'],
                                   channels=mic_config['channels'],
                                   dtype=mic_config['dtype'],
                                   blocksize=self.blocksize,
                                   callback=self._audio_callback):
                
                print(f">>> Audio stream opened successfully with config: {mic_config} <<<")
                
                while not self.stop_event.is_set():
                    try:
                        audio_data = self.q.get(timeout=self.silence_duration)
                        current_time = time.time()
                        
                        # Process with main recognizer for wake/stop/command detection
                        if self.recognizer.AcceptWaveform(audio_data):
                            result_text = self.recognizer.Result()
                            text = json.loads(result_text).get("text", "")
                            
                            if text.strip() and len(text.strip()) > 2:
                                print(f"Vosk heard: '{text}' (Mode: {self.mode})")
                                
                                # Check for browser/system commands first (works in any mode)
                                if self._check_browser_command(text):
                                    continue
                                    
                                # Then check for dictation commands
                                elif self._contains_wake_phrase(text) and self.mode == 'WAITING':
                                    print("🎤 WAKE PHRASE DETECTED! Starting to record...")
                                    self.mode = 'DICTATING'
                                    self.status_callback("Status: 🔴 RECORDING... (say 'stop typing' when done)")
                                    self.audio_segments.clear()
                                    self.buffer_start_time = current_time
                                    self.last_stop_detection_time = None
                                
                                elif self._contains_stop_phrase(text) and self.mode == 'DICTATING':
                                    print("🛑 STOP PHRASE DETECTED! Processing recorded speech...")
                                    self.last_stop_detection_time = current_time
                                    self._process_whisper_buffer()
                        
                        # Store audio data with timestamp if we're dictating
                        if self.mode == 'DICTATING':
                            self.audio_segments.append((audio_data, current_time))
                            
                            # Limit buffer size to prevent memory issues (keep last 30 seconds)
                            max_segments = int(30 * self.samplerate / self.blocksize)
                            if len(self.audio_segments) > max_segments:
                                self.audio_segments.popleft()
                                # Adjust buffer start time
                                if self.audio_segments:
                                    self.buffer_start_time = self.audio_segments[0][1]

                    except queue.Empty:
                        if self.mode == 'DICTATING':
                            print("--- Silence detected! Processing recorded speech. ---")
                            self._process_whisper_buffer()
                        continue
                    except Exception as e:
                        print(f"An error occurred in speech loop: {e}")
                        break
                        
        except Exception as e:
            print(f"Could not open audio stream: {e}")

        print("Enhanced Speech Commander thread finished.")

    def cleanup(self):
        """Clean up tkinter resources."""
        try:
            if self.root:
                self.root.destroy()
        except:
            pass

if __name__ == '__main__':
    print("Running speech_commander.py directly for testing...")
    stop_event = threading.Event()
    status_callback = lambda text: print(f"STATUS: {text}")
    
    commander = None
    try:
        commander = EnhancedSpeechCommander(stop_event, status_callback)  # ✅ FIXED
        commander.run()
    except KeyboardInterrupt:
        print("\nStopping test.")
        stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if commander:
            commander.cleanup()


def run_speech_commander(stop_event, status_callback):
    commander = EnhancedSpeechCommander(stop_event, status_callback)  # ✅ FIXED
    commander.run()
