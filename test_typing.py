# Save this as test_typing.py
import pyautogui
import time

print("Starting the typing test in 5 seconds...")
print("!!! CLICK ON NOTEPAD NOW TO MAKE IT THE ACTIVE WINDOW !!!")

# A 5-second delay to give you time to switch to Notepad
time.sleep(5)

# The script will try to type this text
pyautogui.typewrite("Hello, if you see this, the typing script is working!", interval=0.05)

print("Test finished.")