import sounddevice as sd

print("Querying for audio devices...")
try:
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    print("--------------------------")
    for i, device in enumerate(devices):
        # Check for input channels
        if device['max_input_channels'] > 0:
            # I've added the Max Channels information to the output
            print(f"Input Device ID {i}: {device['name']} (Max Channels: {int(device['max_input_channels'])})")
    
    print("\n--------------------------")
    print("Your default input device is:")
    default_input_id = sd.default.device[0]
    default_device = devices[default_input_id]
    print(f"--> Device ID {default_input_id}: {default_device['name']} (Max Channels: {int(default_device['max_input_channels'])})")

except Exception as e:
    print(f"An error occurred: {e}")