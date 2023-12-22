import subprocess

def volume_up():
    # Get the current volume
    get_volume_cmd = "pactl get-sink-volume @DEFAULT_SINK@"
    current_volume = subprocess.check_output(get_volume_cmd, shell=True).decode()

    # Extract the current volume percentage
    current_volume_percent = int(current_volume.split('/')[1].strip().strip('%'))

    # Calculate the new volume, ensuring it does not exceed 100%
    new_volume = min(100, current_volume_percent + 10)

    # Set the new volume
    set_volume_cmd = f"pactl set-sink-volume @DEFAULT_SINK@ {new_volume}%"
    subprocess.run(set_volume_cmd, shell=True)

    return f"Volume increased to {new_volume}%"
    
    
def volume_down():
    # Get the current volume
    get_volume_cmd = "pactl get-sink-volume @DEFAULT_SINK@"
    current_volume = subprocess.check_output(get_volume_cmd, shell=True).decode()

    # Extract the current volume percentage
    current_volume_percent = int(current_volume.split('/')[1].strip().strip('%'))

    # Calculate the new volume, ensuring it does not go below 0%
    new_volume = max(0, current_volume_percent - 10)

    # Set the new volume
    set_volume_cmd = f"pactl set-sink-volume @DEFAULT_SINK@ {new_volume}%"
    subprocess.run(set_volume_cmd, shell=True)

    return f"Volume decreased to {new_volume}%"