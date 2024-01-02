import time
from threading import Timer
import simpleaudio as sa

def timer_notification(title, sound_file):
    print(f"Timer '{title}' finished!")
    sa.WaveObject.from_wave_file('./utils/thinking.wav').play().wait_done()

def start_timer(duration, title, sound_file):
    """
    Starts a timer for a specified duration in seconds with a given title.

    Args:
    duration (int): The duration for the timer in seconds.
    title (str): The title of the timer.
    sound_file (str): The path to the WAV file to play when the timer ends.
    """
    print(f"Timer '{title}' started for {duration} seconds.")
    Timer(duration, lambda: timer_notification(title, sound_file)).start()

# Example of usage
# start_timer(1, "Break Timer", "./thinking.wav")