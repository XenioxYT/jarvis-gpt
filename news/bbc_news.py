import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import pyaudio
import io
from pydub import AudioSegment
import time
import numpy as np

def download_bbc_news_summary():
    # Set the directory for downloading files
    download_dir = os.getcwd()  # Get the current working directory

    # Get the download URL from the Flask API
    response = requests.get('http://192.168.1.157:9445/download-bbc-news-summary')
    if response.status_code == 200:
        download_url = response.json().get('download_url')

        # Use Python requests to download the file
        response = requests.get(download_url)

        # Save the file to the desired location
        with open(os.path.join(download_dir, 'bbc_news_summary.mp3'), 'wb') as file:
            file.write(response.content)
        return "Downloaded BBC News Summary. It will play after your response. If the user would like to interrupt the playback, they can say your name (however, do not mention your name)."
    else:
        return "Unable to retrieve the download URL"

def convert_and_play_mp3(path_to_mp3, stop_event):
    global current_playback

    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(path_to_mp3)
    audio = audio.set_frame_rate(24000).set_channels(1)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)

    # Apply fade-in to the audio
    fade_in_duration = 0.2
    fade_in_samples = int(fade_in_duration * 24000)
    fade_in_curve = np.linspace(0, 1, fade_in_samples)
    audio_data[:fade_in_samples] *= fade_in_curve.astype(audio_data.dtype)

    # Convert the numpy array back to bytes
    modified_audio_content = audio_data.tobytes()
    audio_buffer = io.BytesIO(modified_audio_content)

    # Callback function to play audio
    def callback(in_data, frame_count, time_info, status):
        if stop_event.is_set():
            return (None, pyaudio.paComplete)
        data = audio_buffer.read(frame_count * 2)
        return (data, pyaudio.paContinue if data else pyaudio.paComplete)
    
    # Open a pyaudio stream and start playback
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2), channels=1, rate=24000, output=True, stream_callback=callback)
    stream.start_stream()

    # Keep the stream active until it's stopped or the stop event is set
    while stream.is_active() and not stop_event.is_set():
        time.sleep(0.1)

    # Clean up the stream and audio resources
    stream.stop_stream()
    stream.close()
    audio_buffer.close()
    p.terminate()

    # Clear the stop event for the next playback
    stop_event.clear()


# download_bbc_news_summary()
# stop_event = threading.Event()
# convert_and_play_mp3('bbc_news_summary.mp3', stop_event)
