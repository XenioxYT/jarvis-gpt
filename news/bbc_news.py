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
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set the directory for downloading files
    download_dir = os.getcwd()  # Get the current working directory
    prefs = {"download.default_directory": download_dir,
             "download.prompt_for_download": False,
             "download.directory_upgrade": True,
             "safebrowsing.enabled": True}
    chrome_options.add_experimental_option("prefs", prefs)

    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Open the BBC News Summary page
        driver.get('https://www.bbc.co.uk/programmes/p002vsn1/episodes/player')

        # Wait for the page to load and the list of episodes to appear
        latest_episode = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.programme__titles a'))
        )

        # Click on the latest episode
        latest_episode.click()

        # Wait for the new page to load and the download link to appear
        download_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-bbc-title="cta_download"]'))
        )

        # Get the URL for the higher quality download
        download_url = download_link.get_attribute('href')

        # If the URL is relative, prepend the domain to it
        if not download_url.startswith('https:') or download_url.startswith('http:'):
            download_url = 'https:' + download_url

        # Use Python requests to download the file
        response = requests.get(download_url)

        # Save the file to the desired location
        with open(os.path.join(download_dir, 'bbc_news_summary.mp3'), 'wb') as file:
            file.write(response.content)
    finally:
        # Close the WebDriver
        driver.quit()
        return "Downloaded BBC News Summary. It will play after your next response."

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

# stop_event = threading.Event()
# convert_and_play_mp3('bbc_news_summary.mp3', stop_event)
