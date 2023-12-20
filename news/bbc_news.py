from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import pygame
import threading

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

def play_mp3(stop_event):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load('./bbc_news_summary.mp3')
    # Play the MP3 file
    pygame.mixer.music.play()
    
    # While the music is playing, check for the stop event
    while pygame.mixer.music.get_busy():
        if stop_event.is_set():
            pygame.mixer.music.stop()  # Stop the music if event is set
            break
        pygame.time.delay(100)  # Wait a short time and check again

    pygame.mixer.quit()  # Quit the mixer when done

def threaded_playback(path_to_mp3):
    # Create a thread to play the MP3 file
    playback_thread = threading.Thread(target=play_mp3, args=(path_to_mp3,))
    # Start the playback thread
    playback_thread.start()

# Call the function
download_bbc_news_summary()
