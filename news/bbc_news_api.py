from flask import Flask, jsonify, request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from flask_apscheduler import APScheduler
import os
import requests

app = Flask(__name__)
scheduler = APScheduler()
download_url = None

def download_bbc_news_summary():
    global download_url
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    download_dir = os.getcwd()
    prefs = {"download.default_directory": download_dir,
             "download.prompt_for_download": False,
             "download.directory_upgrade": True,
             "safebrowsing.enabled": True}
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get('https://www.bbc.co.uk/programmes/p002vsn1/episodes/player')
        latest_episode = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.programme__titles a')))
        latest_episode.click()

        download_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-bbc-title="cta_download"]')))
        download_url = download_link.get_attribute('href')

        if not download_url.startswith('https:') and not download_url.startswith('http:'):
            download_url = 'https:' + download_url

    finally:
        driver.quit()

def scheduled_task():
    download_bbc_news_summary()

@app.route('/download-bbc-news-summary', methods=['GET'])
def download_bbc_news_summary_api():
    global download_url
    if download_url:
        return jsonify({"download_url": download_url})
    else:
        return jsonify({"error": "Unable to retrieve the download URL"}), 500

if __name__ == '__main__':
    download_bbc_news_summary()  # Call the function once before starting the Flask application
    scheduler.add_job(id='Scheduled task', func=scheduled_task, trigger='interval', minutes=30)
    scheduler.start()
    app.run(debug=True)
