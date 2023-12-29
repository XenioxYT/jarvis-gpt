from flask import Flask, jsonify, request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from flask_apscheduler import APScheduler
import os
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from queue import Queue
from werkzeug.utils import secure_filename


app = Flask(__name__)
scheduler = APScheduler()
UPLOAD_FOLDER = './api_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
download_url = None

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipeline_model = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe_audio(filename):
    result = pipeline_model(filename)
    transcription = result["text"]
    return transcription


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav']


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


@app.route('/transcribe', methods=['POST'])
def transcribe_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        transcription = transcribe_audio(file_path)
        return jsonify({"transcription": transcription})


if __name__ == '__main__':
    download_bbc_news_summary()  # Call the function once before starting the Flask application
    scheduler.add_job(id='Scheduled task', func=scheduled_task, trigger='interval', minutes=30)
    scheduler.start()
    app.run(host='0.0.0.0', port=9445)
