import json
import os
import re
import sqlite3
import threading
import time
import pyaudio
import wave
import requests
import numpy as np
import torch
import webrtcvad
import pvporcupine
import shutil
import random
from dotenv import load_dotenv
from openai import OpenAI
import torch
from google.cloud import texttospeech
import io
import datetime
import simpleaudio as sa
from queue import Queue

# imports for the tools
from utils.reminders import load_reminders, save_reminders
from utils.spotify import toggle_spotify_playback
from pveagle_speaker_identification import enroll_user, determine_speaker
from noise_reduction import reduce_noise_and_normalize
from news.bbc_news import convert_and_play_mp3
from testing import generate_response as get_chatgpt_response
from utils.store_conversation import get_conversation, store_conversation
from text_to_speech import text_to_speech

load_dotenv()

thinking_sound_stop_event = threading.Event()

current_playback = None
bbc_news_thread = False


def play_sound(sound_file, loop=False):
    global current_playback
    # Create wave_obj and play_obj once outside the loop
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    play_obj = wave_obj.play()
    def play():
        global current_playback
        while not thinking_sound_stop_event.is_set():
            current_playback = play_obj
            play_obj.wait_done()
            if not loop or thinking_sound_stop_event.is_set():
                break
    threading.Thread(target=play, daemon=True).start()


def stop_thinking_sound():
    global current_playback

    # Set the event to stop future sound replay in the loop
    thinking_sound_stop_event.set()

    # Stop the current playback if it exists
    if current_playback:
        current_playback.stop()

    # Reset the event for the next playback loop
    thinking_sound_stop_event.clear()

    # Clear the current playback reference
    current_playback = None


# Load environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

discord_token = os.environ.get('DISCORD_TOKEN')

oai_client = OpenAI(base_url=api_base, api_key=api_key)
pv_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LISTENING_SOUND = "./sounds/started_listening.wav"
STOPPED_LISTENING_SOUND = "./sounds/stopped_listening.wav"
# THINKING_SOUND = "./sounds/thinking.wav"
SUCCESS_SOUND = "./sounds/success.wav"

client = texttospeech.TextToSpeechClient()

SCOPES = ['https://www.googleapis.com/auth/calendar']

REMINDERS_DB_FILE = 'reminders.json'

porcupine = pvporcupine.create(access_key=pv_access_key, keywords=["jarvis"])

# Initialize PyAudio
pa = pyaudio.PyAudio()


def check_reminders(cursor=None, db_conn=None):
    current_time = datetime.datetime.now().replace(second=0, microsecond=0)
    reminders = load_reminders()
    # print(reminders)

    due_reminders = [r for r in reminders if
                     not r['notified'] and datetime.datetime.fromisoformat(r['time']) == current_time]

    for reminder in due_reminders:
        message = (f"A reminder has been triggered for {reminder['time']} with text: {reminder['text']}. Please "
                   f"deliver this reminder to the user.")
        response = get_chatgpt_response(text=message, cursor=cursor, db_conn=db_conn, function=True,
                                        function_name="speak_reminder")
        text_to_speech(response)

        reminder['notified'] = True  # Mark as notified

    save_reminders(reminders)  # Update the reminders in the database


def reminder_daemon():
    while True:
        db_conn = sqlite3.connect('conversations.db')
        c = db_conn.cursor()
        check_reminders(cursor=c, db_conn=db_conn)
        db_conn.close()
        time.sleep(30)  # Wait for one minute before checking again


def enroll_user_handler(name):
    # Create the destination directory if it doesn't exist
    os.makedirs(f'./user_dataset_temp/{name}', exist_ok=True)

    # Check if the text file exists
    counter_file = f"./user_dataset_temp/{name}/counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            count = int(f.read())
        if count >= 8:
            return "User already enrolled"
        else:
            count += 1
    else:
        count = 1

    # Write the new count to the file
    with open(counter_file, 'w') as f:
        f.write(str(count))

    # Generate a random number
    random_number = random.randint(1, 1000)
    reduce_noise_and_normalize('./temp.wav')

    # Copy and move the file
    destination = f"./user_dataset_temp/{name}/{random_number}.wav"
    shutil.copy("./temp_cleaned_normalised.wav", destination)

    audio_files = [f'./user_dataset_temp/{name}/{file}' for file in os.listdir(f'./user_dataset_temp/{name}') if file.endswith('.wav')]

    return enroll_user(pv_access_key, audio_files, f"./user_models/{name}.pv")


def determine_user_handler(queue):
    # Check if the directory is empty
    if not os.listdir('./user_models/'):
        print("The directory is empty")
        result = "Unknown"
        queue.put(result)
        return "Unknown"
    input_profile_paths = [f'./user_models/{name}' for name in os.listdir('./user_models/')]
    audio_path = './temp_cleaned_normalised.wav'
    result = determine_speaker(access_key=pv_access_key, input_profile_paths=input_profile_paths,
                               test_audio_path=audio_path)
    queue.put(result)
    return "Unknown"


# Function to save the recorded audio to a WAV file
def save_audio(frames, filename='temp.wav'):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))


def transcribe(queue, filename='temp.wav', server_url='https://api.xeniox.tv/transcribe'):
    files = {'file': open(filename, 'rb')}
    response = requests.post(server_url, files=files)

    if response.status_code == 200:
        transcription = response.json().get('transcription', '')
    else:
        transcription = f"Error: Server responded with status code {response.status_code}"

    queue.put(transcription)



def split_first_sentence(text):
    # Look for a period, exclamation mark, or question mark that might indicate the end of a sentence
    match = re.search(r'[.!?]', text)
    if match:
        # Check if the match is likely the end of a sentence
        index = match.start()
        possible_end = text[:index + 1]
        remainder = text[index + 1:]

        # Look ahead to see if the next character is a digit (part of a decimal) or an uppercase letter (start of a new sentence)
        next_char_match = re.search(r'\s*([A-Z]|\d)', remainder)
        if next_char_match and next_char_match.group(1).isupper():
            # It's an uppercase letter, so likely a new sentence
            return possible_end.strip(), remainder.strip()
        else:
            # It's a digit or there's no immediate uppercase letter, so likely not the end of a sentence
            return text, ''
    else:
        return text, ''


def main():
    audio_stream = pa.open(
        rate=16000,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    play_sound(SUCCESS_SOUND)

    vad = webrtcvad.Vad(3)
    volume_boost_factor = 2.5

    print("Say 'Jarvis' to wake up the assistant...")

    while True:
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = np.frombuffer(pcm, dtype='h', count=porcupine.frame_length)

        pcm_boosted = np.multiply(pcm_unpacked, volume_boost_factor).astype(int)
        # pcm_suppressed = koala.process(pcm_unpacked)
        keyword_index = porcupine.process(pcm_boosted)

        if keyword_index >= 0:
            print("Jarvis activated. Listening for your command...")
            spotify_was_playing = False
            # if is_spotify_playing_on_device():
            #     spotify_was_playing = True
            # toggle_spotify_playback()
            play_sound(LISTENING_SOUND)
            time.sleep(0.25)
            accumulated_frames = []
            num_silent_frames = 0
            vad_frame_accumulator = []
            vad_frame_len = int(0.02 * 16000)  # 20 ms

            while True:
                pcm = audio_stream.read(vad_frame_len, exception_on_overflow=False)
                pcm_unpacked = np.frombuffer(pcm, dtype='h', count=vad_frame_len)

                pcm_boosted = np.multiply(pcm_unpacked, volume_boost_factor).astype(int)

                # Apply Koala noise suppression
                # pcm_suppressed = koala.process(pcm_boosted)

                # Accumulate the suppressed frames for a full VAD frame
                vad_frame_accumulator = np.append(vad_frame_accumulator, pcm_boosted)
                # Once enough samples are accumulated for a 20 ms frame, process with VAD
                if len(vad_frame_accumulator) >= vad_frame_len:
                    vad_frame = vad_frame_accumulator[:vad_frame_len]
                    vad_buffer = vad_frame.astype(np.int16).tobytes()
                    is_speech = vad.is_speech(vad_buffer, 16000)

                    # Remove processed samples from the accumulator
                    vad_frame_accumulator = vad_frame_accumulator[vad_frame_len:]

                    if is_speech:
                        num_silent_frames = 0
                    else:
                        num_silent_frames += 1

                    # Accumulate the suppressed frames
                    accumulated_frames.append(vad_buffer)

                    if num_silent_frames > 60:  # Stop capturing after a short period of silence
                        print("Done capturing.")
                        play_sound(STOPPED_LISTENING_SOUND)
                        if spotify_was_playing:
                            toggle_spotify_playback(force_play=True)
                        break

            # Save the suppressed audio
            save_audio(accumulated_frames)
            reduce_noise_and_normalize('./temp.wav')
            print("Processing audio...")
            # Create a queue for each thread to put their result into
            transcribe_queue = Queue()
            user_handler_queue = Queue()

            # Initialize and start threads with the queues as arguments
            transcript_thread = threading.Thread(target=transcribe, args=(transcribe_queue,))
            user_handler_thread = threading.Thread(target=determine_user_handler, args=(user_handler_queue,))
            transcript_thread.start()
            user_handler_thread.start()

            # Wait for threads to finish
            transcript_thread.join()
            print("transcription thread finished!")
            user_handler_thread.join()
            print("determine user thread finished!")

            # Retrieve results from the queues
            command = transcribe_queue.get()
            print(command)
            user = user_handler_queue.get()
            print(user)
            response, tts_thread = get_chatgpt_response(command, speaker=str(user))
            if spotify_was_playing:
                toggle_spotify_playback()
            if tts_thread and tts_thread.is_alive():
                tts_thread.join()
            text_to_speech(response)
            global bbc_news_thread
            stop_event = threading.Event()
            if bbc_news_thread:
                news_thread = threading.Thread(target=convert_and_play_mp3, args=("bbc_news_summary.mp3", stop_event))
                news_thread.start()
                while news_thread.is_alive():
                    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                    pcm_unpacked = np.frombuffer(pcm, dtype='h', count=porcupine.frame_length)
                    pcm_boosted = np.multiply(pcm_unpacked, volume_boost_factor).astype(int)
                    keyword_index = porcupine.process(pcm_boosted)
                    if keyword_index >= 0:
                        stop_event.set()  # If wake word detected, signal to stop TTS playback
                        bbc_news_thread = False
                        text_to_speech("Stopped playback of BBC News Summary.")
                        messages = get_conversation(1)
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "Stopped playback of BBC News Summary.",
                            }
                        )
                        store_conversation(1, messages)
                        bbc_news_thread = False
                        break
                news_thread.join()  # Ensure TTS thread is finished before restarting the loop

            stop_event.clear()  # Clear the stop event for the next speech cycle
            bbc_news_thread = False
            if spotify_was_playing:
                toggle_spotify_playback(force_play=True)

    audio_stream.close()
    pa.terminate()
    porcupine.delete()
    koala.delete()


reminder_daemon_thread = threading.Thread(target=reminder_daemon, daemon=True)
reminder_daemon_thread.start()

if __name__ == '__main__':
    main()