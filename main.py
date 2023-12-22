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
import spacy

# imports for the tools
from utils.tools import tools
from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders, load_reminders, save_reminders
from utils.weather import get_weather_data
from calendar_utils import check_calendar, add_event_to_calendar
from utils.home_assistant import toggle_entity
from utils.spotify import search_spotify_song, toggle_spotify_playback, is_spotify_playing_on_device, play_spotify, \
    pause_spotify
from utils.store_conversation import store_conversation, check_conversation
from pveagle_speaker_identification import enroll_user, determine_speaker
from noise_reduction import reduce_noise_and_normalize
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils.google_search import google_search
from news.bbc_news import download_bbc_news_summary, convert_and_play_mp3
from utils.predict_intent import predict_intent
from utils.send_to_discord import send_message_sync
from utils.volume_control import volume_up, volume_down
from utils.notes import save_note, retrieve_notes, edit_or_delete_notes
from utils.initialise_conversation import initialize_conversation


# Profiling
# from scalene import scalene_profiler

# scalene_profiler.start()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

load_dotenv()

thinking_sound_stop_event = threading.Event()

current_playback = None
bbc_news_thread = False

try:
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading language model for the spaCy POS tagger")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

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

# if not check_conversation(1):
#     messages = initialize_conversation(1)
    
# else: 
#     messages = check_conversation(1)

def get_location():
    api_token = os.getenv('IPINFO_TOKEN')
    headers = {
        'Authorization': 'Bearer ' + api_token
    }
    response = requests.get('https://ipinfo.io', headers=headers)
    data = response.json()
    return data.get('city', 'Unknown')


city = get_location()

messages = [
    {
        "role": "system",
        "content": "You are Jarvis, a voice-based personal assistant currently located in " + city + " and based off the GPT-4 AI model. You are speaking to the user now. "
        "The user that activated you is provded to you at the start of each message ('At [timestamp] [user] said:'), along with the date at time. Refer to them by their name. If the user is 'Unknown', then say you don't recognize them, however continue with the action if it is non-personal. "
        "ONLY perform actions for verified users. DO NOT perform reminders or calendar management actions for 'Unknown' users. Some users require specific actions. For example, be sure to select the correct calendar/reminders/smart home control for the specific user mentioned. "
        "You can enroll users using the function. However, BEFORE using this function you MUST give the user a sentence to say, AND ask their name. For example: 'Tell me the weather... [name]'. Insert this name into the correct field. This is to train the model to recognize the user's voice. "
        "Make the sentence you give one that they will ask you, for example 'Tell me the weather in'" + city + "'. It doesn't have to match exactly, but it should be similar. "
        "Keep repeating this process (sentence, function), until the user's voice is recognized. "
        "You are a voice assistant, so keep responses short and concise, but maintain all the important information. Remember that some words may be spelled incorrectly due to speech-to-text errors, so keep this in mind when responding. "
        "You are equipped with a variety of tools, which you can use to perform various tasks. Do not mention you are a text-based assistant. "
        "You can send responses to the user's phone. For example, you can use this to send a summary of events from the calendar, a list of reminders or notes, or links to events/search results. Offer this option to the user. "
        "Since you are a voice assistant, you must remember to not include visual things, like text formatting, as this will not play well with TTS. "
        "Before using a tool, you must first give a response to the user AND THEN call the correct tool or tools. "
        "You CANNOT call a function after giving a text response, so DO NOT say thing like 'Please hold on for a moment', instead ask the user whether they'd like you to continue. "
        "You are allowed to give opinions and thoughts to the user. Don't respond with lists of things, instead give a concise overview and ask the user if they'd like to hear more. If a list is needed, provide it more conversationally. "
        "When giving calendar events, you should give a very concise overview, and ask the user if they'd like to hear more. Don't just list them all out. "
        "ALWAYS check the calendar, weather, etc. before giving a response that includes this. Do NOT hallucinate or make up events without checking. "
        "The date and time is provided at the beginning of the message. This indicates the current date and time, and is used to give you a reference point. "
        "Use this as well to give a sense of time passing and time-contextual responses. "
        "The current date and time is: " + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
    },
]


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
device = "cuda" if torch.cuda.is_available() else "cpu"

SCOPES = ['https://www.googleapis.com/auth/calendar']

REMINDERS_DB_FILE = 'reminders.json'

porcupine = pvporcupine.create(access_key=pv_access_key, keywords=["jarvis"])

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

# koala = pvkoala.create(access_key=pv_access_key)
# cheetah = pvcheetah.create(access_key=pv_access_key, enable_automatic_punctuation=True)


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


# Initialize PyAudio
pa = pyaudio.PyAudio()


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

    audio_files = [f'./user_dataset_temp/{name}/{file}' for file in os.listdir(f'./user_dataset_temp/{name}')]

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


def transcribe(queue, filename='temp.wav', pipeline_model=pipeline_model):
    if pipeline_model is None:
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
    result = pipeline_model(filename)
    transciption = result["text"]
    queue.put(transciption)



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


def text_to_speech_thread(text):
    # This function will run in a separate thread
    text_to_speech(text)

# Function to get response from ChatGPT, making any necessary tool calls
def get_chatgpt_response(text, function=False, function_name=None, cursor=None, db_conn=None, speaker="Unknown"):
    
    if not speaker == "Unknown":
        enroll_user_thread = threading.Thread(target=enroll_user_handler, args=(speaker,))
        enroll_user_thread.start()
    
    global bbc_news_thread
    if function:
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": text,
            }
        )
    timestamp = datetime.datetime.now().strftime("%H:%M on %a %d %B %Y")

    messages.append({"role": "user", "content": f"At {timestamp} {speaker} said: {text}"})
    if cursor:
        store_conversation(1, messages, cursor, db_conn)
    else:
        store_conversation(1, messages)

    intent = predict_intent(text, nlp)
    print(intent)
    completion = ""
    full_completion = ""
    full_completion_2 = ""
    first_sentence_processed = False
    first_sentence_processed_second_response = False
    waiting_for_number = False
    waiting_for_number_second_response = False
    tool_calls = []
    if intent == None:

        # Send the initial message and the available tool to the model
        response = oai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=tools,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content or delta.content == '':
                completion += chunk.choices[0].delta.content
                # print(completion)
                full_completion += chunk.choices[0].delta.content

                if waiting_for_number and completion[0].isdigit():
                    # Append the number to the previously processed sentence
                    string1 += completion
                    waiting_for_number = False

                elif not first_sentence_processed and any(punctuation in completion for punctuation in ["!", ".", "?"]):
                    if not re.search(r'\d+\.\d+', completion):
                        string1, rest = split_first_sentence(completion)

                        # Check if string1 ends with a pattern like "number."
                        if re.search(r'\d\.$', string1):
                            waiting_for_number = True
                        else:
                            if string1:
                                # Start the text-to-speech function in a separate thread
                                tts_thread = threading.Thread(target=text_to_speech_thread, args=(string1,))
                                tts_thread.start()
                                completion = rest  # Reset completion to contain only the remaining text
                                first_sentence_processed = True

            if chunk.choices[0].delta.tool_calls:
                tcchunklist = delta.tool_calls
                for tcchunk in tcchunklist:
                    if len(tool_calls) <= tcchunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    tc = tool_calls[tcchunk.index]

                    if tcchunk.id:
                        tc["id"] += tcchunk.id
                        # print(tc["id"])
                    if tcchunk.function.name:
                        tc["function"]["name"] += tcchunk.function.name
                        # print(tc["function"]["name"])
                    if tcchunk.function.arguments:
                        tc["function"]["arguments"] += tcchunk.function.arguments
                        # print(tc["function"]["arguments"])
    else: 
        tool_calls.append({"id": "", "type": "function", "function": {"name": intent, "arguments": ""}})
        
    if tool_calls:

        # Dictionary mapping function names to actual function implementations
        available_functions = {
            "get_weather_data": get_weather_data,
            "check_calendar": check_calendar,
            # "google_search": google_search,
            "set_reminder": add_reminder,
            "edit_reminder": edit_reminder,
            "list_reminders": list_unnotified_reminders,
            "add_event_to_calendar": add_event_to_calendar,
            "control_switch": toggle_entity,
            "play_song_on_spotify": search_spotify_song,
            "enroll_user": enroll_user_handler,
            "google_search": google_search,
            "bbc_news_briefing": download_bbc_news_summary,
            "send_to_phone": send_message_sync,
            "volume_up": volume_up,
            "volume_down": volume_down,
            "save_note": save_note,
            "retrieve_notes": retrieve_notes,
            "edit_or_delete_notes": edit_or_delete_notes,
        }

        messages.append(
            {
                "role": "function",
                "name": "tool_calls",
                "content": str(tool_calls),
            }
        )

        # Map function names to their TTS messages
        tts_messages = {
            "play_song_on_spotify": "Connecting to your speakers...",
            "set_reminder": "Accessing reminders...",
            "add_event_to_calendar": "Adding event to your calendar...",
            "get_weather_data": "Getting live weather data...",
            "check_calendar": "Checking your calendar...",
            "control_switch": "Controling your smart home...",
            "edit_reminder": "Editing your reminders...",
            "list_reminders": "Getting your reminders...",
            "enroll_user": "Learning your voice...",
            "google_search": "Searching the web...",
            "bbc_news_briefing": "Getting the latest news...",
            "send_to_phone": "Sending a message to your phone...",
            "volume_up": "Increasing the volume...",
            "volume_down": "Decreasing the volume...",
            "save_note": "Saving your note...",
            "retrieve_notes": "Retrieving your notes...",
            "edit_or_delete_notes": "Editing your notes...",
        }

        multiple_tool_calls = len(tool_calls) > 1
        tts_multiple_spoken = False

        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            try:
                function_args = json.loads(tool_call['function']['arguments'])
            except:
                function_args = {}
            if function_name == "bbc_news_briefing":
                bbc_news_thread = True
            
            if function_name in ["save_note", "edit_or_delete_notes", "retrieve_notes"]:
                function_args["user"] = speaker

            if function_name == "send_to_phone":
                username_mapping = {"Tom": "xeniox"}
                # Update the username if it matches the speaker and is in the mapping
                if function_args["username"] == speaker and speaker in username_mapping:
                    function_args["username"] = username_mapping[speaker]
                    
            
            # Print tool call information
            print(f"Tool call: {tool_call}")
            print(f"Function name: {function_name}", f"Function args: {function_args}")

            # Determine the TTS message based on the function name and whether multiple tools are called
            tts_message = tts_messages.get(function_name, "Connecting to the internet")
            if multiple_tool_calls:
                tts_message = "Accessing multiple tools..."

            # Start the TTS thread
            
            if tts_multiple_spoken == False:
                tts_thread_function = threading.Thread(target=text_to_speech_thread, args=(tts_message,))
                tts_thread_function.start()
                tts_multiple_spoken = True

            # Execute the function if available and store the response
            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                print(function_response)

                # Send the function response back to the model and store the conversation
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                })
                store_conversation(1, messages, cursor, db_conn) if cursor else store_conversation(1, messages)

        try:
            second_response = oai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                stream=True,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            second_response = oai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                stream=True,
            )
        if second_response:
            store_conversation(1, messages)
            completion = ""
            for chunk in second_response:
                delta = chunk.choices[0].delta
                if delta.content or delta.content == '':
                    completion += chunk.choices[0].delta.content
                    # print(completion)
                    full_completion_2 += chunk.choices[0].delta.content
                    try:
                        if tts_thread_function.is_alive():
                            tts_thread_function.join()
                    except:
                        pass

                    if waiting_for_number_second_response and completion[0].isdigit():
                        # Append the number to the previously processed sentence
                        string1 += completion
                        waiting_for_number_second_response = False

                    elif not first_sentence_processed_second_response and any(
                            punctuation in completion for punctuation in ["!", ".", "?"]):
                        if not re.search(r'\d+\.\d+', completion):
                            string1, rest = split_first_sentence(completion)

                            # Check if string1 ends with a pattern like "number."
                            if re.search(r'\d\.$', string1):
                                waiting_for_number_second_response = True
                            else:
                                if string1:
                                    # Start the text-to-speech function in a separate thread
                                    tts_thread = threading.Thread(target=text_to_speech_thread, args=(string1,))
                                    tts_thread.start()
                                    completion = rest  # Reset completion to contain only the remaining text
                                    first_sentence_processed_second_response = True
            messages.append(
                {
                    "role": "assistant",
                    "content": full_completion_2,
                }
            )
            store_conversation(1, messages, cursor, db_conn) if cursor else store_conversation(1, messages)
            # Assume that we return the final response text after the tool call handling
            try:
                if tts_thread.is_alive():
                    tts_thread.join()
            except:
                pass
            return completion
    else:
        messages.append(
            {
                "role": "assistant",
                "content": full_completion,
            }
        )
        if cursor:
            store_conversation(1, messages, cursor, db_conn)
        else:
            store_conversation(1, messages)
        # Return the direct response text when no tool calls are needed
        try:
            if tts_thread.is_alive():
                tts_thread.join()
        except:
            pass
        return completion


# Function to convert text to speech using Google Cloud TTS
def text_to_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Use a British accent voice, for example "en-GB-Wavenet-B"
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-GB',
        name='en-GB-Neural2-B',  # You can choose a different British Wavenet voice if desired
        # ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Use  audio encoding for high quality
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Perform the Text-to-Speech request on the text input with the selected voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    fade_in_duration = 0.2  # Duration of the fade-in effect in seconds
    # Convert the audio content to a numpy array and make it writable
    audio_data = np.frombuffer(response.audio_content, dtype=np.int16).copy()
    fade_in_samples = int(fade_in_duration * 24000)  # Number of samples over which to apply the fade-in
    fade_in_curve = np.linspace(0, 1, fade_in_samples, dtype=np.float64)

    # Apply the fade-in effect
    try:
        for i in range(fade_in_samples):
            audio_data[i] = np.int16(float(audio_data[i]) * fade_in_curve[i])
    except IndexError:
        pass

    # Convert the audio data back to bytes
    modified_audio_content = audio_data.tobytes()

    # First, save the modified audio to a buffer
    audio_buffer = io.BytesIO(modified_audio_content)

    # Then, play the audio buffer using PyAudio
    # Define PyAudio stream callback for asynchronous playback
    def callback(in_data, frame_count, time_info, status):
        data = audio_buffer.read(frame_count * 2)  # 2 bytes per sample for LINEAR16
        return (data, pyaudio.paContinue)

    # Initialize PyAudio and open a stream for playback
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=24000,  # Ensure this matches the sample rate from the TTS response
                    output=True,
                    stream_callback=callback)

    # Start the playback stream and wait for it to finish
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)  # Pause in the loop to reduce CPU usage
    stream.stop_stream()
    stream.close()
    audio_buffer.close()
    p.terminate()


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
            response = get_chatgpt_response(command, speaker=str(user))
            if spotify_was_playing:
                toggle_spotify_playback()
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
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "Stopped playback of BBC News Summary.",
                            }
                        )
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