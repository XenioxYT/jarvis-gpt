import json
import os
import re
import sqlite3
import struct
import threading
import time
import pyaudio
import pvkoala
import wave
import requests
import numpy as np
import torch
import webrtcvad
import queue as thread_queue
import pvporcupine
import shutil
import random
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import texttospeech
import pyttsx3
import whisper
import io
import datetime
import simpleaudio as sa

# imports for the tools
from utils.tools import tools
from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders, load_reminders, save_reminders
from utils.weather import get_weather_data
from calendar_utils import check_calendar, add_event_to_calendar
from utils.home_assistant import toggle_entity
from utils.spotify import search_spotify_song, toggle_spotify_playback, is_spotify_playing_on_device, play_spotify, pause_spotify
from classification import is_english_text
from utils.store_conversation import store_conversation
from pveagle_speaker_identification import enroll_user, determine_speaker

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

load_dotenv()

thinking_sound_stop_event = threading.Event()

current_playback = None

def play_sound(sound_file, loop=False):
    global current_playback

    def play():
        global current_playback
        while not thinking_sound_stop_event.is_set():
            wave_obj = sa.WaveObject.from_wave_file(sound_file)
            play_obj = wave_obj.play()
            current_playback = play_obj
            play_obj.wait_done()
            if not loop or thinking_sound_stop_event.is_set():
                break

    # If there's already a sound playing, stop it
    # if current_playback:
    #     current_playback.stop()

    # Start a new thread for the next sound to play
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
        "content": "You are Jarvis, a voice-based personal assistant currently located in " + city + " and based off the GPT-4 AI model. You are speaking to him now. "
        "The user that activated you is provded to you at the start of each message ('At [timestamp] [user] said:'), along with the date at time. Refer to them by their name. If the user is 'Unknown', then say you don't recognize the speaker. ALWAYS check the user before performing any actions. "
        "ONLY perform actions for verified users. DO NOT perform actions for 'Unknown' users. Some users require specific actions. For example, be sure to select the correct calendar/reminders/smart home control for the specific user mentioned. "
        "You can enroll users using the function. However, BEFORE using this function you MUST give the user a sentence of 10 words to say, AND ask their name. For example: 'The quick brown... [name]'. Insert this name into the correct field. This is to train the model to recognize the user's voice. "
        "You are a voice assistant, so keep responses short and concise, but maintain all the important information. Remember that some words may be spelled incorrectly due to speech-to-text errors, so keep this in mind when responding. "
        "You are equipped with a variety of tools, which you can use to perform various tasks. For example, you can play music on spotify for the user. Do not mention you are a text-based assistant. "
        "Since you are a voice assistant, you must remember to not include visual things, like text formatting, as this will not play well with TTS. "
        "You CANNOT call a function after giving a text response, so DO NOT say thing like 'Please hold on for a moment', instead ask the user whether they'd like you to continue. "
        "You are allowed to give opinions and thoughts to the user. Don't respond with lists of things, instead give a concise overview and ask the user if they'd like to hear more. If a list is needed, provide it more conversationally. "
        "When giving calendar events, you should give a very concise overview, and ask the user if they'd like to hear more. Don't just list them all out. "
        "ALWAYS check the calendar, weather, etc. before giving a response that includes this. Do NOT hallucinate or make up events without checking. "
        "The date and time is provided at the beginning of the message. This indicates the current date and time, and is used to give you a reference point. "
        "Use this as well to give a sense of time passing and time-contextual responses. "
        "The current date and time is: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
]

store_conversation(1, messages)

# Load environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

oai_client = OpenAI(base_url=api_base, api_key=api_key)
pv_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LISTENING_SOUND = "./sounds/started_listening.wav"
STOPPED_LISTENING_SOUND = "./sounds/stopped_listening.wav"
THINKING_SOUND = "./sounds/thinking.wav"
SUCCESS_SOUND = "./sounds/success.wav"

client = texttospeech.TextToSpeechClient()
device = "cuda" if torch.cuda.is_available() else "cpu"

SCOPES = ['https://www.googleapis.com/auth/calendar']

REMINDERS_DB_FILE = 'reminders.json'

porcupine = pvporcupine.create(access_key=pv_access_key, keywords=["jarvis"])
koala = pvkoala.create(access_key=pv_access_key)

def check_reminders(cursor=None, db_conn=None):
    current_time = datetime.datetime.now().replace(second=0, microsecond=0)
    reminders = load_reminders()
    # print(reminders)
    
    due_reminders = [r for r in reminders if not r['notified'] and datetime.datetime.fromisoformat(r['time']) == current_time]
    
    for reminder in due_reminders:
        message = f"A reminder has been triggered for {reminder['time']} with text: {reminder['text']}. Please deliver this reminder to the user."
        response = get_chatgpt_response(text=message, cursor=cursor, db_conn=db_conn, function=True, function_name="speak_reminder")
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
    # Generate a random number
    random_number = random.randint(1, 1000)

    # Create the destination directory if it doesn't exist
    os.makedirs(f'./user_dataset_temp/{name}', exist_ok=True)

    # Copy and move the file
    destination = f"./user_dataset_temp/{name}/{random_number}.wav"
    shutil.copy("./temp.wav", destination)
    
    audio_files = [f'./user_dataset_temp/{name}/{file}' for file in os.listdir(f'./user_dataset_temp/{name}')]

    return enroll_user(pv_access_key, audio_files, f"./user_models/{name}.pv")

def determine_user_handler():
    # Check if the directory is empty
    if not os.listdir('./user_models/'):
        print("The directory is empty")
        return "Unknown"  # or handle the error in another appropriate way
    input_profile_paths = [f'./user_models/{name}' for name in os.listdir('./user_models/')]
    audio_path = './temp.wav'
    return determine_speaker(access_key=pv_access_key, input_profile_paths=input_profile_paths, test_audio_path=audio_path)

# Function to continuously capture audio until user stops speaking
def capture_speech(vad, audio_stream):
    print("Listening for your command...")
    frames = []
    num_silent_frames = 0

    while True:
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)  # ensure little-endian

        # Convert frames to 10-ms chunks as required by webrtcvad
        ms_frame = b''.join([struct.pack('h', sample) for sample in pcm_unpacked])

        is_speech = vad.is_speech(ms_frame, 16000)

        if is_speech:
            num_silent_frames = 0
        else:
            num_silent_frames += 1

        frames.append(pcm)

        # Stop capturing after a short period of silence
        if num_silent_frames > 30:
            print("Done capturing.")
            break

    return frames

# Function to save the recorded audio to a WAV file
def save_audio(frames, filename='temp.wav'):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

# Function to transcribe speech to text using Whisper
def transcribe(filename='temp.wav'):
    # model = whisper.load_model("base")  # this is done in the head of the file
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(filename, generate_kwargs={"language": "english"})
    return result["text"]

import re

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

    # Send the initial message and the available tool to the model
    response = oai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        stream=True,
    )
    
    completion = ""
    full_completion = ""
    full_completion_2 = ""
    first_sentence_processed = False
    first_sentence_processed_second_response = False
    waiting_for_number = False
    waiting_for_number_second_response = False
    tool_calls = []

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
                # Continue with text-to-speech and rest of the processing
                # ...
    
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
                    tool_calls.append({"id": "", "type": "function", "function": { "name": "", "arguments": "" } })
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
    if tool_calls:

        # Dictionary mapping function names to actual function implementations
        available_functions = {
            "get_weather_data": get_weather_data,
            "check_calendar": check_calendar,
            # "google_search": google_search,
            "set_reminder": add_reminder,
            "edit_reminder": edit_reminder,
            "list_unnotified_reminders": list_unnotified_reminders,
            "add_event_to_calendar": add_event_to_calendar,
            "control_switch": toggle_entity,
            "play_song_on_spotify": search_spotify_song,
            "enroll_user": enroll_user_handler,
        }
        
        if len(tool_calls) > 1:
            multiple_tool_calls = True
        else:
            multiple_tool_calls = False

        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            
            # messages.append(
            #     {
            #         "role": "assistant",
            #         "content": "You called a function with the following parameters" + function_name + " " + str(function_args),
            #     }
            # )
            if cursor:
                store_conversation(1, messages, cursor, db_conn)
            else:
                store_conversation(1, messages)

            print(f"Tool call: {tool_call}")
            print(f"Function name: {function_name}", f"Function args: {function_args}")
            
            if multiple_tool_calls:
                tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Accessing multiple tools...",))
                tts_thread_function.start()
                
            else:
                if function_name == "play_song_on_spotify":
                    tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Connecting to your speakers...",))
                    tts_thread_function.start()
                elif function_name == "set_reminder":
                    tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Accessing reminders...",))
                    tts_thread_function.start()

                elif function_name == "add_event_to_calendar":
                    event_name = function_args['title']
                    tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Adding " + event_name + " to your calendar...",))
                    tts_thread_function.start()

                elif function_name == "get_weather_data":
                    tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Getting live weather data...",))
                    tts_thread_function.start()

                else:
                    tts_thread_function = threading.Thread(target=text_to_speech_thread, args=("Connecting to the internet",))
                    tts_thread_function.start()

            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                print(function_response)
                
                # Send the function response back to the model
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                if cursor:
                    store_conversation(1, messages, cursor, db_conn)
                else:
                    store_conversation(1, messages)
                    continue
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
                    
                    if tts_thread_function.is_alive():
                        tts_thread_function.join()

                    if waiting_for_number_second_response and completion[0].isdigit():
                        # Append the number to the previously processed sentence
                        string1 += completion
                        waiting_for_number_second_response = False
                        # Continue with text-to-speech and rest of the processing
                        # ...

                    elif not first_sentence_processed_second_response and any(punctuation in completion for punctuation in ["!", ".", "?"]):
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
            if cursor:
                store_conversation(1, messages, cursor, db_conn)
            else:
                store_conversation(1, messages)
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
    
def handle_follow_ups(audio_stream, vad, response):
    # This function will handle follow-up interactions repeatedly
    while True:
        play_sound(LISTENING_SOUND)
        time.sleep(1)
        print("Listening for a follow-up command...")
        accumulated_frames = []
        num_silent_frames = 0
        vad_frame_len = int(0.02 * 16000)

        while num_silent_frames < 50:  # Adjust the threshold as needed
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
            vad_buffer = b''.join(struct.pack('h', frame) for frame in pcm_unpacked)
            is_speech = vad.is_speech(vad_buffer[:2 * vad_frame_len], 16000)

            if is_speech:
                num_silent_frames = 0
                accumulated_frames.append(vad_buffer)
            else:
                num_silent_frames += 1

            if len(accumulated_frames) >= 16000 * 15 / (2 * vad_frame_len):  # Stop after 15 seconds of recording
                break

        if accumulated_frames:
            save_audio(accumulated_frames)
            play_sound(STOPPED_LISTENING_SOUND)
            print("Processing follow-up audio...")
            follow_up_command = transcribe()
            
            if not follow_up_command:
                print("No follow-up command detected, stopping the follow-up loop.")
                break  # Exit the follow-up loop if no command is detected or a stop condition is met

            print(f"Follow-up command: {follow_up_command}")

            # Now we check if the follow-up command is discernible English text
            # Assuming we have a function 'is_english_text' to check the transcribed text
            if is_english_text(follow_up_command, response):
                # Process the follow-up command as needed, similar to the initial command
                response = get_chatgpt_response(follow_up_command)
                text_to_speech(response)
            else:
                print("The follow-up command does not appear to be valid English.")
        else:
            print("No speech detected, stopping the follow-up loop.")
            break  # Exit the follow-up loop if no speech is detected

def main():
    audio_stream = pa.open(
        rate=16000,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    
    play_sound(LISTENING_SOUND)
    
    # get_chatgpt_response("Can you play your fav song and then set a reminder for me to do my homework at 5pm?")

    vad = webrtcvad.Vad(3)

    print("Say 'Jarvis' to wake up the assistant...")

    while True:
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
        # pcm_suppressed = koala.process(pcm_unpacked)
        keyword_index = porcupine.process(pcm_unpacked)

        if keyword_index >= 0:
            print("Jarvis activated. Listening for your command...")
            spotify_was_playing = False
            # if is_spotify_playing_on_device():
            #     spotify_was_playing = True
            # toggle_spotify_playback()
            play_sound(LISTENING_SOUND)
            accumulated_frames = []
            num_silent_frames = 0
            vad_frame_accumulator = []
            vad_frame_len = int(0.02 * 16000)  # 20 ms
            volume_boost_factor = 2.5  # Adjust this value as needed

            while True:
                pcm = audio_stream.read(koala.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * koala.frame_length, pcm)

                pcm_boosted = [int(sample * volume_boost_factor) for sample in pcm_unpacked]

                # Apply Koala noise suppression
                pcm_suppressed = koala.process(pcm_boosted)

                # Accumulate the suppressed frames for a full VAD frame
                vad_frame_accumulator.extend(pcm_suppressed)

                # Once enough samples are accumulated for a 20 ms frame, process with VAD
                if len(vad_frame_accumulator) >= vad_frame_len:
                    vad_frame = vad_frame_accumulator[:vad_frame_len]
                    vad_buffer = b''.join(struct.pack('h', sample) for sample in vad_frame)
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
            print("Processing audio...")
            command = transcribe()
            print(f"You said: {command}")
            user = determine_user_handler()
            response = get_chatgpt_response(command, speaker=str(user))
            if spotify_was_playing:
                toggle_spotify_playback()
            text_to_speech(response)
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