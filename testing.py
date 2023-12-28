import datetime
import io
import json
import random
import re
import threading
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
from calendar_utils import add_event_to_calendar, check_calendar
from news.bbc_news import download_bbc_news_summary
from noise_reduction import reduce_noise_and_normalize
from utils.home_assistant import toggle_entity
from utils.notes import edit_or_delete_notes, retrieve_notes, save_note
from utils.predict_intent import predict_intent
from utils.google_search import google_search
from utils.send_to_discord import send_message_sync
from google.cloud import texttospeech
from utils.spotify import search_spotify_song
# from utils.tools import tools
from utils.strings import tts_messages, username_mapping
import os
from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders
from utils.store_conversation import store_conversation, get_conversation
from utils.volume_control import volume_down, volume_up
from utils.weather import get_weather_data
from utils.tools import tools
# from main import text_to_speech

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

oai_client = OpenAI(base_url=api_base, api_key=api_key)

tts_thread = None

messages = [
    {
        "role": "system",
        "content": "As ChatGPT, you are a version of ChatGPT that has been optimized for engaging in general conversation, providing informative and accurate responses across a wide range of topics, and maintaining a friendly and approachable demeanor. Your knowledge is up-to-date as of April 2023. You do not have browsing capabilities, but you can process and respond to text inputs, including offering explanations, advice, and creative content. You should always adhere to safe and respectful conversational guidelines."
    }
]

client = texttospeech.TextToSpeechClient()
pa = pyaudio.PyAudio()


# Function to convert text to speech using Google Cloud TTS
def text_to_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code='en-GB',
        name='en-GB-Neural2-B',
        # ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Use audio encoding for high quality
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
        return (audio_buffer.read(frame_count * 2), pyaudio.paContinue)

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


def process_text(completion, waiting_for_number, first_sentence_processed, tts_thread):
    string1 = ""

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
                    tts_thread = threading.Thread(target=text_to_speech, args=(string1, ))
                    tts_thread.start()
                    completion = rest  # Reset completion to contain only the remaining text
                    first_sentence_processed = True

    return completion, waiting_for_number, first_sentence_processed, string1, tts_thread


def process_tool_call(tool_call, speaker, username_mapping, available_functions, tts_messages, multiple_tool_calls, tts_multiple_spoken, messages, cursor=None, db_conn=None):
    function_name = tool_call['function']['name']
    try:
        function_args = json.loads(tool_call['function']['arguments'])
    except:
        function_args = {}
    # if function_name == "bbc_news_briefing":
    #     bbc_news_thread = True

    if function_name in ["save_note", "edit_or_delete_notes", "retrieve_notes"]:
        function_args["user"] = speaker

    if function_name in ["add_event_to_calendar", "check_calendar", "send_to_phone"]:
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
        tts_thread_function = threading.Thread(target=text_to_speech, args=(tts_message, ))
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
        if cursor:
            store_conversation(1, messages, cursor, db_conn)
        else:
            store_conversation(1, messages)

    return messages, tts_multiple_spoken


def generate_response(input_message, speaker="Unknown", cursor=None, db_conn=None):
    global tts_thread
    global bbc_news_thread
    available_functions = {
        "get_weather_data": get_weather_data,
        "check_calendar": check_calendar,
        "set_reminder": add_reminder,
        "edit_reminder": edit_reminder,
        "list_reminders": list_unnotified_reminders,
        "add_event_to_calendar": add_event_to_calendar,
        "control_switch": toggle_entity,
        "play_song_on_spotify": search_spotify_song,
        # "enroll_user": enroll_user_handler,
        "google_search": google_search,
        "bbc_news_briefing": download_bbc_news_summary,
        "send_to_phone": send_message_sync,
        "volume_up": volume_up,
        "volume_down": volume_down,
        "save_note": save_note,
        "retrieve_notes": retrieve_notes,
        "edit_or_delete_notes": edit_or_delete_notes,
    }
    
    messages = get_conversation(1)
    
    # if speaker is not "Unknown":
    #     enroll_user_thread = threading.Thread(target=enroll_user_handler, args=(speaker,))
    #     enroll_user_thread.start()
    
    timestamp = datetime.datetime.now().strftime("%H:%M on %a %d %B %Y")
    messages.append({"role": "user", "content": f"At {timestamp}, {speaker} said: {input_message}"})
    
    store_conversation(1, messages, cursor, db_conn) if cursor else store_conversation(1, messages)
    
    # intent = predict_intent(input_message)
    
    while True:
        completion = ""
        full_completion = ""
        first_sentence_processed = False
        waiting_for_number = False
        tool_calls = []
        
        response = oai_client.chat.completions.create(
            messages=messages,
            model="gpt-4-1106-preview",
            tools=tools,
            stream=True
        )
        try:
            if tts_thread.is_alive():
                tts_thread.join()
        except Exception as e:
            print(e)
        
        for chunk in response:
            finish_reason = chunk.choices[0].finish_reason
            print(finish_reason) if finish_reason else None
            delta = chunk.choices[0].delta
            
            if delta.content or delta.content == '':
                completion += delta.content
                # print(delta.content)
                full_completion += delta.content
                
                completion, waiting_for_number, first_sentence_processed, string1, tts_thread = process_text(completion, waiting_for_number, first_sentence_processed, tts_thread)
            
            if delta.tool_calls:
                tcchunklist = delta.tool_calls
                for tcchunk in tcchunklist:
                    if len(tool_calls) <= tcchunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    tc = tool_calls[tcchunk.index]
                    
                    if tcchunk.id:
                        tc["id"] += tcchunk.id
                    if tcchunk.function.name:
                        tc["function"]["name"] += tcchunk.function.name
                    if tcchunk.function.arguments:
                        tc["function"]["arguments"] += tcchunk.function.arguments
                        
        if tool_calls:
            
            messages.append({
                "role": "function",
                "name": "tool_calls",
                "content": str(tool_calls)
            })
            
            multiple_tool_calls = len(tool_calls) > 1
            tts_multiple_spoken = False
            
            for tool_call in tool_calls:
                messages, tts_multiple_spoken = process_tool_call(tool_call, speaker, username_mapping, available_functions, tts_messages, multiple_tool_calls, tts_multiple_spoken, messages, cursor, db_conn)
            
            if finish_reason == "tool_calls":
                if full_completion != "":
                    messages.append({
                        "role": "assistant",
                        "content": full_completion
                    })
                    if completion != "":
                        # Start the TTS thread
                        tts_thread = threading.Thread(target=text_to_speech, args=(completion, ))
                        tts_thread.start()

                full_completion = ""
                completion = ""
                continue
            
        if completion != "":
            messages.append({
                "role": "assistant",
                "content": full_completion
            })
        
        if finish_reason == "stop" and len(tool_calls) == 0:
            try:
                if tts_thread.is_alive():
                    tts_thread.join()
            except Exception as e:
                print (e)
            store_conversation(1, messages, cursor, db_conn) if cursor else store_conversation(1, messages)
            return completion, tts_thread
            break
        
        

response, tts_thread = generate_response(input_message="Send a message to my phone with the current weather", speaker="Tom")
print(response)
try:
    if tts_thread.is_alive():
        tts_thread.join()
except Exception as e:
    print(e)
text_to_speech(response)