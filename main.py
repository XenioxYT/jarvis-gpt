import json
import os
import struct
import time
import pyaudio
import wave
import requests
import webrtcvad
import pvporcupine
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import texttospeech
import pyttsx3
import whisper
import io
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

messages = [
            {
            "role": "system",
            "content": "You are Jarvis, a voice-based personal assistant to Tom. You are speaking to him now. You are a voice assistant, so keep responses short and concise, but maintain all the important information. Since you are a voice assistant, you must remember to not include visual things, like text formatting, as this will not play well with TTS."
        },
]

# Load environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

oai_client = OpenAI(base_url=api_base, api_key=api_key)
pv_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

client = texttospeech.TextToSpeechClient()
tts_engine = pyttsx3.init()
model = whisper.load_model("base")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check the calendar for a given date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check, e.g., 2021-10-31. You can also give a range, e.g., 2021-10-31 - 2021-11-01",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Use the 'google_search' tool to retrieve internet search results relevant to your input. The results will return links and snippets of text from the webpages. Summarise the results in your response, not citing specific sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "The term to search for."
                    },
                    "num_results": {
                        "type": "integer",
                        "enum": [5, 10, 15],
                        "description": "Number of search results."
                    },
                },
                "required": ["search_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the weather forecast for a given location. Always use celcius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, or city and country, e.g., Sheffield, UK or Paris, France",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_current_weather(location):
    """Get the current weather in a given location using OpenWeatherMap One Call API"""
    # Use geocoding to get the latitude and longitude for the location
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
    geocode_response = requests.get(geocode_url).json()

    if not geocode_response:
        return json.dumps({"error": "Location not found"})

    lat = geocode_response[0]["lat"]
    lon = geocode_response[0]["lon"]

    # Call the One Call API with retrieved latitude and longitude
    weather_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,daily&units=metric&appid={OPENWEATHER_API_KEY}"
    weather_response = requests.get(weather_url).json()

    if "current" not in weather_response:
        return json.dumps({"error": "Could not retrieve weather data"})

    current_weather = weather_response["current"]
    return json.dumps({
        "location": location,
        "temperature": current_weather["temp"],
        "weather": current_weather["weather"][0]["description"],
        "unit": "Celsius"
    })

def authenticate_google_calendar_api():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    service = build('calendar', 'v3', credentials=creds)
    return service

def check_calendar(date):
    """Check the calendar for events on a given date or date range."""
    service = authenticate_google_calendar_api()
    date_range = date.split(" - ")
    start_date_str = date_range[0]
    end_date_str = date_range[1] if len(date_range) > 1 else start_date_str
    try:
        # Parse dates from strings and create date range for the query
        time_min = datetime.datetime.fromisoformat(start_date_str).isoformat() + 'Z'
        time_max = (datetime.datetime.fromisoformat(end_date_str) + datetime.timedelta(days=1)).isoformat() + 'Z'

        # Call the Google Calendar API
        events_result = service.events().list(calendarId='primary', timeMin=time_min,
                                              timeMax=time_max, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
        print(f"Found {len(events)} events")

        # Prepare the list of events in the required output format
        event_list = [
            {
                "summary": event["summary"],
                "location": event.get("location", "No location specified"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "description": event.get("description", "No description provided")
            } 
            for event in events
        ]

        # Output the events for the given date range
        return json.dumps({"date": date, "events": event_list})
    except Exception as e:
        print(f"An error occurred: {e}")
        return json.dumps({"date": date, "error": str(e), "events": []})

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(access_key=pv_access_key, keywords=["jarvis"])

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
    # model = whisper.load_model("base")  # Choose the appropriate model size
    result = model.transcribe(filename)
    return result["text"]

# Function to get response from ChatGPT, making any necessary tool calls
def get_chatgpt_response(text):
    messages.append({"role": "user", "content": text})

    # Send the initial message and the available tool to the model
    response = oai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Extract the response message
    response_message = response.choices[0].message
    # Process any tool calls
    tool_calls = getattr(response_message, 'tool_calls', [])
    
    if tool_calls:
        # Dictionary mapping function names to actual function implementations
        available_functions = {
            "get_current_weather": get_current_weather,
            "check_calendar": check_calendar,
            # "google_search": google_search,
        }

        for tool_call in tool_calls:
            print(f"Tool call: {tool_call}")
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Function name: {function_name}", f"Function args: {function_args}")
            
            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                
                # Send the function response back to the model
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                
                # Resolve any follow-up after the tool call
            second_response = oai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages
            )
                # Assume that we return the final response text after the tool call handling
            return second_response.choices[0].message.content
        else:
            raise Exception(f"Function '{function_name}' is not implemented.")
    else:
        # Return the direct response text when no tool calls are needed
        return response_message.content

# Function to convert text to speech using Google Cloud TTS
def text_to_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Use a British accent voice, for example "en-GB-Wavenet-B"
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-GB',
        name='en-GB-Wavenet-B',  # You can choose a different British Wavenet voice if desired
        ssml_gender=texttospeech.SsmlVoiceGender.MALE  # Assuming Jarvis has a male voice
    )

    # Use LINEAR16 audio encoding for high quality
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Perform the Text-to-Speech request on the text input with the selected voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # First, save the audio to a buffer
    audio_buffer = io.BytesIO(response.audio_content)

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

    vad = webrtcvad.Vad(2)

    print("Say 'Jarvis' to wake up the assistant...")

    while True:
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm_unpacked)

        if keyword_index >= 0:
            print("Jarvis activated. Listening for your command...")
            accumulated_frames = []

            # Initialize the variable to keep track of silence
            num_silent_frames = 0
            vad_frame_len = int(0.02 * 16000)  # 20 ms

            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
                vad_buffer = b''.join(struct.pack('h', frame) for frame in pcm_unpacked)

                # Verify if the buffer is speech
                is_speech = vad.is_speech(vad_buffer[:2 * vad_frame_len], 16000)

                if is_speech:
                    num_silent_frames = 0
                else:
                    num_silent_frames += 1

                accumulated_frames.append(vad_buffer)

                # Stop capturing after a short period of silence
                if num_silent_frames > 30:
                    break

            # Save and process the captured speech
            save_audio(accumulated_frames)
            print("Done capturing. Processing audio...")
            command = transcribe()
            print(f"You said: {command}")
            response = get_chatgpt_response(command)
            text_to_speech(response)

    audio_stream.close()
    pa.terminate()
    porcupine.delete()


if __name__ == '__main__':
    main()