import json
import os
import struct
import time
import pyaudio
import wave
import webrtcvad
import pvporcupine
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import texttospeech
import pyttsx3
import whisper
import io

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

client = texttospeech.TextToSpeechClient()
tts_engine = pyttsx3.init()

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
    }
]

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "62", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

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
    model = whisper.load_model("base")  # Choose the appropriate model size
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

    vad = webrtcvad.Vad(1)

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