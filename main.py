import struct
import pyaudio
import wave
import os
from openai import OpenAI
import google.cloud.texttospeech as tts
# import stt  # Importing coqui-stt
import pvporcupine
# load .env file
from dotenv import load_dotenv
load_dotenv()
import pyttsx3

# Initialize Coqui STT
# model_file_path = 'coqui_model.pbmm'  # Update the model file path
# model = stt.Model(model_file_path)
# model.enable_external_scorer('coqui_scorer.scorer')  # Update the external scorer

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

oai_client = OpenAI(base_url=api_base, api_key=api_key, max_retries=0)
pv_access_key = os.getenv("PORCUPINE_ACCESS_KEY")

# Initialize pyttsx3 Text-to-Speech engine
tts_engine = pyttsx3.init()

# Initialize PyAudio
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=16000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=4096  # Try increasing the buffer size
)

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(access_key=pv_access_key, keywords=["jarvis"])

# Initialize Google Text-to-Speech client
client = tts.TextToSpeechClient()

# Function to transcribe speech to text
import wave
import whisper

def transcribe():
    print("Listening for your command...")
    frames = []
    for _ in range(0, int(16000 / 1024 * 5)):  # 5-second audio
        data = audio_stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    # Write the frames to a WAV file
    with wave.open('temp.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Load Whisper model
    model = whisper.load_model("base")  # Choose the appropriate model size

    # Transcribe the audio
    result = model.transcribe('temp.wav')
    return result["text"]



# Function to get response from ChatGPT (OpenAI API)
def get_chatgpt_response(text):
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    messages.append({"role": "user", "content": text})
    response = oai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages,
    )
    print(response)
    message_content = response.choices[0].message.content
    return message_content
# Function to convert text to speech using Google TTS
def text_to_speech(text):
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code="en-US", ssml_gender=tts.SsmlVoiceGender.NEUTRAL)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open('response.mp3', 'wb') as out:
        out.write(response.audio_content)
        os.system('mpg321 response.mp3')

# def text_to_speech(text):
#     tts_engine.say(text)
#     tts_engine.runAndWait()

# Main loop
print("Say 'Jarvis' to wake up the assistant...")

while True:
    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

    keyword_index = porcupine.process(pcm)
    if keyword_index >= 0:
        print("Jarvis activated.")
        command = transcribe()
        print(f"You said: {command}")
        response = get_chatgpt_response(command)
        text_to_speech(response)

# Cleanup
audio_stream.close()
pa.terminate()
porcupine.delete()
