import os
from google.cloud import texttospeech
import pyaudio
import time
import io
import numpy as np


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../credentials/creds.json'
client = texttospeech.TextToSpeechClient()
pa = pyaudio.PyAudio()


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