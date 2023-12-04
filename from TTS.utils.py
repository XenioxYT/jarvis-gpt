from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import sounddevice as sd
import torch

def download_and_setup_model(model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    # Initialize the Model Manager
    model_manager = ModelManager()

    # Download and set up the specified TTS model
    model_path, config_path, model_item = model_manager.download_model(model_name)

    # Download and set up the default vocoder for the selected model
    vocoder_name = model_item["default_vocoder"]
    vocoder_path, _, _ = model_manager.download_model(vocoder_name)

    # Initialize the synthesizer with the downloaded models
    synthesizer = Synthesizer(
        model_path, 
        config_path,
        vocoder_path,
        use_cuda=torch.cuda.is_available()
    )
    return synthesizer

def synthesize_speech(synthesizer, text):
    # Synthesize the speech
    wav = synthesizer.tts(text)
    return wav

def main():
    # Download and setup the TTS and vocoder models
    synthesizer = download_and_setup_model()

    # The text to be synthesized
    text = "Hello, this is a test using Coqui TTS."

    # Synthesize the speech
    wav = synthesize_speech(synthesizer, text)

    # Save the output to a WAV file
    from scipy.io.wavfile import write
    write('output.wav', synthesizer.sample_rate, wav)

    # Play the synthesized speech
    sd.play(wav, samplerate=synthesizer.sample_rate)

if __name__ == "__main__":
    main()
