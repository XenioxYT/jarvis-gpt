import subprocess
import os

def reduce_noise_and_normalize(input_wav_path, noise_reduction_level=0.21, normalization_level=0.0):
    """
    Reduces noise from an audio file and normalizes volume using SoX.

    Parameters:
    - input_wav_path: str, path to the input .wav file that contains noise.
    - noise_reduction_level: float, level of noise reduction (0.0 to 1.0).
    - normalization_level: float, target volume level for normalization.

    Returns:
    - output_wav_path: str, path to the output .wav file with reduced noise and normalized volume.
    """
    
    # Check if the input file exists
    if not os.path.isfile(input_wav_path):
        raise FileNotFoundError(f"No such file: {input_wav_path}")
    
    noise_profile_path = "./sounds/noise.prof"
    intermediate_wav_path = "./temp_intermediate.wav"
    output_wav_path = "./temp_cleaned_normalised.wav"
    
    # Apply noise reduction using the profile
    # subprocess.run([
    #     'sox', input_wav_path, intermediate_wav_path, 'noisered', noise_profile_path, str(noise_reduction_level)
    # ], check=True)
    
    # Normalize the volume
    subprocess.run([
        'sox', input_wav_path, output_wav_path, 'gain', '-n', str(normalization_level)
    ], check=True)
    
    # Clean up: remove the intermediate file
    # os.remove(intermediate_wav_path)

# Example usage:
# reduce_noise_and_normalize('./temp.wav', noise_reduction_level=0.21)
# print(f"Reduced noise and normalized audio saved.")
