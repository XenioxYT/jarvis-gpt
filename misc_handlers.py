import os
import random
import shutil
from pveagle_speaker_identification import enroll_user, determine_speaker
from noise_reduction import reduce_noise_and_normalize

try:
    pv_access_key = os.getenv("picovoice_access_key")
except:
    raise Exception("Picovoice access key not found.")


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
    try:
        result = determine_speaker(access_key=pv_access_key, input_profile_paths=input_profile_paths, test_audio_path=audio_path)
    except:
        return "PicoVoice API key not found. Tell the user to set this in the setup -> configuration section."
        
    queue.put(result)
    return "Unknown"