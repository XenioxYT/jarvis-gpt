import wave
import struct
import os
import pveagle
import numpy as np

FEEDBACK_TO_DESCRIPTIVE_MSG = {
    pveagle.EagleProfilerEnrollFeedback.AUDIO_OK: 'Good audio',
    pveagle.EagleProfilerEnrollFeedback.AUDIO_TOO_SHORT: 'Insufficient audio length',
    pveagle.EagleProfilerEnrollFeedback.UNKNOWN_SPEAKER: 'Different speaker in audio',
    pveagle.EagleProfilerEnrollFeedback.NO_VOICE_FOUND: 'No voice found in audio',
    pveagle.EagleProfilerEnrollFeedback.QUALITY_ISSUE: 'Low audio quality due to bad microphone or environment'
}


def read_file(file_name, sample_rate):
    with wave.open(file_name, mode="rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        if wav_file.getframerate() != sample_rate:
            raise ValueError(
                "Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))
        if sample_width != 2:
            raise ValueError("Audio file should be 16-bit. got %d" % sample_width)
        if channels != 1:
            raise ValueError("Eagle processes single-channel audio but stereo file is provided.")

        samples = wav_file.readframes(num_frames)
    frames = struct.unpack('h' * num_frames, samples)
    return frames


def determine_speaker(access_key, input_profile_paths, test_audio_path):
    # Load speaker profiles using list comprehension
    speaker_labels = [os.path.splitext(os.path.basename(path))[0] for path in input_profile_paths]
    speaker_profiles = [pveagle.EagleProfile.from_bytes(open(path, 'rb').read()) for path in input_profile_paths]
    # Create the Eagle recognizer
    eagle = pveagle.create_recognizer(access_key=access_key, speaker_profiles=speaker_profiles)
    # Process the audio and determine the speaker using list comprehension
    audio = read_file(test_audio_path, eagle.sample_rate)
    num_frames = len(audio) // eagle.frame_length
    speakers_scores = [eagle.process(audio[i * eagle.frame_length:(i + 1) * eagle.frame_length]) for i in range(num_frames)]
    # Delete the Eagle recognizer to free resources
    eagle.delete()
    # Calculate the average scores using numpy's mean function
    average_scores = np.mean(speakers_scores, axis=0)
    # Find the index of the maximum score using numpy's argmax function
    max_score_index = np.argmax(average_scores)
    # Assign the selected_speaker using a ternary operator
    selected_speaker = speaker_labels[max_score_index] if average_scores[max_score_index] >= 0.1 else "Unknown"
    return selected_speaker


# Example usage:
# Replace these with the appropriate paths and access key.
# access_key = ''

# input_profile_paths = ['./Tom.pv'] # Paths to speaker profiles
# test_audio_path = './temp.wav' # Path to the audio file to test

# speaker = determine_speaker(access_key, input_profile_paths, test_audio_path)
# print("The determined speaker is:", speaker)

def enroll_user(access_key, enroll_audio_paths, output_profile_path):
    eagle_profiler = pveagle.create_profiler(access_key=access_key)

    try:
        enroll_percentage = 0.0
        enrollment_feedback = None
        for audio_path in enroll_audio_paths:
            if not audio_path.lower().endswith('.wav'):
                raise ValueError(f'The file at "{audio_path}" must have a WAV file extension')

            audio = read_file(audio_path, eagle_profiler.sample_rate)
            enroll_percentage, feedback = eagle_profiler.enroll(audio)
            enrollment_feedback = feedback
            print(f'Enrolled audio file {audio_path} [Enrollment percentage: {enroll_percentage:.2f}% - Enrollment feedback: {FEEDBACK_TO_DESCRIPTIVE_MSG[feedback]}]')

        if enroll_percentage == 100.0:
            speaker_profile = eagle_profiler.export()
            if output_profile_path:
                with open(output_profile_path, 'wb') as f:
                    f.write(speaker_profile.to_bytes())
                print(f'Speaker profile is saved to {output_profile_path}')
                return f"{FEEDBACK_TO_DESCRIPTIVE_MSG[enrollment_feedback]} User enrolled successfully."
        else:
            return f"Insufficient enrollment percentage: {enroll_percentage}. {FEEDBACK_TO_DESCRIPTIVE_MSG[enrollment_feedback]} Please get the user to say another sentence. Keep going until 100 percent is reached."
    finally:
        eagle_profiler.delete()

# Example usage:
# access_key = ''
# library_path = None  # or the path to the Picovoice Eagle dynamic library
# model_path = None  # or the path to the Picovoice Eagle model file
# enroll_audio_paths = ['./harvard.wav', './audio_8.wav']  # Paths to enrollment audio files
# output_profile_path = './output_profile.pv'

# profile_path, feedback = enroll_user(access_key, enroll_audio_paths, output_profile_path)
# print("Enrollment Feedback:", feedback)
# if profile_path:
#     print("Saved Profile Path:", profile_path)
