# import pveagle
# from pvrecorder import PvRecorder


# DEFAULT_DEVICE_INDEX = -1
# access_key = "{YOUR_ACCESS_KEY}";


# # Step 1: Enrollment
# try:
#     eagle_profiler = pveagle.create_profiler(access_key="+urKHbMJGDpVUab5/mZ/v9QEQtsiEGLF1h13rsbUGGTI1Qt4b1pggA==")
# except pveagle.EagleError as e:
#     pass


# enroll_recorder = PvRecorder(
#     device_index=DEFAULT_DEVICE_INDEX,
#     frame_length=eagle_profiler.min_enroll_samples)


# enroll_recorder.start()


# enroll_percentage = 0.0
# while enroll_percentage < 100.0:
#     print(enroll_percentage)
#     audio_frame = enroll_recorder.read()
#     enroll_percentage, feedback = eagle_profiler.enroll(audio_frame)


# enroll_recorder.stop()


# speaker_profile = eagle_profiler.export()


# enroll_recorder.delete()
# eagle_profiler.delete()


# # Step 2: Recognition
# try:
#     eagle = pveagle.create_recognizer(
#         access_key="+urKHbMJGDpVUab5/mZ/v9QEQtsiEGLF1h13rsbUGGTI1Qt4b1pggA==",
#         speaker_profiles=[speaker_profile]
#         )
# except pveagle.EagleError as e:
#     print(e)
#     pass


# recognizer_recorder = PvRecorder(
#     device_index=DEFAULT_DEVICE_INDEX,
#     frame_length=eagle.frame_length)


# recognizer_recorder.start()


# while True:
#     audio_frame = recognizer_recorder.read()
#     scores = eagle.process(audio_frame)
#     print(scores)


# recognizer_recorder.stop()


# recognizer_recorder.delete()
# eagle.delete()

import wave
import struct
import os
import pveagle

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
            raise ValueError("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))
        if sample_width != 2:
            raise ValueError("Audio file should be 16-bit. got %d" % sample_width)
        if channels != 1:
            raise ValueError("Eagle processes single-channel audio but stereo file is provided.")
            
        samples = wav_file.readframes(num_frames)
    frames = struct.unpack('h' * num_frames, samples)
    return frames

def determine_speaker(access_key, library_path, model_path, input_profile_paths, test_audio_path):
    # Read the audio file
    def read_file(file_name, sample_rate):
        with wave.open(file_name, mode="rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()

            if wav_file.getframerate() != sample_rate:
                raise ValueError("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))
            if sample_width != 2:
                raise ValueError("Audio file should be 16-bit. got %d" % sample_width)
            if channels != 1:
                raise ValueError("Eagle processes single-channel audio but stereo file is provided.")
                
            samples = wav_file.readframes(num_frames)

        frames = struct.unpack('h' * num_frames, samples)
        return frames
    
    # Load speaker profiles
    speaker_labels = [os.path.splitext(os.path.basename(path))[0] for path in input_profile_paths]
    speaker_profiles = []
    for input_profile_path in input_profile_paths:
        with open(input_profile_path, 'rb') as f:
            speaker_profiles.append(pveagle.EagleProfile.from_bytes(f.read()))
    
    # Create the Eagle recognizer
    eagle = pveagle.create_recognizer(
        access_key=access_key,
        model_path=model_path,
        library_path=library_path,
        speaker_profiles=speaker_profiles)

    # Process the audio and determine the speaker
    audio = read_file(test_audio_path, eagle.sample_rate)
    num_frames = len(audio) // eagle.frame_length
    speakers_scores = []
    for i in range(num_frames):
        frame = audio[i * eagle.frame_length:(i + 1) * eagle.frame_length]
        scores = eagle.process(frame)
        speakers_scores.append(scores)

    # Delete the Eagle recognizer to free resources
    eagle.delete()

    # Return the speaker labels with the highest score
    average_scores = [sum(x) / len(x) for x in zip(*speakers_scores)]
    max_score_index = average_scores.index(max(average_scores))
    selected_speaker = speaker_labels[max_score_index]

    return selected_speaker

# Example usage:
# Replace these with the appropriate paths and access key.
# access_key = ''
# library_path = None # or the path to the Picovoice Eagle dynamic library
# model_path = None # or the path to the Picovoice Eagle model file
# input_profile_paths = ['./output_profile.pv'] # Paths to speaker profiles
# test_audio_path = './harvard.wav' # Path to the audio file to test

# speaker = determine_speaker(access_key, library_path, model_path, input_profile_paths, test_audio_path)
# print("The determined speaker is:", speaker)

def enroll_user(access_key, enroll_audio_paths, output_profile_path):
    # Check if the provided audio paths are valid WAV files.
    for audio_path in enroll_audio_paths:
        if not audio_path.lower().endswith('.wav'):
            raise ValueError('The file at "{}" must have a WAV file extension'.format(audio_path))

    # Initialize the Eagle profile.
    eagle_profiler = pveagle.create_profiler(
        access_key=access_key,)

    try:
        # Enroll the speaker using the provided audio files.
        enroll_percentage = 0.0
        enrollment_feedback = None
        for audio_path in enroll_audio_paths:
            audio = read_file(audio_path, eagle_profiler.sample_rate)
            enroll_percentage, feedback = eagle_profiler.enroll(audio)
            enrollment_feedback = feedback
            print('Enrolled audio file %s [Enrollment percentage: %.2f%% - Enrollment feedback: %s]' %
                  (audio_path, enroll_percentage, FEEDBACK_TO_DESCRIPTIVE_MSG[feedback]))

        # Optionally save the speaker profile.
        if enroll_percentage == 100.0:
            speaker_profile = eagle_profiler.export()
            if output_profile_path:
                with open(output_profile_path, 'wb') as f:
                    f.write(speaker_profile.to_bytes())
                print('Speaker profile is saved to %s' % output_profile_path)
                return str(FEEDBACK_TO_DESCRIPTIVE_MSG[enrollment_feedback]) + " User enrolled successfully. "
        else:
            print('Failed to create speaker profile. Insufficient enrollment percentage: %.2f%%. '
                  'Please add more audio files for enrollment.' % enroll_percentage)
            return "Insufficient enrollment percentage: {}.".format(enroll_percentage) + "Entrollment feedback: " + str(FEEDBACK_TO_DESCRIPTIVE_MSG[enrollment_feedback]) + " Please get the user to say another sentence. Keep going until 100 percent is reached. "
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