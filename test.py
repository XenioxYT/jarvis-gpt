# import nltk
# from nltk.tokenize import word_tokenize
# from transformers import BertTokenizer, BertModel
# import torch
# from langdetect import detect
# from textstat import flesch_reading_ease

# # Ensure necessary resources from nltk are downloaded
# nltk.download('punkt')

# # Initialize BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def is_english_text(text):
#     """
#     Enhanced check to see if text:
#     A) Is likely to be a follow-up in a conversation
#     B) Is in English
#     C) Is coherent and sensible
#     """
#     # Language Check
#     try:
#         if detect(text) != 'en':
#             return False
#     except:
#         return False

#     # Tokenization
#     words = word_tokenize(text)
#     if len(words) == 0:
#         return False

#     # Adjusted for short but valid sentences and common interjections
#     common_short_responses = {'yes', 'no', 'maybe', 'stop', 'okay', 'thanks', 'hi', 'hello', 'umm', 'ah'}
#     if len(words) <= 3 or any(word.lower() in common_short_responses for word in words):
#         return True

#     # Simplicity Score Check
#     if flesch_reading_ease(text) < 60:  # Adjust this threshold as needed
#         return False

#     # Semantic Coherence Check with BERT
#     inputs = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)

#     embeddings = outputs.last_hidden_state[:, 0, :]
#     if torch.norm(embeddings) < 6.5:  # Adjusted threshold
#         return False

#     return True


# # Assuming is_english_text is defined as per your enhanced version

# def test_is_english_text():
#     test_sentences = [
#         "Turn off the lights.",
#         "What's the weather like today?",
#         "Reminder set for 3 PM.",
#         "Play some jazz music.",
#         "How to boil an egg?",
#         "Pineapple and pizza.",
#         "The cat in the hat.",
#         "Bonjour, comment ça va?",
#         "Could you repeat that?",
#         "Who won the football game last night?",
#         "Set alarm for six in the morning.",
#         "Where is the nearest gas station?",
#         "Text message to John: Running late.",
#         "Why is the sky blue?",
#         "Call me an Uber.",
#         "Baking soda volcano experiment steps.",
#         "What about tomorrow's weather?",
#         "Can you set it for 10 minutes later?",
#         "And how about the traffic to work?",
#         "What time does it open?",
#         "Could you play the next song?",
#         "Did I get any new emails?",
#         "Who's calling?",
#         "Can it be louder?",
#         "What ingredients do I need for a cake?",
#         "How long will it take to get there?",
#         "Mumbling under breath.",
#         "Ah, nevermind.",
#         "Hmm, I'm not sure.",
#         "Just thinking aloud.",
#         "Oh, it's nothing.",
#         "Umm, let me see.",
#         "Es ist ein schöner Tag.",
#         "No sé qué hacer.",
#         "忘れてください",
#         "Random gibberish text here.",
#         "Stop.",
#         "Yes.",
#         "No thanks.",
#         "Maybe.",
#         "That's cool.",
#         "Really?",
#         "I see.",
#         "Hold on.",
#         "Wait a minute.",
#         "Okay then."
#     ]

#     for sentence in test_sentences:
#         result = is_english_text(sentence)
#         print(f"Sentence: '{sentence}' - Is English: {result}")

# Run the test function
# test_is_english_text()

# import os
# import librosa
# import joblib
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score

# # Step 2: Data Preparation
# dataset_path = "./dataset"
# labels = []
# features = []

# # Extract features from each audio file
# for speaker in os.listdir(dataset_path):
#     speaker_path = os.path.join(dataset_path, speaker)
#     if os.path.isdir(speaker_path):
#         for filename in os.listdir(speaker_path):
#             file_path = os.path.join(speaker_path, filename)
#             if file_path.endswith('.wav'):
#                 audio, sr = librosa.load(file_path)
#                 mfcc = librosa.feature.mfcc(y=audio, sr=sr)
#                 mean_mfcc = np.mean(mfcc, axis=1)
#                 features.append(mean_mfcc)
#                 labels.append(speaker)

# # Convert to numpy arrays
# features = np.array(features)
# labels = np.array(labels)

# # Step 3: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Step 4: Model Training
# model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
# model.fit(X_train, y_train)

# # Step 5: Testing the Model
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Save the model
# model_filename = 'speaker_identification_model.pkl'
# joblib.dump(model, model_filename)

# # Function to extract MFCC features
# def extract_features(file_path):
#     audio, sr = librosa.load(file_path)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr)
#     mean_mfcc = np.mean(mfcc, axis=1)
#     return mean_mfcc.reshape(1, -1)

# # New function to predict speaker with confidence check
# def predict_speaker(model, features, threshold=0.7):
#     probabilities = model.predict_proba(features)[0]
#     max_prob = np.max(probabilities)
#     predicted_speaker = model.classes_[np.argmax(probabilities)]
#     if max_prob > threshold:
#         return predicted_speaker
#     else:
#         return "Unknown"

# # Load the model
# model = joblib.load('speaker_identification_model.pkl')

# # Path to a new audio file
# new_audio_file = './Recording.wav'

# # Extract features and predict
# new_features = extract_features(new_audio_file)
# speaker = predict_speaker(model, new_features)
# print(f"Predicted Speaker: {speaker}")


# Sample dataset
phrases = [
    # Turn on the light phrases
    "turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights",
    "illuminate the room", "light up", "brighten the room", "activate the lights", "bring light",
    "let there be light", "shine the light", "enable lighting", "turn lights to full", "light this place up",
    "more light please", "lights to maximum", "open the blinds", "raise the lights", "lights to 100%",

    # Turn off the light phrases
    "turn off the light", "lights out", "can you dim the lights", "make it dark", "switch off the lights",
    "extinguish the lights", "darken the room", "turn the lights down", "deactivate the lights", "lights off please",
    "kill the lights", "shut off the lights", "lower the lights", "reduce brightness", "fade the lights",
    "lights to zero", "no more light", "eliminate light", "turn down lights", "lights to minimum",

    # Other phrases
    "what's the weather today", "play some music", "set an alarm for 7 am", "remind me to call John",
    "how do I get to the nearest gas station",
    "tell me the news", "set a timer for 20 minutes", "what's on my calendar today", "play the latest podcast",
    "find a nearby restaurant",
    "open the window", "start the dishwasher", "is my coffee ready", "schedule a meeting", "call my mom",
    "send a message", "what's the date today", "turn up the heat", "lower the thermostat", "is the door locked",

    # Multiple intents
    "turn on the light and play some music", "lights on and play some jazz", "turn off the light after 20 minutes",
    "turn off the light in 5 minutes", "play your favorite music and dim the lights",
    "dim the lights and start the movie", "lights out and set alarm for 6 am",
    "brighten the room and read the news aloud", "play some music and turn on the light",
    "edit my reminder for 3 pm, and what's on my calendar?",
    "switch off the lights then play relaxing sounds", "light up the room and tell me the weather",
    "turn on the lights and make coffee", "set a reminder for 10 and play music",
    "tell me my reminders and my calendar with the lights on",
    "activate lights and play morning playlist", "turn lights to full and open the blinds",
    "kill the lights and start the dishwasher", "list my reminders and tell me the weather",
    "what's on my calendar and reminders for 3pm?",
    "lights to minimum and call my mom", "bring light and set a timer for cooking",
    "raise the lights and schedule my meeting", "play a song and tell me the weather?",
    "is it going to rain when my reminder is on?",
    "deactivate the lights then send a message", "illuminate the room and find a nearby cafe",
    "lights off please and remind me to call John", "what time are my appointments and will it rain then?",
    "whats on my calendar and my reminders?",
]

labels = [
    # Labels for turn on the light
    "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light",
    "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light",
    "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light",
    "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light", "turn_on_light",

    # Labels for turn off the light
    "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light",
    "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light",
    "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light",
    "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light", "turn_off_light",

    # Labels for other
    "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other",

    # Labels for multiple intents
    "multiple", "multiple", "multiple", "multiple", "multiple",
    "multiple", "multiple", "multiple", "multiple", "multiple",
    "multiple", "multiple", "multiple", "multiple", "multiple",
    "multiple", "multiple", "multiple", "multiple", "multiple",
    "multiple", "multiple", "multiple", "multiple", "multiple",
    "multiple", "multiple", "multiple", "multiple", "multiple",
]

# from sklearn.feature_extraction.text import TfidfVectorizer

# # Initialize the vectorizer
# vectorizer = TfidfVectorizer()

# # Fit and transform the phrases
# X = vectorizer.fit_transform(phrases)
# from sklearn.linear_model import LogisticRegression

# # Initialize the classifier
# classifier = LogisticRegression()

# # Train the classifier
# classifier.fit(X, labels)
import joblib

# # Save the classifier
# joblib.dump(classifier, 'light_intent_classifier.pkl')

# # Save the vectorizer
# joblib.dump(vectorizer, 'vectorizer.pkl')
# Load the classifier and vectorizer
classifier = joblib.load('light_intent_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Function to predict intent
def predict_intent(command):
    X = vectorizer.transform([command])
    return classifier.predict(X)[0]


# Example usage
command = "plunge me into darkness"
intent = predict_intent(command)
print(f"Predicted intent: {intent}")
