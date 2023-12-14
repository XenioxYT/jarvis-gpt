import os
import librosa
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Extract features from each audio file
def train_model():
    dataset_path = "./dataset"
    labels = []
    features = []
    for speaker in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker)
        if os.path.isdir(speaker_path):
            for filename in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, filename)
                if file_path.endswith('.wav'):
                    audio, sr = librosa.load(file_path)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
                    mean_mfcc = np.mean(mfcc, axis=1)
                    features.append(mean_mfcc)
                    labels.append(speaker)

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Step 3: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Model Training
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    # Step 5: Testing the Model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model_filename = 'speaker_identification_model.pkl'
    joblib.dump(model, model_filename)

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mean_mfcc = np.mean(mfcc, axis=1)
    return mean_mfcc.reshape(1, -1)

# Function to predict speaker with confidence check
def predict_speaker(model, features, threshold=0.7):
    probabilities = model.predict_proba(features)[0]
    print(probabilities)
    max_prob = np.max(probabilities)
    predicted_speaker = model.classes_[np.argmax(probabilities)]
    if max_prob > threshold:
        return predicted_speaker
    else:
        return "Unknown"

# Load the model
model = joblib.load('speaker_identification_model.pkl')

# New function to identify speaker from an audio file
def identify_speaker(audio_file_path):
    features = extract_features(audio_file_path)
    speaker = predict_speaker(model, features)
    print(speaker)
    if speaker == "speaker1":
        speaker = "Tom"
    else:
        speaker = "Unknown"
    return speaker

# Example usage
# train_model()
new_audio_file = './Recording.wav'
speaker = identify_speaker(new_audio_file)
print(f"Predicted Speaker: {speaker}")
