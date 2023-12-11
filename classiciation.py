import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel, pipeline
from nltk.util import ngrams
import torch
from langdetect import detect
from utils.strings import common_bigrams, common_trigrams
import string

# Ensure necessary resources from nltk are downloaded
nltk.download('punkt')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def remove_punctuation(text):
    """
    Remove punctuation from the text.
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def is_english_text(text):
    """
    Enhanced multi-stage check to see if text:
    A) Contains common patterns found in voice assistant interactions
    B) Is semantically coherent
    C) Is in English
    """
    # Language Check
    try:
        if detect(text) != 'en':
            return False
    except:
        return False

    # Punctuation Normalization (if necessary)
    text = remove_punctuation(text)

    # Tokenization
    words = word_tokenize(text)
    if len(words) == 0:
        return False

    # First Stage: N-gram Analysis
    trigrams_in_text = set(ngrams(words, 3)) if len(words) >= 3 else set()
    bigrams_in_text = set(ngrams(words, 2)) if len(words) >= 2 else set()

    if trigrams_in_text.intersection(common_trigrams) or bigrams_in_text.intersection(common_bigrams):
        return True

    # Second Stage: Semantic Coherence Check with BERT
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    if torch.norm(embeddings) < 6.5:  # Adjusted threshold
        return False

    return True

# Initialize the intent classification model
intent_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

def user_query(input):
    if input == "":
        return None
    # Define potential intents
    spotify_commands = ["resume music", "pause music", "stop music", "turn on device", "turn off device"]
    other_intents = ["general conversation", "other task"]

    # Classify the intent of the input
    result = intent_classifier(input, spotify_commands + other_intents)
    print(result)

    # Extract the top intent where the score is above 0.75
    for i in range(len(result["scores"])):
        if result["scores"][i] > 0.75:
            top_intent = result["labels"][i]
            break
        else:
            top_intent = None

    # Map top intent to Spotify commands
    if top_intent == "resume music":
        return "play_spotify"
    
    elif top_intent in ["pause music", "stop music"]:
        return "pause_spotify"
    
    elif top_intent in ["skip song", "next song"]:
        return "skip_song"
    
    elif top_intent == "previous song":
        return "previous_song"
    
    elif top_intent == "turn on device":
        return "turn_on_device"
    
    elif top_intent == "turn off device":
        return "turn_off_device"
    
    else:
        return None

# Test the function
# test_queries = ["Play The Weeknd", "pause the music", "stop the track", "skip to the next song", "Do you play games?"]
# test_queries = [
#     "turn on the lights",
#     "turn off the lights",
#     "turn on the desk lamp",
# ]

# for query in test_queries:
#     print(f"Query: '{query}' - Command: {user_query(query)}")
