import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel, pipeline
from nltk.util import ngrams
import torch
from langdetect import detect
from utils.strings import common_bigrams, common_trigrams
import string
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

llama_client = OpenAI(base_url=api_base, api_key=api_key)

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

def is_english_text(user, previous_response):
    message = [
        {
            "role": "user",
            "content": "You are an AI designed to class intent of a sentace for a voice assistant. You class whether the sentence makes sense regarding current context and language understanding. DO NOT give any other output other than JSON. Here is how you should format your output:"
            "{"
            "    'followup': true"
            "}"
            "where true is if the output is a follow up to the previous response from the assistant. This can be anything from a follow up question to a 'Thank you'', expressing gratitude for the answer, so long as it makes sense in the context. The assistant reply is: '" + previous_response + "'"
            "And the user input:"
            "'"+ user + "'"
        }
    ]
    response = llama_client.chat.completions.create(
        model="mistral-7b",
        messages=message
    )
    print(response)
    classification = response = response.choices[0].message.content
    if "true" in classification:
        print("followup")
        return True
    else:
        print("not followup")
        return False

#test the function
print(is_english_text("Hello", "Hello, how are you?"))

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
