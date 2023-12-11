import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
from langdetect import detect
from textstat import flesch_reading_ease

# Ensure necessary resources from nltk are downloaded
nltk.download('punkt')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def is_english_text(text):
    """
    Enhanced check to see if text:
    A) Is likely to be a follow-up in a conversation
    B) Is in English
    C) Is coherent and sensible
    """
    # Language Check
    try:
        if detect(text) != 'en':
            return False
    except:
        return False

    # Tokenization
    words = word_tokenize(text)
    if len(words) == 0:
        return False

    # Adjusted for short but valid sentences and common interjections
    common_short_responses = {'yes', 'no', 'maybe', 'stop', 'okay', 'thanks', 'hi', 'hello', 'umm', 'ah'}
    if len(words) <= 3 or any(word.lower() in common_short_responses for word in words):
        return True

    # Simplicity Score Check
    if flesch_reading_ease(text) < 60:  # Adjust this threshold as needed
        return False

    # Semantic Coherence Check with BERT
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    if torch.norm(embeddings) < 6.5:  # Adjusted threshold
        return False

    return True


# Assuming is_english_text is defined as per your enhanced version

def test_is_english_text():
    test_sentences = [
        "Turn off the lights.",
        "What's the weather like today?",
        "Reminder set for 3 PM.",
        "Play some jazz music.",
        "How to boil an egg?",
        "Pineapple and pizza.",
        "The cat in the hat.",
        "Bonjour, comment ça va?",
        "Could you repeat that?",
        "Who won the football game last night?",
        "Set alarm for six in the morning.",
        "Where is the nearest gas station?",
        "Text message to John: Running late.",
        "Why is the sky blue?",
        "Call me an Uber.",
        "Baking soda volcano experiment steps.",
        "What about tomorrow's weather?",
        "Can you set it for 10 minutes later?",
        "And how about the traffic to work?",
        "What time does it open?",
        "Could you play the next song?",
        "Did I get any new emails?",
        "Who's calling?",
        "Can it be louder?",
        "What ingredients do I need for a cake?",
        "How long will it take to get there?",
        "Mumbling under breath.",
        "Ah, nevermind.",
        "Hmm, I'm not sure.",
        "Just thinking aloud.",
        "Oh, it's nothing.",
        "Umm, let me see.",
        "Es ist ein schöner Tag.",
        "No sé qué hacer.",
        "忘れてください",
        "Random gibberish text here.",
        "Stop.",
        "Yes.",
        "No thanks.",
        "Maybe.",
        "That's cool.",
        "Really?",
        "I see.",
        "Hold on.",
        "Wait a minute.",
        "Okay then."
    ]

    for sentence in test_sentences:
        result = is_english_text(sentence)
        print(f"Sentence: '{sentence}' - Is English: {result}")

# Run the test function
# test_is_english_text()
