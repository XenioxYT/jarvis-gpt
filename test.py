import spacy

# Load spaCy's medium or large model for English
try:
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading language model for the spaCy POS tagger")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

def detect_intent(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Define reference sentences for different intents
    ref_sentences = {
        "Turn on lights": "Turn on the lights",
        "Turn off lights": "Turn off the lights",
        "News": "Tell me the news",
        "Reminders": "What are my reminders"
    }

    # Initialize variables to store the best match
    best_intent = None
    highest_similarity = 0.0

    # Check for semantic similarity with each reference sentence
    for intent, ref_text in ref_sentences.items():
        ref_doc = nlp(ref_text)
        similarity = doc.similarity(ref_doc)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_intent = intent

    # Determine if the highest similarity score is above a threshold
    if highest_similarity > 0.7:  # Adjust the threshold as needed
        return f"Intent detected: {best_intent}"
    
    return "No clear intent detected"

# Test the function
print(detect_intent("Can you turn on the lights?"))
print(detect_intent("can you turn the lights off?"))
print(detect_intent("What's the latest news today?"))
print(detect_intent("Show me my reminders for today"))
