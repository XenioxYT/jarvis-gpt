def predict_intent(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)

    # Define reference sentences for different intents
    ref_sentences = {
        "control_lights": "Turn on the lights",
        "control_lights": "Turn off the lights",
        "bbc_news_briefing": "Tell me the news",
        "list_reminders": "What are my reminders",
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
        return best_intent
    
    return None