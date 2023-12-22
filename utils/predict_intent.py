def predict_intent(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)

    # Define reference sentences for different intents
    # ref_sentences = {
    #     "control_lights_on": ["Turn on the lights", "Lights on", "Activate the lights"],
    #     "control_lights_off": ["Turn off the lights", "Lights off", "Deactivate the lights"],
    #     "bbc_news_briefing": ["Tell me the news", "What's the news today", "News update, please"],
    #     "list_reminders": ["What are my reminders", "List my reminders for today"],
    #     "volume_down": ["Turn down the volume", "Lower the volume", "Volume lower, please"],
    #     "volume_up": ["Turn up the volume", "Increase the volume", "Volume higher, please"],
    #     "retrieve_notes": ["What are my notes", "List my notes", "Show me my notes"]
    # }
    
    ref_sentences = {
        "control_switch": "Turn on the lights",
        "control_switch": "Turn off the lights",
        "bbc_news_briefing": "Tell me the news",
        "list_reminders": "What are my reminders",
        "volume_down": "Turn down the volume",
        "volume_up": "Turn up the volume",
        "retrieve_notes": "What are my notes",
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
    if highest_similarity > 0.85:  # Adjust the threshold as needed
        return best_intent
    
    return None