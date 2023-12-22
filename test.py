# import joblib

# phrases = [
#     # Turn on the light phrases
#     "turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights",
#     "illuminate the room", "light up", "brighten the room", "activate the lights", "bring light",
#     "let there be light", "shine the light", "enable lighting", "turn lights to full", "light this place up",
#     "more light please", "lights to maximum", "raise the lights", "lights to 100%",

#     # Turn off the light phrases
#     "turn off the light", "lights out", "can you dim the lights", "make it dark", "switch off the lights",
#     "extinguish the lights", "darken the room", "turn the lights down", "deactivate the lights", "lights off please",
#     "kill the lights", "shut off the lights", "lower the lights", "fade the lights",
#     "lights to zero", "no more light", "eliminate light", "turn down lights", "lights to minimum",
    
#     # News phrases
#     "tell me the news", "play the news", "play the latest news", "play the news for today", "play the news for me",
#     "give me the latest from BBC News", "play today's news summary", "I'd like to hear the news from BBC", 
#     "what's the latest news on BBC", "BBC news update please", "can you play the top stories from BBC News",  
#     "start the BBC news briefing", "play BBC's news report", "I need to hear the morning news from BBC", 
#     "what are the evening headlines on BBC", "play the most recent news from BBC", "BBC News, please",
#     "BBC's main news stories today",
#     "can you play the midday news from BBC", "show me the latest news broadcast from BBC", 
#     "what's the current news on BBC",
#     "tell me today's headlines from BBC", "BBC's latest news coverage, please", "play the news headlines by the BBC",
    
#     # List reminders phrases
#     "list my reminders", "what are my reminders", "show me all my reminders", "can you tell me my reminders for today", "what reminders do I have set", 
#     "read out my upcoming reminders", "display my reminder list", "what's on my reminder list", 
#     "tell me my scheduled reminders", "what reminders do I have for this week", "list all my current reminders", 
#     "can you review my reminders", "I'd like to hear my list of reminders", "remind me about my upcoming tasks", 
#     "what tasks are in my reminders", "show me the reminders I set", "read me my to-do list from reminders", 
#     "what are the reminders for tomorrow", "list my reminders for the month", "can you list my pending reminders",
#     "what are my recent reminders", "show my reminders for the next few days", "tell me about my reminders for the weekend",
#     "can you display the reminders on my calendar", "I need an overview of my reminders", "read out my reminders for the day",
#     "what reminders are due soon", "can you check my reminders for appointments", "show my reminders for this afternoon",

#     # Other phrases
#     "what's the weather today", "play some music", "set an alarm for 7 am", "remind me to call John",
#     "how do I get to the nearest gas station",
#     "set a timer for 20 minutes", "what's on my calendar today", "play the latest podcast",
#     "find a nearby restaurant",
#     "open the window", "start the dishwasher", "is my coffee ready", "schedule a meeting", "call my mom",
#     "send a message", "what's the date today", "turn up the heat", "lower the thermostat", "is the door locked",
#     "tell me about the latest fortnite season", "what's the latest on the stock market", "what's the latest on the pandemic",
#     "What do i have on my calendar today", "what's the weather like tomorrow", "what's the weather like in London", 
#     "yes please", "no thanks", "no thank you", "Yes, please", "No, thanks", "No, thank you", "Sure", "Do it now, please", "Sure, why not",
#     "Go for it."
    

#     # Multiple intents
#     "turn on the light and play some music", "lights on and play some jazz", "turn off the light after 20 minutes",
#     "turn off the light in 5 minutes", "play your favorite music and dim the lights",
#     "dim the lights and start the movie", "lights out and set alarm for 6 am",
#     "brighten the room and read the news aloud", "play some music and turn on the light",
#     "edit my reminder for 3 pm, and what's on my calendar?",
#     "switch off the lights then play relaxing sounds", "light up the room and tell me the weather",
#     "turn on the lights and make coffee", "set a reminder for 10 and play music",
#     "tell me my reminders and my calendar with the lights on",
#     "activate lights and play morning playlist", "turn lights to full and open the blinds",
#     "kill the lights and start the dishwasher", "list my reminders and tell me the weather",
#     "what's on my calendar and reminders for 3pm?",
#     "lights to minimum and call my mom", "bring light and set a timer for cooking",
#     "raise the lights and schedule my meeting", "play a song and tell me the weather?",
#     "is it going to rain when my reminder is on?",
#     "deactivate the lights then send a message", "illuminate the room and find a nearby cafe",
#     "lights off please and remind me to call John", "what time are my appointments and will it rain then?",
#     "whats on my calendar and my reminders?",
# ]

# labels = [
#     # Labels for turn on the light
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch",

#     # Labels for turn off the light
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
#     "control_switch", "control_switch", "control_switch", "control_switch",
    
#     # Labels for the news
#     "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
#     "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
#     "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
#     "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
#     "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
    
#     # Labels for listing reminders
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders", "list_reminders",
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders", "list_reminders",
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders", "list_reminders",
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders", "list_reminders",
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders", "list_reminders",
#     "list_reminders", "list_reminders", "list_reminders", "list_reminders",

#     # Labels for other
#     "other", "other", "other", "other", "other",
#     "other", "other", "other", "other", "other",
#     "other", "other", "other", "other", "other",
#     "other", "other", "other", "other", "other",

#     # Labels for multiple intents
#     "multiple", "multiple", "multiple", "multiple", "multiple",
#     "multiple", "multiple", "multiple", "multiple", "multiple",
#     "multiple", "multiple", "multiple", "multiple", "multiple",
#     "multiple", "multiple", "multiple", "multiple", "multiple",
#     "multiple", "multiple", "multiple", "multiple", "multiple",
#     "multiple", "multiple", "multiple", "multiple", "multiple",
# ]

# # from sklearn.feature_extraction.text import TfidfVectorizer

# # # Initialize the vectorizer
# # vectorizer = TfidfVectorizer()

# # # Fit and transform the phrases
# # X = vectorizer.fit_transform(phrases)
# # from sklearn.linear_model import LogisticRegression

# # # Initialize the classifier
# # classifier = LogisticRegression()

# # # Train the classifier
# # classifier.fit(X, labels)

# # # Save the classifier
# # joblib.dump(classifier, 'light_intent_classifier.pkl')

# # # Save the vectorizer
# # joblib.dump(vectorizer, 'vectorizer.pkl')
# # # Load the classifier and vectorizer
# classifier = joblib.load('light_intent_classifier.pkl')
# vectorizer = joblib.load('vectorizer.pkl')


# # Function to predict intent
# def predict_intent(command, classifier, vectorizer):
#     X = vectorizer.transform([command])
#     return classifier.predict(X)[0]

# # # Example usage
# # command = "tell me about the BBC"
# # intent = predict_intent(command, classifier, vectorizer)
# # phrases = [
# #     "turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights", "illuminate the room", 
# #     "light up", "brighten the room", "activate the lights", "bring light", "let there be light", "shine the light", "enable lighting", "turn lights to full", 
# #     "light this place up", "more light please", "lights to maximum", "open the blinds", "raise the lights", "lights to 100%",
# #     "remind me to call john", "how do I get to the nearest gas station", "play the news", "play the latest podcast", "find a nearby restaurant",
# #     ]
# # for command in phrases:
# #     intent = predict_intent(command)
# #     print(f"Command: {command} - Intent: {intent}")

# # from transformers import BertTokenizer, BertModel
# # import torch
# # import numpy as np

# # # Function to encode text using BERT
# # def encode_text(text, model, tokenizer):
# #     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
# #     outputs = model(**inputs)
# #     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# # # Function to compute cosine similarity
# # def cosine_similarity(a, b):
# #     # a is a single vector, b is a matrix of vectors
# #     similarities = [np.dot(a, b_vec.T) / (np.linalg.norm(a) * np.linalg.norm(b_vec)) for b_vec in b]
# #     return np.mean(similarities)

# # # Updated function to verify intent with semantic similarity
# # def verify_intent_with_semantics(request, expected_intent, classifier, vectorizer, model, tokenizer, intent_examples):
# #     predicted_intent = predict_intent(request, classifier, vectorizer)

# #     # Encode the request
# #     request_embedding = encode_text(request, model, tokenizer)

# #     # Compute similarities
# #     similarities = {}
# #     for intent, examples in intent_examples.items():
# #         example_embeddings = encode_text(examples, model, tokenizer)
# #         similarity_score = cosine_similarity(request_embedding, example_embeddings)
# #         similarities[intent] = similarity_score

# #     most_similar_intent = max(similarities, key=similarities.get)
# #     intent_match = (most_similar_intent == expected_intent) and (predicted_intent == expected_intent)
# #     return predicted_intent, intent_match, similarities

# # # Example setup
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained('bert-base-uncased')

# # # Define example phrases for each intent
# # intent_examples = {
# #     "control_switch": ["turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights",
# #     "illuminate the room", "light up", "brighten the room", "activate the lights", "bring light",
# #     "let there be light", "shine the light", "enable lighting", "turn lights to full", "light this place up",
# #     "more light please", "lights to maximum", "raise the lights", "lights to 100%",

# #     # Turn off the light phrases
# #     "turn off the light", "lights out", "can you dim the lights", "make it dark", "switch off the lights",
# #     "extinguish the lights", "darken the room", "turn the lights down", "deactivate the lights", "lights off please",
# #     "kill the lights", "shut off the lights", "lower the lights", "fade the lights",
# #     "lights to zero", "no more light", "eliminate light", "turn down lights", "lights to minimum"],
# #     # Add other intents and their example phrases here
# # }

# # # Example usage
# # request = "Do it now, please"
# # expected_intent = "other"
# # predicted_intent, intent_match, similarities = verify_intent_with_semantics(request, expected_intent, classifier, vectorizer, model, tokenizer, intent_examples)
# # print(f"Predicted Intent: {predicted_intent}, Intent Match: {intent_match}, Similarities: {similarities}")
# # request = "Tell me the weather"
# # expected_intent = "other"
# # predicted_intent, intent_match, similarities = verify_intent_with_semantics(request, expected_intent, classifier, vectorizer, model, tokenizer, intent_examples)
# # print(f"Predicted Intent: {predicted_intent}, Intent Match: {intent_match}, Similarities: {similarities}")


import spacy

# Load spaCy's medium or large model for English
# try:
#     spacy.prefer_gpu()
#     nlp = spacy.load("en_core_web_lg")
# except OSError:
#     print("Downloading language model for the spaCy POS tagger")
#     from spacy.cli import download
#     download("en_core_web_lg")
#     nlp = spacy.load("en_core_web_lg")

def predict_intent(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)

    # Define reference sentences for different intents
    ref_sentences = {
        "control_lights_on": ["Turn on the lights", "Lights on", "Activate the lights"],
        "control_lights_off": ["Turn off the lights", "Lights off", "Deactivate the lights"],
        "bbc_news_briefing": ["Tell me the news", "What's the news today", "News update, please"],
        "list_reminders": ["What are my reminders", "List my reminders for today"],
        "volume_down": ["Turn down the volume", "Lower the volume", "Volume lower, please"],
        "volume_up": ["Turn up the volume", "Increase the volume", "Volume higher, please"]
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

# Test the function
print(predict_intent("Can you turn on the lights?"))
print(predict_intent("can you turn the lights off?"))
print(predict_intent("What's the latest news today?"))
print(predict_intent("Show me my reminders for today"))
print(predict_intent("Tell me the news"))
print(predict_intent("what's the news"))
print(predict_intent("Turn down the volume"))
print(predict_intent("Turn up the volume"))
print(predict_intent("can you turn the volume down"))