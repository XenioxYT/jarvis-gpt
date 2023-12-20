import joblib

phrases = [
    # Turn on the light phrases
    "turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights",
    "illuminate the room", "light up", "brighten the room", "activate the lights", "bring light",
    "let there be light", "shine the light", "enable lighting", "turn lights to full", "light this place up",
    "more light please", "lights to maximum", "raise the lights", "lights to 100%",

    # Turn off the light phrases
    "turn off the light", "lights out", "can you dim the lights", "make it dark", "switch off the lights",
    "extinguish the lights", "darken the room", "turn the lights down", "deactivate the lights", "lights off please",
    "kill the lights", "shut off the lights", "lower the lights", "fade the lights",
    "lights to zero", "no more light", "eliminate light", "turn down lights", "lights to minimum",
    
    # News phrases
    "tell me the news", "play the news", "play the latest news", "play the news for today", "play the news for me",
    "give me the latest from BBC News", "play today's news summary", "I'd like to hear the news from BBC", 
    "what's the latest news on BBC", "BBC news update please", "can you play the top stories from BBC News",  
    "start the BBC news briefing", "play BBC's news report", "I need to hear the morning news from BBC", 
    "what are the evening headlines on BBC", "play the most recent news from BBC", "BBC News, please",
    "BBC's main news stories today",
    "can you play the midday news from BBC", "show me the latest news broadcast from BBC", 
    "what's the current news on BBC",
    "tell me today's headlines from BBC", "BBC's latest news coverage, please", "play the news headlines by the BBC",

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
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch",

    # Labels for turn off the light
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch", "control_switch",
    "control_switch", "control_switch", "control_switch", "control_switch",
    
    # Labels for the news
    "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
    "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
    "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
    "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",
    "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing", "bbc_news_briefing",

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

# # Save the classifier
# joblib.dump(classifier, 'light_intent_classifier.pkl')

# # Save the vectorizer
# joblib.dump(vectorizer, 'vectorizer.pkl')
# # Load the classifier and vectorizer
# classifier = joblib.load('light_intent_classifier.pkl')
# vectorizer = joblib.load('vectorizer.pkl')


# Function to predict intent
def predict_intent(command, classifier, vectorizer):
    X = vectorizer.transform([command])
    return classifier.predict(X)[0]

# # Example usage
# command = "tell me about the BBC"
# intent = predict_intent(command)
# phrases = [
#     "turn on the light", "lights on", "can you light up the room", "make it brighter", "switch on the lights", "illuminate the room", 
#     "light up", "brighten the room", "activate the lights", "bring light", "let there be light", "shine the light", "enable lighting", "turn lights to full", 
#     "light this place up", "more light please", "lights to maximum", "open the blinds", "raise the lights", "lights to 100%",
#     "remind me to call john", "how do I get to the nearest gas station", "play the news", "play the latest podcast", "find a nearby restaurant",
#     ]
# for command in phrases:
#     intent = predict_intent(command)
#     print(f"Command: {command} - Intent: {intent}")