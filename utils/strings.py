common_trigrams = [
    # Smart Home Control
    ('turn', 'off', 'lights'),
    ('adjust', 'thermostat', 'to'),
    ('play', 'next', 'song'),

    # Reminders and Alarms
    ('set', 'new', 'reminder'),
    ('cancel', 'current', 'alarm'),
    ('change', 'reminder', 'time'),

    # Information Queries
    ('what', 'time', 'is'),
    ('find', 'recipe', 'for'),
    ('how', 'to', 'make'),

    # Weather and News
    ('current', 'weather', 'forecast'),
    ('latest', 'sports', 'news'),
    ('update', 'on', 'traffic'),

    # Entertainment
    ('suggest', 'good', 'movies'),
    ('play', 'some', 'music'),
    ('find', 'popular', 'podcasts'),

    # Navigation and Travel
    ('nearest', 'gas', 'station'),
    ('best', 'restaurants', 'nearby'),
    ('public', 'transport', 'schedule'),

    # Personal Assistant Tasks
    ('send', 'quick', 'email'),
    ('organize', 'my', 'schedule'),
    ('book', 'flight', 'tickets'),

    # General Conversational Trigrams
    ('that', 'sounds', 'great'),
    ('sure', 'go', 'ahead'),
    ('not', 'right', 'now'),
    ('maybe', 'later'),
    ('that', 'helps', 'alot'),
    ('could', 'you', 'repeat'),
    ('did', 'not', 'understand'),
    ('please', 'clarify', 'that'),
    ('just', 'wondering'),
    ('by', 'the', 'way'),
    ('let', 'me', 'think')
]

common_bigrams = [
    # Affirmative Responses
    ('yes', 'please'),
    ('thank', 'you'),
    ('much', 'appreciated'),

    # Negative Responses
    ('no', 'thanks')
]

# You are an AI designed to class intent of a sentace for a voice assistant. You class whether the sentence makes sense regarding current context and language understanding. DO NOT give any other output other than JSON. Here is how you should format your output:
# {
#     "followup": true
# }
# where true is if the output is a follow up to the previous response from the assistant. This can be anything from a follow up question to a "Thank you", expressing gratitude for the answer, so long as it makes sense in the context. The assistant reply is: "Currently in Newcastle, it's 4.5 degrees and cloudy. Can I help you with anything else?"

# And the user input:
# "Thanks"