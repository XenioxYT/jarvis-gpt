# import re

# def user_query(input):
#     # Normalize the input string for easier processing
#     input_normalized = input.lower().strip()

#     # Basic patterns to match various commands
#     play_pattern = re.compile(r"\bplay\b|\bstart\b|\bresume\b")
#     pause_pattern = re.compile(r"\bpause\b|\bhold\b")
#     stop_pattern = re.compile(r"\bstop\b|\bend\b|\bquit\b")
#     next_pattern = re.compile(r"\bnext\b|\bskip\b")
#     previous_pattern = re.compile(r"\bprevious\b|\bback\b")

#     # Advanced pattern to capture details after 'play' command
#     play_detail_pattern = re.compile(r"\bplay\s+(.*)")

#     # Check if input matches basic commands
#     if play_pattern.search(input_normalized):
#         # Check for specific details after 'play' command
#         detail_match = play_detail_pattern.search(input_normalized)
#         if detail_match:
#             # Extract and return the detail (e.g., artist or song name)
#             return f"play_spotify('{detail_match.group(1)}')"
#         else:
#             return "play_spotify()"

#     elif pause_pattern.search(input_normalized):
#         return "pause_spotify()"
    
#     elif stop_pattern.search(input_normalized):
#         return "stop_spotify()"
    
#     elif next_pattern.search(input_normalized):
#         return "skip_song()"
    
#     elif previous_pattern.search(input_normalized):
#         return "previous_song()"
    
#     else:
#         return None

# # Test the function with different inputs
# test_queries = ["Play The Weeknd", "pause the music", "stop the track", "skip to the next song", "go to the previous track", "Just play something", "Do you play games?"]
# for query in test_queries:
#     print(f"Query: '{query}' - Command: {user_query(query)}")


from transformers import pipeline

# Initialize the intent classification model
intent_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

def user_query(input):
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
