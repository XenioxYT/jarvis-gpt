from utils.strings import common_bigrams, common_trigrams
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

llama_client = OpenAI(base_url=api_base, api_key=api_key)

def is_english_text(user, previous_response):
    message = [
        {
            "role": "user",
            "content":
            "You are an AI designed to class intent of a sentace for a voice assistant. You class whether the sentence makes sense regarding current context and language understanding. DO NOT give any other output other than JSON. Here is how you should format your output:"
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

def user_query(input):
    message = [
        {
            "role": "user",
            "content":
                "You are an AI designed to classify the intent of a voice command for a virtual assistant. "
                "DO NOT give any other output other than JSON."
                "Your task is to analyze the user command and determine the specific action the user wants to perform. "
                "These actions include playing music, pausing music, skipping songs, turning devices on or off. "
                "Provide your classification in a JSON format like this: "
                "{'intent': 'play_music'}, where 'intent' is the classified action."
                "These are the possible intents: "
                "resume_music, pause_music, skip_song, previous_song, turn_on_device, turn_off_device"
                "Set intent to None if the command does not fit this list or if the command is a false positive, for example 'Play my favorite song' is not a command."
                "The user command is: '" + input + "'"
        }
    ]
    response = llama_client.chat.completions.create(
        model="mistral-7b",
        messages=message
    )
    print(response)
    classification = response = response.choices[0].message.content
    if "resume_music" in classification:
        return "resume_music"
    if "pause_music" in classification:
        return "pause_music"
    if "skip_song" in classification:
        return "skip_song"
    if "previous_song" in classification:
        return "previous_song"
    if "turn_on_device" in classification:
        return "turn_on_device"
    if "turn_off_device" in classification:
        return "turn_off_device"
    else:
        return None
    
print(user_query("resume music"))