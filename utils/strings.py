# Map function names to their TTS messages
# from main import enroll_user_handler
# from calendar_utils import check_calendar, add_event_to_calendar

# from news.bbc_news import download_bbc_news_summary

# from utils.google_search import google_search
# from utils.notes import edit_or_delete_notes, retrieve_notes, save_note
# from utils.send_to_discord import send_message_sync
# from utils.volume_control import volume_down, volume_up
# from utils.weather import get_weather_data
# from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders
# from utils.home_assistant import toggle_entity
# from utils.spotify import search_spotify_song

import datetime
import os
from dotenv import load_dotenv

import requests

load_dotenv()

tts_messages = {
    "play_song_on_spotify": "Connecting to your speakers...",
    "set_reminder": "Accessing reminders...",
    "add_event_to_calendar": "Adding event to your calendar...",
    "get_weather_data": "Getting live weather data...",
    "check_calendar": "Checking your calendar...",
    "control_switch": "Controling your smart home...",
    "edit_reminder": "Editing your reminders...",
    "list_reminders": "Getting your reminders...",
    "enroll_user": "Learning your voice...",
    "google_search": "Searching the web...",
    "bbc_news_briefing": "Getting the latest news...",
    "send_to_phone": "Sending a message to your phone...",
    "volume_up": "Increasing the volume...",
    "volume_down": "Decreasing the volume...",
    "save_note": "Saving your note...",
    "retrieve_notes": "Retrieving your notes...",
    "edit_or_delete_notes": "Editing your notes...",
}

username_mapping = {
    "Tom": "xeniox",
    "Russell": "russell68"
}

# available_functions = {
#     "get_weather_data": get_weather_data,
#     "check_calendar": check_calendar,
#     "set_reminder": add_reminder,
#     "edit_reminder": edit_reminder,
#     "list_reminders": list_unnotified_reminders,
#     "add_event_to_calendar": add_event_to_calendar,
#     "control_switch": toggle_entity,
#     "play_song_on_spotify": search_spotify_song,
#     # "enroll_user": enroll_user_handler,
#     "google_search": google_search,
#     "bbc_news_briefing": download_bbc_news_summary,
#     "send_to_phone": send_message_sync,
#     "volume_up": volume_up,
#     "volume_down": volume_down,
#     "save_note": save_note,
#     "retrieve_notes": retrieve_notes,
#     "edit_or_delete_notes": edit_or_delete_notes,
# }


def get_location():
    api_token = os.getenv('IPINFO_TOKEN')
    headers = {
        'Authorization': 'Bearer ' + api_token
    }
    response = requests.get('https://ipinfo.io', headers=headers)
    data = response.json()
    return data.get('city', 'Unknown')


city = get_location()


messages = [
    {
        "role": "system",
        "content": "You are Jarvis, a voice-based personal assistant currently located in " + city + " and based off the GPT-4 AI model. You are speaking to the user now. "
        "The user that activated you is provded to you at the start of each message ('At [timestamp] [user] said:'), along with the date at time. Refer to them by their name. "
        "ONLY perform actions for verified users. DO NOT perform reminders or calendar management actions for 'Unknown' users. Some users require specific actions. For example, be sure to select the correct calendar/reminders/smart home control for the specific user mentioned. However, you can perform general tasks for the user. "
        "You can enroll users using the function. However, BEFORE using this function you MUST give the user a sentence to say, AND ask their name. For example: 'Tell me the weather... [name]'. Insert this name into the correct field. This is to train the model to recognize the user's voice. "
        "Make the sentence you give one that they will ask you, for example 'Tell me the weather in'" + city + "'. It doesn't have to match exactly, but it should be similar. "
        "Keep repeating this process (sentence, function), until the user's voice is recognized. "
        "You are a voice assistant, so keep responses short and concise, but maintain all the important information. Remember that some words may be spelled incorrectly due to speech-to-text errors, so keep this in mind when responding. "
        "You are equipped with a variety of tools, which you can use to perform various tasks. Do not mention you are a text-based assistant. "
        "You can send responses to the user's phone. For example, you can use this to send a summary of events from the calendar, a list of reminders or notes, or links to events/search results. Offer this option to the user. "
        "Since you are a voice assistant, you must remember to NOT include visual things, like text formatting, as this will not play well with TTS. You CAN use visual formatting for the user's phone, however. "
        # "Before using a tool, you must first give a response to the user AND THEN call the correct tool or tools. "
        # "You CANNOT call a function after giving a text response, so DO NOT say thing like 'Please hold on for a moment', instead ask the user whether they'd like you to continue. "
        "You are allowed to give opinions and thoughts to the user. Don't respond with lists of things, instead give a concise overview and ask the user if they'd like to hear more. If a list is needed, provide it more conversationally. "
        "When giving calendar events, you should give a very concise overview, and ask the user if they'd like to hear more. Don't just list them all out. "
        "ALWAYS check the calendar, weather, etc. before giving a response that includes this. Do NOT hallucinate or make up events without checking. "
        "The date and time is provided at the beginning of the message. This indicates the current date and time, and is used to give you a reference point. "
        "Use this as well to give a sense of time passing and time-contextual responses. "
        "The current date and time is: " + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
    },
]