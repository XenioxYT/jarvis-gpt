# Map function names to their TTS messages
# from main import enroll_user_handler
from calendar_utils import check_calendar, add_event_to_calendar

from news.bbc_news import download_bbc_news_summary

from utils.google_search import google_search
from utils.notes import edit_or_delete_notes, retrieve_notes, save_note
from utils.send_to_discord import send_message_sync
from utils.volume_control import volume_down, volume_up
from utils.weather import get_weather_data
from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders
from utils.home_assistant import toggle_entity
from utils.spotify import search_spotify_song

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

available_functions = {
    "get_weather_data": get_weather_data,
    "check_calendar": check_calendar,
    "set_reminder": add_reminder,
    "edit_reminder": edit_reminder,
    "list_reminders": list_unnotified_reminders,
    "add_event_to_calendar": add_event_to_calendar,
    "control_switch": toggle_entity,
    "play_song_on_spotify": search_spotify_song,
    # "enroll_user": enroll_user_handler,
    "google_search": google_search,
    "bbc_news_briefing": download_bbc_news_summary,
    "send_to_phone": send_message_sync,
    "volume_up": volume_up,
    "volume_down": volume_down,
    "save_note": save_note,
    "retrieve_notes": retrieve_notes,
    "edit_or_delete_notes": edit_or_delete_notes,
}