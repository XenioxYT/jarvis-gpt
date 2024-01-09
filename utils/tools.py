# from utils.home_assistant import list_light_switch_entities
import sqlite3

conn = sqlite3.connect('../jarvis-setup/jarvisSetup/db.sqlite3') # TODO: Change this to the correct path when setup is implemented in the main project
id, live_weather, bbc_news, google_search, google_calendar_check, google_calendar_add, google_maps_directions, google_maps_info, set_reminder, list_reminder, edit_reminder, smart_home_control, discord_message, save_notes, list_notes, edit_notes = conn.execute('SELECT * FROM webserver_tools').fetchone()
conn.close()

tools = []

if live_weather:
    tools.append({
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "Get the weather data for a specific location and optionally for a particular date or date range using OpenWeatherMap API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., York",
                    },
                    "date": {
                        "type": "string",
                        "description": "An optional date (or date range) to fetch the weather for, e.g., '2023-09-12' or '2023-11-12 - 2023-11-15'. For current weather, this parameter can be omitted. Specify a range for upcoming days. ",
                    }
                },
                "required": ["location"],
            },
        }
    })

if bbc_news:
    tools.append({
        "type": "function",
        "function": {
            "name": "bbc_news_briefing",
            "description": "Play a BBC News briefing to the user. It will be played after your next response. Don't ask the user if they want to hear the news, as it will play automatically.",
        },
    })
    
if google_search:
    tools.append({
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search Google for a given query and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    })
    
if google_calendar_check:
    tools.append({
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check the calendar for a given date. Give a very concise overview, and ask the user if they'd like to hear more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check, e.g., 2021-10-31. You can also give a range, e.g., 2021-10-31 - 2021-11-01",
                    },
                    "username": {
                        "type": "string",
                        "description": "The user to check the calendar for. At the beginning of the message.",
                    }
                },
                "required": ["date", "username"],
            },
        },
    })
    
if google_calendar_add:
    tools.append({
        "type": "function",
        "function": {
            "name": "add_event_to_calendar",
            "description": "Add a new event to the Google Calendar with specified details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "The location of the event (optional).",
                        "optional": True,
                    },
                    "description": {
                        "type": "string",
                        "description": "A description of the event (optional).",
                        "optional": True,
                    },
                    "start": {
                        "type": "string",
                        "description": "The start date and time of the event in ISO format, e.g., '2023-07-21T15:00:00-07:00'.",
                    },
                    "end": {
                        "type": "string",
                        "description": "The end date and time of the event in ISO format, e.g., '2023-07-21T16:00:00-07:00'.",
                    },
                    "username": {
                        "type": "string",
                        "description": "The user to add the event for.",
                    }
                },
                "required": ["summary", "start", "end", "username"],
            },
        },
    })
    
if google_maps_directions:
    tools.append({
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": "Get directions from Google Maps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_location": {
                        "type": "string",
                        "description": "The start location of the route.",
                    },
                    "end_location": {
                        "type": "string",
                        "description": "The end location of the route.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "The mode of transport. Can be 'driving', 'walking', 'bicycling', or 'transit'. Defaults to 'driving'.",
                    }
                },
                "required": ["start_location", "end_location"],
            },
        },
    })
    
if google_maps_info:
    tools.append({
        "type": "function",
        "function": {
            "name": "search_places",
            "description": "Search Google maps places for locations. Gives the name, address, types, website, google maps link, and rating of a given place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_query": {
                        "type": "string",
                        "description": "The query to search for.",
                    }
                },
                "required": ["text_query"],
            },
        },
    })
    
if set_reminder:
    tools.append({
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specified time",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_text": {
                        "type": "string",
                        "description": "The content of the reminder",
                    },
                    "reminder_time": {
                        "type": "string",
                        "description": "The time for the reminder in format 'YYYY-MM-DD HH:MM'",
                    },
                },
                "required": ["reminder_text", "reminder_time"],
            },
        },
    })
    
if list_reminder:
    tools.append({
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all reminders that have not yet been notified",
        },
    })
    
if edit_reminder:
    tools.append({
        "type": "function",
        "function": {
            "name": "edit_reminder",
            "description": "Edit an existing reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_text": {
                        "type": "string",
                        "description": "Natural language text to describe the reminder to be edited",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new content for the reminder (optional)",
                        "optional": True,
                    },
                    "new_time": {
                        "type": "string",
                        "description": "The new time for the reminder in format 'YYYY-MM-DD HH:MM' (optional)",
                        "optional": True,
                    }
                },
                "required": ["search_text"],
            },
        },
    })

print(tools)