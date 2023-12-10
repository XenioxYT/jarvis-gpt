tools = [
    {
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
    },
    {
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
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
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
                },
                "required": ["summary", "start", "end"],
            },
        },
    },
    {
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
    },
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "list_unnotified_reminders",
            "description": "List all reminders that have not yet been notified",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy_variable": {
                        "type": "string",
                        "description": "This is a dummy variable to allow the function to be called without any parameters.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_device",
            "description": "Toggle the state of an entity in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID to toggle. These entities are: switch.desk_lamp_socket_1'"
                    }
                },
                "required": ["entity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_song_on_spotify",
            "description": "Searches for a song on Spotify and plays it. The function requires the search term for the song.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "The search term for the song, e.g., 'Shape of You - Ed Sheeran'."
                    },
                },
            }
        }
    },
]