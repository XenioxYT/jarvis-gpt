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
            "name": "enroll_user",
            "description": "Make the sentence you give one that they will ask you, for example 'Tell me the weather in'. It doesn't have to match exactly, but it should be similar. Needs to be in the form [sentence] [name of user]",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the user.",
                    }
                },
                "required": ["name"],
            },
        },
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
                    },
                    "username": {
                        "type": "string",
                        "description": "The user to check the calendar for. At the beginning of the message.",
                    }
                },
                "required": ["date", "username"],
            },
        },
    },
    {
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
                    "username": {
                        "type": "string",
                        "description": "The user to add the event for.",
                    }
                },
                "required": ["summary", "start", "end", "username"],
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
            "name": "list_reminders",
            "description": "List all reminders that have not yet been notified",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bbc_news_briefing",
            "description": "Play a BBC News briefing to the user. It will be played after your next response. Don't ask the user if they want to hear the news, as it will play automatically.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "volume_up",
            "description": "Increase the volume by 10%.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "volume_down",
            "description": "Decrease the volume by 10%.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_switch",
            "description": "Toggle the state of an entity in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID to toggle. These entities are: light.dad_s_house_bedroom'"
                    },
                    "switch": {
                        "type": "boolean",
                        "description": "The switch parameter to turn the entity on or off. True for on, False for off.",
                    },
                },
                "required": ["entity_id", "switch"],
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
                "required": ["search_term"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_phone",
            "description": "Send a message to the user's phone, for example links, calendar events or reminders. Link the user to their username. Format the message using markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The username of the user to send the message to. Use the references: ['user' = 'username'], ['Tom' = 'xeniox']",
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to send to the user's phone.",
                    },
                },
                "required": ["username", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note for the user. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the note."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text content of the note."
                    }
                },
                "required": ["title", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_notes",
            "description": "Retrieves all notes for a given user, formatted with title, text, and creation date.",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_or_delete_notes",
            "description": "Edits or deletes a note for a given user. If multiple notes with the same title are found, it lists all matches. Otherwise, it updates or deletes the note based on the new title and text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The search title of the note to be edited or deleted."
                    },
                    "new_title": {
                        "type": "string",
                        "description": "The new title for the note. If not provided but new_text is, only the text will be updated. If neither is provided, the note will be deleted."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new text for the note. If not provided but new_title is, only the title will be updated. If neither is provided, the note will be deleted."
                    },
                    "index": {
                        "type": "integer",
                        "description": "The index of the note to be edited or deleted if multiple notes with the same title are found."
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
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
    },
    {
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
    },
]