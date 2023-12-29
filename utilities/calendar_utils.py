import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import json
import threading
import time
import uuid
import requests
from device_control.send_to_discord import send_message_sync

SCOPES = ['https://www.googleapis.com/auth/calendar']


def authenticate_google_calendar_api(username):
    creds = None
    token_path = f'../user_tokens/{username}/token.json'
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
    else:
        session_id = str(uuid.uuid4())
        state = f'{session_id}_user_{username}'
        redirect_uri = 'https://auth.xeniox.tv/oauth2callback'
        flow = InstalledAppFlow.from_client_secrets_file(
            '../credentials/credentials.json', SCOPES, redirect_uri=redirect_uri)
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline', state=state)

        send_message_sync(username, f"Please visit this URL to authorize this application: {auth_url}")

        # Start the polling in a separate thread
        polling_thread = threading.Thread(target=poll_for_token, args=(session_id, username))
        polling_thread.start()

    if creds:
        service = build('calendar', 'v3', credentials=creds)
        return service
    else:
        return False

def poll_for_token(session_id, username, timeout=7200, interval=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f'https://auth.xeniox.tv/download_token/{session_id}')
        if response.status_code == 200:
            token_data = response.json().get('token')
            save_token_locally(username, token_data)
            print("Token successfully retrieved and saved.")
            return
        time.sleep(interval)
    print("Authentication process timed out or failed.")

def save_token_locally(username, token_data):
    user_token_dir = f'../user_tokens/{username}'
    os.makedirs(user_token_dir, exist_ok=True)
    with open(f'{user_token_dir}/token.json', 'w') as token_file:
        token_file.write(token_data)

        
def check_calendar(date, username):
    """Check the calendar for events on a given date or date range."""
    
    service = authenticate_google_calendar_api(username)
    if service == False:
        return "You have sent the user a message to their phone with the authentication link. Please wait for them to authenticate and try again."
    
    
    date_range = date.split(" - ")
    start_date_str = date_range[0]
    end_date_str = date_range[1] if len(date_range) > 1 else start_date_str
    try:
        # Parse dates from strings and create date range for the query
        time_min = datetime.datetime.fromisoformat(start_date_str).isoformat() + 'Z'
        time_max = (datetime.datetime.fromisoformat(end_date_str) + datetime.timedelta(days=1)).isoformat() + 'Z'

        # Call the Google Calendar API
        events_result = service.events().list(calendarId='primary', timeMin=time_min,
                                              timeMax=time_max, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
        print(f"Found {len(events)} events")

        # Prepare the list of events in the required output format
        event_list = [
            {
                "summary": event["summary"],
                "location": event.get("location", "No location specified"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "description": event.get("description", "No description provided")
            } 
            for event in events
        ]

        # Output the events for the given date range
        return json.dumps({"date": date, "events": event_list})
    except Exception as e:
        print(f"An error occurred: {e}")
        return json.dumps({"date": date, "error": str(e), "events": []})
    

def manage_google_calendar(operation, username, event_data=None, event_id=None, date=None):
    service = authenticate_google_calendar_api(username)

    if operation == 'add':
        event = service.events().insert(calendarId='primary', body=event_data).execute()
        print(f"Event created: {event.get('htmlLink')}")
        return event.get('id')

    elif operation == 'remove':
        if event_id:
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            print("Event removed")
        else:
            print("No event ID provided to remove an event")

    elif operation == 'list':
        if date:
            date_range = date.split(" - ")
            start_date_str = date_range[0]
            end_date_str = date_range[1] if len(date_range) > 1 else start_date_str
            time_min = datetime.datetime.fromisoformat(start_date_str).isoformat() + 'Z'
            time_max = (datetime.datetime.fromisoformat(end_date_str) + datetime.timedelta(days=1)).isoformat() + 'Z'
            events_result = service.events().list(calendarId='primary', timeMin=time_min,
                                                  timeMax=time_max, singleEvents=True,
                                                  orderBy='startTime').execute()
            events = events_result.get('items', [])
            for event in events:
                print(event['summary'], event['start']['dateTime'], event['end']['dateTime'])
        else:
            print("No date provided to list events")
            
def add_event_to_calendar(title, start, end, username, location="None", description="None"):
    """
    Adds an event to the Google Calendar with the specified details.

    :param title: The title or summary of the event.
    :param start: The start date and time of the event in ISO format.
    :param end: The end date and time of the event in ISO format.
    :param location: The location of the event (optional).
    :param description: A description of the event (optional).
    :return: The event ID of the created event.
    """
    service = authenticate_google_calendar_api(username)
    
    if service == False:
        return "You have sent the user a message with the authentication link. Please wait for them to authenticate and try again."

    # Construct the event dictionary
    event = {
        'summary': title,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start,  # e.g., '2023-07-21T15:00:00-07:00'
            'timeZone': 'Europe/London',  # This should be changed to the user's preferred time zone
        },
        'end': {
            'dateTime': end,
            'timeZone': 'Europe/London',  # This should be changed to the user's preferred time zone
        },
    }

    # Add optional parameters if they are provided
    if location!="None":
        event['location'] = location
    if description!="None":
        event['description'] = description

    # Call the Google Calendar API to create the event
    try:
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {created_event.get('htmlLink')}")
        return "Event created " + created_event.get('htmlLink') + " Title: " + title + " Start: " + start + " End: " + end + " Location: " + location + " Description: " + description + " ID: " + created_event.get('id')
    except Exception as e:
        print(f"An error occurred when trying to add the event: {e}")
        return f"An error occurred when trying to add the event: {e}"
    
# print(add_event_to_calendar("Test Event", "2021-07-21T15:00:00-07:00", "2021-07-21T16:00:00-07:00", "xeniox"))