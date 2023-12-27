import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import json
import threading
import time


from utils.send_to_discord import send_message_sync

SCOPES = ['https://www.googleapis.com/auth/calendar']


def authenticate_google_calendar_api(username):
    creds = None
    if os.path.exists(f'./tokens/{username}/token.json'):
        creds = Credentials.from_authorized_user_file(f'./tokens/{username}/token.json', SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # creds = None

    if not creds:
        redirect_uri = f'https://calendar.xeniox.tv/oauth2callback'
        state = f'user_{username}'  # Simple example, can be more complex/encoded
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES, redirect_uri=redirect_uri)
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline', state=state)

        # Send the URL to the user
        send_message_sync(username, f"Please visit this URL to authorize this application: {auth_url}")

        # Wait for the user to complete authentication
        # timeout = time.time() + 7200  # 2 hours timeout
        # while not os.path.exists('./tokens/{user}/token.json') and time.time() < timeout:
        #     time.sleep(5)  # Check every 5 seconds

        if not os.path.exists(f'./tokens/{username}/token.json'):
            print("Authentication process timed out or failed.")
            return False

        else:
            creds = Credentials.from_authorized_user_file(f'./tokens/{username}/token.json', SCOPES)
            service = build('calendar', 'v3', credentials=creds)
            return service
    else:
        service = build('calendar', 'v3', credentials=creds)
        return service

        
def check_calendar(date, username):
    """Check the calendar for events on a given date or date range."""
    
    if service == False:
        return "You have sent the user a message to their phone with the authentication link. Please wait for them to authenticate and try again."
    service = authenticate_google_calendar_api(username)
    
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