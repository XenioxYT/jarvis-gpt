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

# example usage for send_message_sync(username, message)

# def authenticate_google_calendar_api():
#     creds = None
#     # The file token.json stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())
    
#     service = build('calendar', 'v3', credentials=creds)
#     return service


def authenticate_google_calendar_api():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())

    if not creds:
        redirect_uri = 'https://calendar.xeniox.tv:49152'
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES, redirect_uri=redirect_uri)
        auth_url, state = flow.authorization_url(prompt='consent', access_type='offline')

        # Send the URL to the user
        send_message_sync("xeniox", f"Please visit this URL to authorize this application: {auth_url}")

        # Wait for the user to complete authentication
        timeout = time.time() + 7200  # 2 hours timeout
        while not os.path.exists('token.json') and time.time() < timeout:
            time.sleep(5)  # Check every 5 seconds

        if not os.path.exists('token.json'):
            print("Authentication process timed out or failed.")
            return False

    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('calendar', 'v3', credentials=creds)
    return service

def complete_auth_flow(flow, state):
    # This will open a local server in the browser for the user to complete authentication
    creds = flow.run_local_server(port=49152, state=state)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
        
def check_calendar(date):
    """Check the calendar for events on a given date or date range."""
    service = authenticate_google_calendar_api()
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
    
def manage_google_calendar(operation, event_data=None, event_id=None, date=None):
    service = authenticate_google_calendar_api()

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
            
def add_event_to_calendar(title, start, end, location="None", description="None"):
    """
    Adds an event to the Google Calendar with the specified details.

    :param title: The title or summary of the event.
    :param start: The start date and time of the event in ISO format.
    :param end: The end date and time of the event in ISO format.
    :param location: The location of the event (optional).
    :param description: A description of the event (optional).
    :return: The event ID of the created event.
    """
    service = authenticate_google_calendar_api()

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
        return "Event created" + created_event.get('htmlLink') + " Title: " + title + " Start: " + start + " End: " + end + " Location: " + location + " Description: " + description + " ID: " + created_event.get('id')
    except Exception as e:
        print(f"An error occurred when trying to add the event: {e}")
        return f"An error occurred when trying to add the event: {e}"
    
print(check_calendar("2021-05-01 - 2021-05-31"))