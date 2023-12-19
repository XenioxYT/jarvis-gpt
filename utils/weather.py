import requests
import datetime
import json
from dateutil import parser as date_parser
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def get_weather_data(location, date=None):
    location = location.split(",")[0]  # Assuming the country is not needed in the API call

    # Geocoding to get latitude and longitude
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&appid={OPENWEATHER_API_KEY}"
    geocode_response = requests.get(geocode_url).json()

    if not geocode_response:
        return json.dumps({"error": "Location not found"}, indent=2)

    lat = geocode_response[0]['lat']
    lon = geocode_response[0]['lon']

    weather_api_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        # Fetch the weather data
        response = requests.get(weather_api_url).json()

        # Prepare the return data structure
        response_data = {
            "location": location,
            "data": []
        }

        # Get current date for comparison
        current_date = datetime.datetime.utcfromtimestamp(response['current']['dt'])
        if date:
            single_date = None  # variable for handling a single date
            # Check if we have a date range or a single date
            if ' - ' in date:
                # User provided a date range
                start_date_str, end_date_str = date.split(' - ')  # Safe to unpack
                start_date = date_parser.parse(start_date_str.strip()).date()
                end_date = date_parser.parse(end_date_str.strip()).date()
                response_data['data'] = response['daily']  # assume you want all daily data
                response_data['date'] = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            else:
                # User provided a single date
                single_date = date_parser.parse(date.strip()).date()
                response_data['date'] = single_date.strftime("%Y-%m-%d")

            if single_date:  # Handle single date case differently
                # Find and return the corresponding data
                for weather_data in response['daily']:
                    weather_date = datetime.datetime.utcfromtimestamp(weather_data['dt']).date()
                    if weather_date == single_date:
                        response_data['data'] = weather_data
                        break

        else:
            # No date specified, default to today's weather
            response_data['date'] = datetime.datetime.utcfromtimestamp(response['current']['dt']).strftime('%Y-%m-%d')
            response_data['data'] = {
                'current': response['current'],
                'daily_forecast': response['daily'][0]  # Today's forecast from the daily data
            }

        if not response_data['data']:
            return json.dumps({"error": f"No weather data found for the date {date}"}, indent=2)
        return json.dumps(response_data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
