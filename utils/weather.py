import requests
from datetime import datetime
import json
from dateutil import parser as date_parser
import os
from dotenv import load_dotenv

load_dotenv()

openweather_api_key = os.getenv("openweather_api_key")


def get_weather_data(location, date=None):
    location = location.split(",")[0]  # Extracting just the city name

    # Geocoding to get latitude and longitude
    geocode_response = requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={location}&appid={openweather_api_key}").json()
    if not geocode_response:
        return json.dumps({"error": "Location not found"}, indent=2)

    lat, lon = geocode_response[0]['lat'], geocode_response[0]['lon']
    weather_api_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&appid={openweather_api_key}&units=metric"
    response = requests.get(weather_api_url).json()

    response_data = {"location": location, "data": []}
    current_date = datetime.utcfromtimestamp(response['current']['dt']).date()

    try:
        if date:
            if ' - ' in date:
                # Handling date range
                start_date_str, end_date_str = date.split(' - ')
                start_date, end_date = date_parser.parse(start_date_str.strip()).date(), date_parser.parse(end_date_str.strip()).date()
                response_data['date'] = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
                response_data['data'] = response['daily']  # Assuming full range is required
            else:
                # Handling single date
                single_date = date_parser.parse(date.strip()).date()
                response_data['date'] = single_date.strftime('%Y-%m-%d')
                for weather_data in response['daily']:
                    if datetime.utcfromtimestamp(weather_data['dt']).date() == single_date:
                        response_data['data'] = weather_data
                        break
        else:
            # Default to current weather data
            response_data.update({
                'date': current_date.strftime('%Y-%m-%d'),
                'data': {
                    'current': response['current'],
                    'daily_forecast': response['daily'][0]  # Today's forecast
                }
            })

        if not response_data['data']:
            return json.dumps({"error": f"No weather data found for the date {date}"}, indent=2)
        return json.dumps(response_data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
