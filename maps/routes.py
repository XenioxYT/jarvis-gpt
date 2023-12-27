import requests

def convert_km_to_miles(km):
    """ Convert kilometers to miles. """
    return km * 0.621371

def format_steps(steps):
    """ Format the steps of each leg for better readability. """
    formatted_steps = []
    for step in steps:
        formatted_step = {
            'instruction': step['html_instructions'],
            'distance': f"{convert_km_to_miles(step['distance']['value'] / 1000):.2f} miles",
            'duration': step['duration']['text'],
            'travel_mode': step['travel_mode']
        }
        formatted_steps.append(formatted_step)
    return formatted_steps

def get_directions(api_key, start_location, end_location):
    """
    Get directions from Google Maps Directions API and format the output.

    :param api_key: Your Google Maps API key.
    :param start_location: The starting location (address).
    :param end_location: The destination location (address).
    :return: Formatted directions.
    """

    base_url = "https://maps.googleapis.com/maps/api/directions/json?"
    params = {
        'origin': start_location,
        'destination': end_location,
        'key': api_key
    }

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        return "Error: " + str(response.status_code)

    directions = response.json()
    formatted_output = {'status': directions['status'], 'routes': []}

    for route in directions['routes']:
        formatted_route = {
            'summary': route['summary'],
            'copyrights': route['copyrights'],
            'legs': []
        }
        for leg in route['legs']:
            formatted_leg = {
                'start_address': leg['start_address'],
                'end_address': leg['end_address'],
                'distance': f"{convert_km_to_miles(leg['distance']['value'] / 1000):.2f} miles",
                'duration': leg['duration']['text'],
                'steps': format_steps(leg['steps'])
            }
            formatted_route['legs'].append(formatted_leg)
        formatted_output['routes'].append(formatted_route)

    return formatted_output

# Example usage
api_key = ''  # Replace with your actual API key
start_location = 'S66 1fq'
end_location = 'End s60 4bx'
directions = get_directions(api_key, start_location, end_location)

print(directions)
