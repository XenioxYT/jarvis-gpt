import os
import requests
import json

def search_places(text_query, max_result_count=5, price_levels=None, open_now=None):
    """
    Perform a Text Search request to Google Places API.

    :param text_query: The text string on which to search.
    :param max_result_count: Optional max number of results to return.
    :param price_levels: Optional price levels to filter.
    :param open_now: Optional flag to filter places open now.
    :return: Response from the API as a JSON object.
    :rtype: dict
    :raises: Exception if the Google Maps API key is not found or if there is an error with the API request.
    """
    try:
        api_key = os.getenv("google_maps_api_key")
    except:
        return "Google Maps API key not found. Tell the user to set this in the setup -> configuration section."
    
    field_mask = 'places.displayName,places.formattedAddress,places.types,places.websiteUri,places.googleMapsUri,places.rating'
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': field_mask
    }

    data = {
        'textQuery': text_query,
        'maxResultCount': max_result_count
    }

    if price_levels:
        data['priceLevels'] = price_levels

    if open_now is not None:
        data['openNow'] = open_now

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return f"Error: {response.status_code}, {response.text}. The user may have exceeded their Google Maps API quota, or the API key may be invalid."


def get_specific_place_results(text_query, max_result_count=1):
    """
    Perform a Text Search request to Google Places API.

    :param api_key: Your Google Maps API key.
    :param text_query: The text string on which to search.
    :param field_mask: Fields to return in the response.
    :param max_result_count: Optional max number of results to return.
    :return: Response from the API as a JSON object.
    """
    try:
        api_key = os.getenv("google_maps_api_key")
    except:
        return "Google Maps API key not found. Tell the user to set this in the setup -> configuration section."
    field_mask = 'places.displayName,places.formattedAddress,places.types,places.websiteUri,places.googleMapsUri,places.rating'
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': field_mask
    }

    data = {
        'textQuery': text_query,
        'maxResultCount': max_result_count
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return f"Error: {response.status_code}, {response.text}. The user may have exceeded their Google Maps API quota, or the API key may be invalid."