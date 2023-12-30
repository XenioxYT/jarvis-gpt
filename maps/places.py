import os
import requests
import json

def search_places(text_query, max_result_count=5, price_levels=None, open_now=None):
    """
    Perform a Text Search request to Google Places API.

    :param api_key: Your Google Maps API key.
    :param text_query: The text string on which to search.
    :param field_mask: Fields to return in the response.
    :param location_bias: Optional location bias (circle or rectangle).
    :param max_result_count: Optional max number of results to return.
    :param price_levels: Optional price levels to filter.
    :param open_now: Optional flag to filter places open now.
    :return: Response from the API as a JSON object.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
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
        return f"Error: {response.status_code}, {response.text}"


def get_specific_place_results(text_query, max_result_count=1):
    """
    Perform a Text Search request to Google Places API.

    :param api_key: Your Google Maps API key.
    :param text_query: The text string on which to search.
    :param field_mask: Fields to return in the response.
    :param max_result_count: Optional max number of results to return.
    :return: Response from the API as a JSON object.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
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
        return f"Error: {response.status_code}, {response.text}"