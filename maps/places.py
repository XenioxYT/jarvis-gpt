import requests
import json

def search_text_places(api_key, text_query, field_mask, location_bias=None, max_result_count=5, price_levels=None, open_now=None):
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

    if location_bias:
        data['locationBias'] = location_bias

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


# Example usage
api_key = ''  # Replace with your actual API key
text_query = 'valentinos'
field_mask = 'places.displayName,places.formattedAddress,places.types,places.websiteUri,places.googleMapsUri,places.rating'
results = search_text_places(api_key, text_query, field_mask)

print(results)

def get_specific_place_results(api_key, text_query, field_mask, max_result_count=1):
    """
    Perform a Text Search request to Google Places API.

    :param api_key: Your Google Maps API key.
    :param text_query: The text string on which to search.
    :param field_mask: Fields to return in the response.
    :param max_result_count: Optional max number of results to return.
    :return: Response from the API as a JSON object.
    """

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
    
# Example usage
text_query = 'Valentinos Hair & Beauty'
field_mask = 'places.displayName,places.formattedAddress,places.types,places.websiteUri,places.googleMapsUri,places.rating,places.currentOpeningHours'
print(get_specific_place_results(api_key, text_query, field_mask))