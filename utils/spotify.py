import os
import requests
access_token = os.getenv('HASS_TOKEN')
home_assistant_url=os.getenv('HASS_URL')
SPOTIFY_URI = 'spotify:playlist:37i9dQZF1DWUBCiFp3lAyi'
DEVICE_NAME = 'media_player.tom_s_room'

def play_spotify_uri(spotify_uri):
    """
    Play a Spotify track on a Google Home speaker using Spotcast in Home Assistant.
    
    :param ha_url: URL of the Home Assistant instance.
    :param token: Long-lived access token for Home Assistant.
    :param spotify_uri: Spotify URI of the track to play.
    :param device_name: Entity ID of the Google Home speaker.
    """
    device_name = 'media_player.tom_s_room'
    token = os.getenv('HASS_TOKEN')
    ha_url=os.getenv('HASS_URL')
    # Headers for the HTTP request
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    # Data payload for the Spotcast service call
    data = {
        'entity_id': device_name,
        'uri': spotify_uri,
    }

    # Make the request to Home Assistant
    response = requests.post(
        f'{ha_url}/api/services/spotcast/start',
        json=data,
        headers=headers,
    )

    # Check the response
    if response.status_code == 200:
        print('Spotify track is now playing on Google Home.')
    else:
        print('Failed to play Spotify track.')

def search_spotify_song(search_term):
    """
    Play a Spotify track on a Google Home speaker using Spotcast in Home Assistant.
    
    :param ha_url: URL of the Home Assistant instance.
    :param token: Long-lived access token for Home Assistant.
    :param spotify_uri: Spotify URI of the track to play.
    :param device_name: Entity ID of the Google Home speaker.
    """
    device_name = 'media_player.tom_s_room'
    token = os.getenv('HASS_TOKEN')
    ha_url=os.getenv('HASS_URL')
    # Headers for the HTTP request
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    # Data payload for the Spotcast service call
    data = {
        'entity_id': device_name,
        'search': search_term,
    }

    # Make the request to Home Assistant
    response = requests.post(
        f'{ha_url}/api/services/spotcast/start',
        json=data,
        headers=headers,
    )

    # Check the response
    if response.status_code == 200:
        print('Spotify track is now playing on Google Home.')
        return "Now playing " + search_term + " to the user. Do not say that you can't play it, as it is now playing."
    else:
        print('Failed to play Spotify track.')
        return "Failed to play " + search_term + " on Spotify."
        
# Example usage
# search_spotify_song("is there someone else")

def toggle_spotify_playback(force_play=False):
    """
    Toggle play/pause on a Spotify media player in Home Assistant. If force_play is True,
    it toggles playback regardless of the current state. Otherwise, it only toggles if Spotify is currently playing.
    
    :param force_play: Boolean to determine whether to toggle playback regardless of the current state.
    """
    device_name = 'media_player.tom_s_room'
    token = os.getenv('HASS_TOKEN')
    ha_url = os.getenv('HASS_URL')

    # Check if Spotify is playing, skip if force_play is True
    if not force_play and not is_spotify_playing_on_device():
        print('Spotify is not playing. No action taken.')
        return

    # Headers for the HTTP request
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    # Toggle the playback
    response = requests.post(
        f'{ha_url}/api/services/media_player/media_play_pause',
        json={'entity_id': device_name},
        headers=headers,
    )

    if response.status_code == 200:
        print('Spotify playback toggled.')
    else:
        print('Failed to toggle Spotify playback.')
        
def is_spotify_playing_on_device():
    """
    Check if Spotify is playing on a specific device in Home Assistant.

    :param ha_url: URL of the Home Assistant instance.
    :param token: Long-lived access token for Home Assistant.
    :param entity_id: Entity ID of the Spotify media player.
    :return: True if Spotify is playing, False otherwise.
    """
    device_name = 'media_player.tom_s_room'
    token = os.getenv('HASS_TOKEN')
    ha_url=os.getenv('HASS_URL')
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    response = requests.get(
        f'{ha_url}/api/states/{device_name}',
        headers=headers,
    )

    if response.status_code == 200:
        state_data = response.json()
        return state_data['state'] == 'playing'
    else:
        print('Failed to retrieve the state of the Spotify player.')
        return False