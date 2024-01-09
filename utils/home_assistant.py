from collections.abc import Mapping
import requests
import os
from dotenv import load_dotenv
load_dotenv()


def toggle_entity(entity_id, switch):
    # Convert the switch parameter to boolean more efficiently
    switch = str(switch).lower() == 'true'
    try:
        access_token = os.getenv('home_assistant_token')
        home_assistant_url = os.getenv('home_assistant_url')
    except:
        return "Home Assistant token or URL not found. Tell the user to set this in the setup -> configuration section."

    # Extract the entity type and determine the service endpoint
    entity_type = entity_id.split(".")[0]
    if entity_type in ["switch", "light"]:
        service = "turn_on" if switch else "turn_off"
    else:
        print(f"Entity type '{entity_type}' not supported yet.")
        return

    url = f"{home_assistant_url}/api/services/{entity_type}/{service}"
    headers = {"Authorization": f"Bearer {access_token}", "content-type": "application/json"}
    data = {"entity_id": entity_id}

    # Sending the POST request to Home Assistant
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        action = "turned on" if switch else "turned off"
        print(f"{entity_id} {action} successfully.")
        return f"{entity_id} {action} successfully."
    else:
        print(f"Failed to control {entity_id}. Response:", response.text)

# Example usage
# entity_id = "switch.bedroom_draws_light" 
# toggle_entity(entity_id, False)

def list_light_switch_entities():
    try:
        access_token = os.getenv('home_assistant_token')
        home_assistant_url = os.getenv('home_assistant_url')
    except:
        return "Home Assistant token or URL not found. Tell the user to set this in the setup -> configuration section."

    url = f"{home_assistant_url}/api/states"
    headers = {"Authorization": f"Bearer {access_token}"}

    # Sending the GET request to Home Assistant
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        all_entities = response.json()
        light_switch_entities = [entity['entity_id'] for entity in all_entities if entity['entity_id'].split('.')[0] in ['light', 'switch']]
        return light_switch_entities
    else:
        print(f"Failed to fetch entities. Response:", response.text)
        return []

# light_switch_entities = list_light_switch_entities()
# print(light_switch_entities)