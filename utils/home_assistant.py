import requests
import os
from dotenv import load_dotenv

load_dotenv()

def toggle_entity(entity_id):
    access_token = os.getenv('HASS_TOKEN')
    home_assistant_url=os.getenv('HASS_URL')
    # Determine the type of the entity (e.g., 'switch')
    entity_type = entity_id.split(".")[0]

    # Define the service endpoint based on the entity type
    # For now, it only supports 'switch'
    if entity_type == "switch":
        service = "toggle"
    else:
        print(f"Entity type '{entity_type}' not supported yet.")
        return

    # Endpoint to toggle the state of the entity
    url = f"{home_assistant_url}/api/services/{entity_type}/{service}"

    # Headers including the long-lived access token for authentication
    headers = {
        "Authorization": f"Bearer {access_token}",
        "content-type": "application/json",
    }

    # Data payload specifying the entity_id
    data = {
        "entity_id": entity_id
    }

    # Sending the POST request to Home Assistant
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        print(f"{entity_id} toggled successfully.")
        return entity_id + " toggled successfully."
    else:
        print(f"Failed to toggle {entity_id}. Response:", response.text)

# Example usage
# entity_id = "switch.desk_lamp_socket_1" 
# toggle_entity(entity_id)