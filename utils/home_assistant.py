import requests
import os
from dotenv import load_dotenv

load_dotenv()


# Proposed optimization:
# 1. Simplified the conversion of the 'switch' variable to a boolean.
# 2. Added support for the 'light' entity.
# 3. Optimized string formatting and code structure for better efficiency and clarity.

def toggle_entity(entity_id, switch):
    # Convert the switch parameter to boolean more efficiently
    switch = str(switch).lower() == 'true'

    access_token = os.getenv('HASS_TOKEN')
    home_assistant_url = os.getenv('HASS_URL')

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
# entity_id = "switch.desk_lamp_socket_1" 
# toggle_entity(entity_id, True)
