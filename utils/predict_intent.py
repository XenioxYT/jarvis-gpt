import requests

def get_intent_from_api(text):
    url = "https://api.xeniox.tv/chat"
    headers = {'Content-Type': 'application/json'}
    payload = {"message": text}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Assuming the API returns a JSON with the intent in a key named 'intent'
        intent = response.json().get('intent')
        if intent == 'control_switch_on' or intent == 'control_switch_off':
            intent = 'control_switch'
        return intent

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return "other"
    except Exception as err:
        print(f"An error occurred: {err}")
        return "other"

# # Example usage
# text = "what's the news?"
# intent = get_intent_from_api(text)
# print(intent)


# intents:
#   - control_switch_on  -> remap to control_switch
#   - control_switch_off -> remap to control_switch
#   - bbc_news_briefing
#   - volume_up
#   - volume_down
#   - other