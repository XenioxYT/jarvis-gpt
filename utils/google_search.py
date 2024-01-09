import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def google_search(query):
    try:
        api_key = os.getenv('google_api_key')
        cse_id = os.getenv('google_cse_id')
    except:
        return "API key or CSE ID not found. Tell the user to set this in the setup -> configuration section."

    # Build a service object for the API
    try:
        service = build("customsearch", "v1", developerKey=api_key)
    except:
        return "Invalid Google search API key. Tell the user to set this in the setup -> configuration section."

    # Perform the search
    try:
        res = service.cse().list(q=query, cx=cse_id, gl='uk').execute()
    except:
        return "Invalid Google CSE ID. Tell the user to set this in the setup -> configuration section."

    # Extract and format the results
    results = []
    for item in res.get('items', []):
        title = item.get('title')
        snippet = item.get('snippet')
        url = item.get('link')
        results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {url}\n")

    return '\n'.join(results)

# # Example usage
# query = 'example query'
# print(google_search(query))

