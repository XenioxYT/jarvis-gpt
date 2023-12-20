import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def google_search(query):
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')

    # Build a service object for the API
    service = build("customsearch", "v1", developerKey=api_key)

    # Perform the search
    res = service.cse().list(q=query, cx=cse_id, gl='uk').execute()

    # Extract and format the results
    results = []
    for item in res.get('items', []):
        title = item.get('title')
        snippet = item.get('snippet')
        results.append(f"Title: {title}\nSnippet: {snippet}\n")

    return '\n'.join(results)

# # Example usage
query = 'example query'
print(google_search(query))

