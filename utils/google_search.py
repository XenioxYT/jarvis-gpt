import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

# def google_search(query):
#     api_key = os.getenv('GOOGLE_API_KEY')
#     cse_id = os.getenv('GOOGLE_CSE_ID')

#     # Build a service object for the API
#     service = build("customsearch", "v1", developerKey=api_key)

#     # Perform the search
#     res = service.cse().list(q=query, cx=cse_id, gl='uk').execute()

#     # Extract and format the results
#     results = []
#     for item in res.get('items', []):
#         title = item.get('title')
#         snippet = item.get('snippet')
#         results.append(f"Title: {title}\nSnippet: {snippet}\n")

#     return '\n'.join(results)

# # Example usage
# query = 'example query'
# print(google_search(query))


import requests


def get_ai_snippets_for_query(query):
    headers = {"X-API-Key": '8f39d52a-ada3-477f-a71c-ec70924ffe1a<__>1OLZBeETU8N2v5f4H0usXeqV'}
    params = {"query": query}
    return requests.get(
        f"https://api.ydc-index.io/search?query={query}",
        params=params,
        headers=headers,
    ).json()


results = get_ai_snippets_for_query("reasons to use AC for long power lines")
print(results)
