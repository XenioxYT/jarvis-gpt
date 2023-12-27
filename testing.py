import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.google_search import google_search
# from utils.tools import tools
# from utils.strings import avaliable_functions
import os
from utils.reminders import add_reminder, edit_reminder, list_unnotified_reminders

from utils.weather import get_weather_data

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

oai_client = OpenAI(base_url=api_base, api_key=api_key)

messages = [
    {
        "role": "system",
        "content": "As ChatGPT, you are a version of ChatGPT that has been optimized for engaging in general conversation, providing informative and accurate responses across a wide range of topics, and maintaining a friendly and approachable demeanor. Your knowledge is up-to-date as of April 2023. You do not have browsing capabilities, but you can process and respond to text inputs, including offering explanations, advice, and creative content. You should always adhere to safe and respectful conversational guidelines."
    }
]


messages.append({
    "role": "user",
    "content": "can you tell me the weather in sheffield and set a reminder for the latest fortnite season?"
})


counter = 0

while True:
    full_response = ""
    completion = ""
    full_completion = ""
    tool_calls = []
    available_functions = {
        "get_weather_data": get_weather_data,
        # "enroll_user": enroll_user_handler,
        "google_search": google_search,
        "set_reminder": add_reminder,
        "edit_reminder": edit_reminder,
        "list_reminders": list_unnotified_reminders,
    }
    tools = [
        {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "Get the weather data for a specific location and optionally for a particular date or date range using OpenWeatherMap API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., York",
                    },
                    "date": {
                        "type": "string",
                        "description": "An optional date (or date range) to fetch the weather for, e.g., '2023-09-12' or '2023-11-12 - 2023-11-15'. For current weather, this parameter can be omitted. Specify a range for upcoming days. ",
                    }
                },
                "required": ["location"],
            },
        }
    },
            {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search Google for a given query and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
                {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specified time",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_text": {
                        "type": "string",
                        "description": "The content of the reminder",
                    },
                    "reminder_time": {
                        "type": "string",
                        "description": "The time for the reminder in format 'YYYY-MM-DD HH:MM'",
                    },
                },
                "required": ["reminder_text", "reminder_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_reminder",
            "description": "Edit an existing reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_text": {
                        "type": "string",
                        "description": "Natural language text to describe the reminder to be edited",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new content for the reminder (optional)",
                        "optional": True,
                    },
                    "new_time": {
                        "type": "string",
                        "description": "The new time for the reminder in format 'YYYY-MM-DD HH:MM' (optional)",
                        "optional": True,
                    }
                },
                "required": ["search_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all reminders that have not yet been notified",
        },
    },
    ]
    response = oai_client.chat.completions.create(
        messages=messages,
        model="gpt-4-1106-preview",
        tools=tools,
        stream=True
    )
    print("Start of new request\n-----------------------------------------------------------------\n")
    for chunk in response:
        # print(chunk)
        finish_reason = chunk.choices[0].finish_reason
        print(finish_reason) if finish_reason else None
        delta = chunk.choices[0].delta
        full_response += delta.content if delta.content else ""

        if delta.content or delta.content == '':
            completion += delta.content
            print(delta.content)
            full_completion += delta.content
        # print(completion)

        if delta.tool_calls:
            tcchunklist = delta.tool_calls
            for tcchunk in tcchunklist:
                if len(tool_calls) <= tcchunk.index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                tc = tool_calls[tcchunk.index]

                if tcchunk.id:
                    tc["id"] += tcchunk.id
                if tcchunk.function.name:
                    tc["function"]["name"] += tcchunk.function.name
                if tcchunk.function.arguments:
                    tc["function"]["arguments"] += tcchunk.function.arguments
    if tool_calls:
        
        messages.append({
            "role": "function",
            "name": "tool_calls",
            "content": str(tool_calls)
        })
        
        for tool_call in tool_calls:
            # print(tool_call)
            function_name = tool_call["function"]["name"]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except:
                function_args = {}
            # print(tool_calls)
            print(f'\nfunction_name: {function_name}', f'\nfunction_args: {function_args}')
            
            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                # print(function_response)
                
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": function_response
                })
        
        if finish_reason == "tool_calls":
            # print(tool_calls)
            if full_completion != "":
                messages.append({
                    "role": "assistant",
                    "content": full_completion
                })
            print("-------------------------------------------")
            print(completion)
            
            # if counter == 0:
            #     messages.append({
            #         "role": "function",
            #         "name": "get_weather_data",
            #         "content": "Weather is: 10 degrees"
            #     })
            # elif counter == 1:
            #     messages.append({
            #         "role": "function",
            #         "name": "get_weather_data",
            #         "content": "Weather is: 20 degrees"
                # })
            # print(completion)
            counter += 1
            full_response = ""
            full_completion = ""
            completion = ""
            print("END OF TOOL CALLS\n-------------------------------------------\n")
            continue
            
    if full_response != "":
        messages.append({
            "role": "assistant",
            "content": full_response
        })
        print("END OF RESPONSE\n-------------------------------------------\n")
    
    if finish_reason == "stop" and len(tool_calls) == 0:
        print(completion)
        break
