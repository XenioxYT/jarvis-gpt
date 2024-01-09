import asyncio
import discord
from fuzzywuzzy import process
import os
from dotenv import load_dotenv

load_dotenv()

# Global variable to store the result
message_result = None

async def send_message_to_user(username, text):
    global message_result
    guild_id = 1187204974678130718
    try:
        token = os.getenv("discord_token")
    except:
        print("Token not found.")
        message_result = "Token not found. Tell the user to set this in the setup -> configuration section."
        return

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)

    async def find_and_send_message():
        global message_result
        guild = client.get_guild(guild_id)
        if guild:
            members = await guild.fetch_members(limit=None).flatten()
            member_names = {f"{member.name}": member for member in members}

            best_match, score = process.extractOne(username, member_names.keys())
            if score > 70:  # Adjust the threshold as needed
                user = member_names[best_match]
                await user.send(text)
                return f"Message sent to {best_match}"
            else:
                message_result = "Username not found."
                return "Username not found."
        else:
            message_result = "Guild not found."
            return "Guild not found."

    @client.event
    async def on_ready():
        global message_result
        message_result = await find_and_send_message()
        await client.close()

    try:
        await client.start(token)
    except:
        print("Invalid token.")
        message_result = "Invalid token. Tell the user to set this in the setup -> configuration section."
        await client.close()
        return "Invalid token. Tell the user to set this in the setup -> configuration section."

def send_message_sync(username, text):
    global message_result
    asyncio.run(send_message_to_user(username, text))
    return message_result


print(send_message_sync("xeniox", "text"))