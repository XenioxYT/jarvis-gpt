import os
import discord
import asyncio
from fuzzywuzzy import process

async def send_message_to_user(username, text):
    guild_id = 1187204974678130718
    token = os.environ.get('DISCORD_TOKEN')

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)

    async def find_and_send_message():
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
                return "Username not found."
        else:
            return "Guild not found."

    @client.event
    async def on_ready():
        result = await find_and_send_message()
        print(result)
        await client.close()

    await client.start(token)

def send_message_sync(username, text):
    asyncio.run(send_message_to_user(username, text))
    return "Message sent."

# Usage
# send_message_sync("xeniox", "Hello, this is your message!")
