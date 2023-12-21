import os
import discord
import asyncio
from fuzzywuzzy import process

async def send_discord_dm(token, user_id, message):
    """
    Sends a DM to a specified user on Discord using Py-cord.

    :param token: Discord Bot Token
    :param user_id: The user ID of the recipient
    :param message: The message to be sent
    """

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    async def send_message():
        await client.wait_until_ready()
        user = await client.fetch_user(user_id)

        if user:
            await user.send(message)
            print(f"Message sent to {user.name}")
        else:
            print("User not found.")

        await client.close()

    client.loop.create_task(send_message())

    try:
        await client.start(token)
    except Exception as e:
        print(f"An error occurred: {e}")
        await client.close()

async def list_users_in_guild(token, guild_id):
    """
    Lists all users in a specified Discord guild.

    :param token: Discord Bot Token
    :param guild_id: The ID of the Discord guild (server)
    """

    intents = discord.Intents.default()
    intents.members = True  # Necessary to access guild members
    client = discord.Client(intents=intents)

    async def list_members():
        await client.wait_until_ready()
        guild = client.get_guild(guild_id)

        if guild:
            print(f"Members in {guild.name}:")
            async for member in guild.fetch_members(limit=None):
                print(f"{member.name}#{member.discriminator} (ID: {member.id})")
        else:
            print("Guild not found.")

        await client.close()

    client.loop.create_task(list_members())

    try:
        await client.start(token)
    except Exception as e:
        print(f"An error occurred: {e}")
        await client.close()

async def search_user_in_guild(token, guild_id, search_query):
    """
    Searches for a user in a Discord guild using fuzzy matching.
    Returns only the exact match if found, otherwise returns top fuzzy matches.

    :param token: Discord Bot Token
    :param guild_id: The ID of the Discord guild (server)
    :param search_query: The username or part of the username to search for
    """

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)

    async def search_member():
        await client.wait_until_ready()
        guild = client.get_guild(guild_id)

        if guild:
            members = await guild.fetch_members(limit=None).flatten()
            member_names = [f"{member.name}" for member in members]

            # Check for exact match first
            exact_match = next((name for name in member_names if name.lower() == search_query.lower()), None)
            if exact_match:
                print(f"Exact match found: {exact_match}")
            else:
                # Fuzzy search
                matches = process.extract(search_query, member_names, limit=5)  # Adjust limit as needed
                print(f"Top matches for '{search_query}' in {guild.name}:")
                for match in matches:
                    print(match)
        else:
            print("Guild not found.")

        await client.close()

    client.loop.create_task(search_member())

    try:
        await client.start(token)
    except Exception as e:
        print(f"An error occurred: {e}")
        await client.close()

# Usage
discord_token = os.environ.get('DISCORD_TOKEN')
recipient_id = 279718966543384578  # Replace with the actual user ID
guild_id = 1087746041069187154
message_text = "Hello, this is a test message from Py-cord!"
search_query = "llvinll"  

# Since this is an asynchronous function, it should be called using asyncio
# asyncio.run(send_discord_dm(discord_token, recipient_id, message_text))
# asyncio.run(list_users_in_guild(discord_token, guild_id))
asyncio.run(search_user_in_guild(discord_token, guild_id, search_query))

