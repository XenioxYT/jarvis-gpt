import json
import sqlite3
import tiktoken
from utilities.strings import messages

db_conn = sqlite3.connect('./conversations.db')
db_conn.execute("""
CREATE TABLE IF NOT EXISTS conversations
    (conversation_id INTEGER PRIMARY KEY,
    conversation TEXT)
""")


def count_tokens_in_conversation(conversation):
    return sum(count_tokens(m["content"]) for m in conversation)


encoding = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text):
    return len(encoding.encode(text))


def ensure_system_message_on_top(conversation):
    """Ensure the system message is the first message in the conversation."""
    if conversation and conversation[0]["role"] != "system":
        # Find the system message
        system_msg_idx = next((idx for idx, m in enumerate(conversation) if m["role"] == "system"), None)

        # If a system message is found and it's not the first message, move it to the top
        if system_msg_idx is not None:
            system_message = conversation.pop(system_msg_idx)
            conversation.insert(0, system_message)


def trim_conversation_to_fit_limit(conversation, token_limit, conversation_id):
    """Trim the earliest non-system messages until the conversation is within the token limit."""
    while count_tokens_in_conversation(conversation) > token_limit:
        # If the second message is not a system message, remove it.
        # This ensures the first message (system message) always remains.
        if conversation[1]["role"] != "system":
            conversation.pop(1)
        else:
            # If for some reason there are multiple system messages or the order is not as expected
            ensure_system_message_on_top(conversation)
            conversation.pop(2)

        store_conversation(conversation_id, conversation)


def store_conversation(conversation_id, conversation, cursor=None, db_conn=None):
    token_limit = 5000

    # Ensure the system message is always the first message in the conversation
    ensure_system_message_on_top(conversation)

    # Trim the conversation to fit within the token (or message) limit
    trim_conversation_to_fit_limit(conversation, token_limit, conversation_id)

    # Now, store the conversation in the database

    if cursor is None:
        db_conn = sqlite3.connect('./conversations.db')
        cursor = db_conn.cursor()
    else:
        cursor = cursor
        db_conn = db_conn

    cursor = db_conn.cursor()
    values = (conversation_id, json.dumps(conversation))
    cursor.execute("REPLACE INTO conversations VALUES (?, ?)", values)
    db_conn.commit()

def check_conversation(conversation_id):
    db_conn = sqlite3.connect('./conversations.db')
    cursor = db_conn.cursor()
    cursor.execute("SELECT conversation FROM conversations WHERE conversation_id = ?", (conversation_id,))
    result = cursor.fetchone()
    if result is None:
        return None
    else:
        return result
    
def initiate_conversation_if_not_exists(conversation_id, cursor=None, db_conn=None):
    
    if cursor is None:
        db_conn = sqlite3.connect('./conversations.db')
        cursor = db_conn.cursor()
    else:
        cursor = cursor
        db_conn = db_conn
    # Check if the conversation exists in the database
    result = check_conversation(conversation_id)

    # If the conversation does not exist or the conversation content is empty, initiate a new conversation
    if result is None or not result[0]:
        store_conversation(conversation_id, messages)
    else:
        print(f'Conversation with id {conversation_id} already exists and has content.')
        
def get_conversation(conversation_id):
    # Check if the conversation exists in the database
    result = check_conversation(conversation_id)

    # If the conversation does not exist or the conversation content is empty, initiate a new conversation
    if result is None or not result[0]:
        initiate_conversation_if_not_exists(conversation_id)
        check = check_conversation(conversation_id)
        messages = json.loads(check[0])
    else:
        # Retrieve the conversation from the database and store it in the messages variable
        messages = json.loads(result[0])

    return messages