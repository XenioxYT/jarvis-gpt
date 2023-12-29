import os
import json
from difflib import get_close_matches

REMINDERS_DB_FILE = './reminders.json'


def list_unnotified_reminders(display_reminders=False):
    reminders = load_reminders()
    if not reminders:
        return "No reminders found. Please ask the user if they'd like to set a reminder. Don't call this function again when there are no reminders."
    return json.dumps(
        {"reminders": [rem for rem in reminders if not rem['notified']]},
        indent=2, default=str
    )


def load_reminders():
    if not os.path.exists(REMINDERS_DB_FILE):
        return []
    with open(REMINDERS_DB_FILE, 'r') as file:
        return json.load(file)


def save_reminders(reminders):
    with open(REMINDERS_DB_FILE, 'w') as file:
        json.dump(reminders, file, indent=2)
        file.flush()
        os.fsync(file.fileno())  # Force write to disk


def add_reminder(reminder_text, reminder_time):
    reminders = load_reminders()
    reminder_id = 1 if not reminders else max(r['id'] for r in reminders) + 1
    reminders.append({
        'id': reminder_id,
        'text': reminder_text,
        'time': reminder_time,
        'notified': False
    })
    save_reminders(reminders)
    return f"Reminder set for {reminder_time} with text: {reminder_text}"


def get_closest_reminder_matches(search_text, threshold=0.5):
    """
    Optimized to reduce unnecessary list comprehensions and improve efficiency in searching.
    
    :param search_text: String to match against reminder descriptions.
    :param threshold: Float, similarity ratio must be greater than this threshold to be considered a match.
    :return: A list of potential reminders that match.
    """
    reminders = load_reminders()
    # Filtering and collecting descriptions in one pass
    descriptions = []
    reminder_map = {}  # Maps description to reminder for quick access
    for reminder in reminders:
        if not reminder['notified']:
            descriptions.append(reminder['text'])
            reminder_map[reminder['text']] = reminder

    matches = get_close_matches(search_text, descriptions, n=3, cutoff=threshold)

    if search_text in reminder_map:
        # Direct access to the reminder if an exact match is found
        return ([reminder_map[search_text]], True)

    # Collecting matching reminders using the map for faster access
    matching_reminders = [reminder_map[desc] for desc in matches if desc in reminder_map]

    return (matching_reminders, False)


def edit_reminder(search_text, new_text="None", new_time="None"):
    reminders = load_reminders()
    matched_reminders, exact_match = get_closest_reminder_matches(search_text)
    print(f"Matching reminders: {matched_reminders}, Exact match: {exact_match}")

    if not matched_reminders:
        return "No matching reminder found."
    elif exact_match or len(matched_reminders) == 1:
        # If an exact match is found, or there is only one possible match, update the reminder
        reminder_to_edit = matched_reminders[0]
        print(f"Found reminder to edit: {reminder_to_edit}")

        # Find the reminder in the list and update it
        for index, rem in enumerate(reminders):
            if rem['id'] == reminder_to_edit['id']:
                print(f"Found reminder at index {index} to update.")
                if new_time != "None":
                    rem['time'] = new_time
                if new_text != "None":
                    rem['text'] = new_text
                rem['notified'] = False  # Reset notification status

        save_reminders(reminders)
        updated_reminders = load_reminders()
        print(f"Updated reminders from file: {updated_reminders}")

        # Find the reminder in the updated list for final confirmation
        for rem in updated_reminders:
            if rem['id'] == reminder_to_edit['id']:
                print(f"Confirmed updated reminder from file: {rem}")
                break

        return "Your reminder has been successfully updated to" + new_time + "with text: " + new_text
    else:
        # If multiple matches are found, explain to the user how to specify their choice
        message = "No exact match found for editing a reminder. " \
                  "Here are the top hits, please specify by saying, " \
                  "for example, 'The first one' or 'The second one':\n"
        message += "\n".join(f"{index + 1}: '{reminder['text']}' for {reminder['time']}"
                             for index, reminder in enumerate(matched_reminders))
        return message
