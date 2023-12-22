import os
import json
import datetime
import shutil
import unittest
from fuzzywuzzy import process

def save_note(user, title, text):
    # Ensure the user's directory exists
    user_dir = f'./notes/{user}'
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Prepare the note data
    note_data = {
        'title': title,
        'text': text,
        'creation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save the note
    notes_file = os.path.join(user_dir, 'notes.json')
    if os.path.exists(notes_file):
        with open(notes_file, 'r+') as file:
            notes = json.load(file)
            notes.append(note_data)
            file.seek(0)
            json.dump(notes, file)
    else:
        with open(notes_file, 'w') as file:
            json.dump([note_data], file)

    return f"Note for {user} saved successfully"

def retrieve_notes(user):
    # Retrieve the notes for the given user
    notes_file = f'./notes/{user}/notes.json'
    if os.path.exists(notes_file):
        with open(notes_file, 'r') as file:
            notes = json.load(file)
            formatted_notes = "\n\n".join([f"Title: {note['title']}\nText: {note['text']}\nCreation date: {note['creation_date']}" for note in notes])
            return formatted_notes
    else:
        return f"No notes found for {user}."

def search_notes(user, search_term):
    # Search the notes for the given user
    notes_file = f'./notes/{user}/notes.json'
    if os.path.exists(notes_file):
        with open(notes_file, 'r') as file:
            notes = json.load(file)
            titles = [note['title'] for note in notes]
            best_match = process.extractOne(search_term, titles, score_cutoff=75)
            if best_match:
                matching_notes = [note for note in notes if note['title'] == best_match[0]]
                return f"The matching notes for {user} are {matching_notes}"
            else:
                return []
    else:
        return []

def edit_or_delete_notes(user, title, new_title=None, new_text=None):
    """
    Edits or deletes a note for a given user. If there are multiple notes with the same title, it lists all matches.
    If there's a single match, it updates or deletes the note based on the provided new_title and new_text.
    If new_title and new_text are None, the note is deleted.
    """
    # Function to search notes (assuming it's defined elsewhere)
    def search_notes(user, search_term):
        # Dummy implementation (Replace with the actual search logic)
        return []

    # Edit or delete notes for a given user
    matching_notes = search_notes(user, title)
    notes_file = f'./notes/{user}/notes.json'

    if len(matching_notes) > 1:
        return "There are multiple matches:\n" + "\n".join([note['title'] for note in matching_notes])
    elif len(matching_notes) == 1:
        with open(notes_file, 'r+') as file:
            notes = json.load(file)
            for note in notes:
                if note['title'] == matching_notes[0]['title']:
                    if new_title is not None or new_text is not None:
                        note['title'] = new_title if new_title is not None else note['title']
                        note['text'] = new_text if new_text is not None else note['text']
                    else:
                        notes.remove(note)
                        file.seek(0)
                        file.truncate()
                        json.dump(notes, file)
                        return f"Note '{note['title']}' deleted successfully for {user}."

            file.seek(0)
            file.truncate()
            json.dump(notes, file)
            return f"Note '{note['title']}' updated successfully for {user}."
    else:
        return "No matching note found."