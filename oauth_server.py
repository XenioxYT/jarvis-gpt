from flask import Flask, request, jsonify
from google_auth_oauthlib.flow import Flow
import os
import json

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']
REDIRECT_URI = 'https://calendar.xeniox.tv/oauth2callback'

@app.route('/oauth2callback')
def oauth2callback():
    code = request.args.get('code', None)
    state = request.args.get('state', None)
    session_id = state.split('_')[0] if state else None
    user = state.split('_')[2] if state else None

    if not code or not user or not session_id:
        return "Invalid request. Missing parameters.", 400

    # Complete the exchange of code for tokens
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)

    # Save the credentials in a temporary directory
    temp_token_dir = f'temp_tokens/{session_id}'
    os.makedirs(temp_token_dir, exist_ok=True)
    with open(f'{temp_token_dir}/token.json', 'w') as token_file:
        token_file.write(flow.credentials.to_json())

    return "Authentication successful. You can close this window.", 200

@app.route('/download_token/<session_id>')
def download_token(session_id):
    token_path = f'temp_tokens/{session_id}/token.json'
    if os.path.exists(token_path):
        with open(token_path, 'r') as token_file:
            token_data = token_file.read()
        os.remove(token_path)  # Optional: Remove token file after sending
        return jsonify({'token': token_data}), 200
    else:
        return 'Token not found', 404

if __name__ == '__main__':
    app.run(port=9444, host='0.0.0.0')