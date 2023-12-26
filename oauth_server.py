# oauth_server.py
from flask import Flask, request, redirect, jsonify
from google_auth_oauthlib.flow import Flow
import os

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']
REDIRECT_URI = 'https://calendar.xeniox.tv/oauth2callback'

@app.route('/oauth2callback')
def oauth2callback():
    code = request.args.get('code', None)
    state = request.args.get('state', None)
    user = state.split('_')[1] if state else None

    if not code or not user:
        return "No authorization code or user found in request.", 400

    # Complete the exchange of code for tokens
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)

    # Save the credentials
    credentials = flow.credentials
    # Save the credentials in user-specific folder
    user_token_dir = f'tokens/{user}'
    os.makedirs(user_token_dir, exist_ok=True)
    with open(f'{user_token_dir}/token.json', 'w') as token_file:
        token_file.write(credentials.to_json())

    return "Authentication successful. You can close this window.", 200

@app.route('/test')
def test_route():
    return jsonify({'message': 'Test route is working!'}), 200

if __name__ == '__main__':
    app.run(port=9444, host='0.0.0.0')
