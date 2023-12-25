# oauth_server.py
from flask import Flask, request, redirect, jsonify
from google_auth_oauthlib.flow import Flow
import os

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']
REDIRECT_URI = 'https://calendar.xeniox.tv'

@app.route('/oauth2callback')
def oauth2callback():
    code = request.args.get('code', None)

    if not code:
        return "No authorization code found in request.", 400

    # Complete the exchange of code for tokens
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)

    # Save the credentials
    credentials = flow.credentials
    with open('token.json', 'w') as token_file:
        token_file.write(credentials.to_json())

    return "Authentication successful. You can close this window.", 200

@app.route('/test')
def test_route():
    return jsonify({'message': 'Test route is working!'}), 200

if __name__ == '__main__':
    app.run(ssl_context='adhoc', port=9444)