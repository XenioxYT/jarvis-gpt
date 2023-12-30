from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json["message"]
    # Sending user message to Rasa server
    rasa_response = requests.post("http://localhost:5005/model/parse", json={"text": user_message})
    rasa_data = rasa_response.json()

    # Extracting the top intent
    top_intent = rasa_data.get("intent", {}).get("name")
    intent_confidence = rasa_data.get("intent", {}).get("confidence")

    # Define a confidence threshold
    confidence_threshold = 0.60  # You can adjust this threshold

    # Check if the intent is matched with enough confidence
    if top_intent and intent_confidence > confidence_threshold:
        return jsonify({"intent": top_intent})
    else:
        return jsonify({"intent": "none"})

if __name__ == '__main__':
    app.run(port=5000)
