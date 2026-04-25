import requests
import json
import base64
import os

API_KEY = "sk-hc-v1-aee58c686bb34783bc5b02763d72e82029bc32dc48dc4d56a56fcdae665337ea"
API_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"
MODEL = "qwen/qwen3-32b"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Use dummy.jpg relative to this script
image_path = os.path.join(os.path.dirname(__file__), "dummy.jpg")

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
else:
    print(f"Testing API with {MODEL}...")
    base64_image = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is this?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals():
            print(f"Response text: {response.text}")
