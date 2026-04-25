import requests
import json

API_KEY = "sk-hc-v1-1931805680294e878ec50f3bafa480afe2e29fa3f8c246c595f5be18f5ea216c"
API_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "qwen/qwen3-32b",
    "messages": [
        {"role": "user", "content": "Hi, are you working?"}
    ]
}

print("Sending request...")
try:
    response = requests.post(API_URL, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response Headers:", response.headers)
    print("Raw Response:", response.text)
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print("Parsed Content:", json_response['choices'][0]['message']['content'])
        except Exception as e:
            print("JSON Parse Error:", e)
except Exception as e:
    print(f"Request failed: {e}")
