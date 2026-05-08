import requests
import json

API_KEY = "sk-hc-v1-ea5ee09993b74cb2ac056aea7d9bd4b15ecf062ece9c46dfbee3fab3a199b63f"
API_URL = "https://ai.hackclub.com/proxy/v1/models"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    print("Fetching models...")
    response = requests.get(API_URL, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        models = response.json()
        print(json.dumps(models, indent=2))
    else:
        print(response.text)
except Exception as e:
    print(e)
    ###done