import requests
import json

API_KEY = "sk-hc-v1-ea5ee09993b74cb2ac056aea7d9bd4b15ecf062ece9c46dfbee3fab3a199b63f"
API_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

label = "Mild Demented"
confidence_details = "Mild Demented: 95.0%, Very Mild: 5.0%"

prompt = f"""
You are an expert neuroradiologist. You are reviewing an MRI analysis report.

Final Diagnosis: {label}
Confidence: {confidence_details}

Write a "Findings" section for the medical report.
Explain WHY this diagnosis is appropriate. Describe the visual evidence of cortical atrophy, ventricular enlargement, or hippocampal shrinkage that justifies the '{label}' classification.
Be authoritative and descriptive.
"""

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

print(f"Sending prompt: {prompt}")
try:
    response = requests.post(API_URL, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print("Usage:", json_response.get('usage'))
            choice = json_response['choices'][0]
            print(f"Finish Reason: {choice.get('finish_reason')}")
            content = choice['message']['content']
            print(f"Content Length: {len(content) if content else 'None'}")
            print(f"Content Repr: {repr(content)}")
        except Exception as e:
            print("JSON Parse Error:", e)
            print("Raw Text:", response.text)
    else:
        print("Error Response:", response.text)

except Exception as e:
    print(f"Request failed: {e}")
