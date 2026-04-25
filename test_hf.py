from huggingface_hub import InferenceClient
from PIL import Image
import io

# Config
HF_API_KEY = "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"
import requests
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
# headers = {"Authorization": f"Bearer {HF_API_KEY}"}
headers = {} # Try anonymous

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response

def test_api():
    print(f"Testing Requests ANONYMOUSLY...")
    
    # Create dummy image if needed
    try:
        with open("dummy.jpg", "rb") as f:
            pass
    except:
        img = Image.new('RGB', (224, 224), color='red')
        img.save("dummy.jpg")

    response = query("dummy.jpg")
    
    try:
        print("Response JSON:", response.json())
    except:
        print("Response Text (first 200 chars):", response.text[:200])
        
    print("\nStatus Code:", response.status_code)

if __name__ == "__main__":
    test_api()
