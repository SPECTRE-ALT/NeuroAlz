import requests
from PIL import Image
import io
import numpy as np

def create_dummy_image():
    # Create a random RGB image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return img

def test_prediction():
    url = 'http://localhost:5001/predict'
    img = create_dummy_image()
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    files = {'file': ('dummy.jpg', buf, 'image/jpeg')}
    
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("Success! Prediction:", response.json())
        else:
            print("Failed:", response.status_code, response.text)
    except Exception as e:
        print("Error sending request:", e)

if __name__ == "__main__":
    test_prediction()
