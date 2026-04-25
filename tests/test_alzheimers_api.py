import requests
import os
import random
import sys

# Add the parent directory to sys.path to allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_alzheimers_api():
    """
    Test the Alzheimer's Disease Detection API by sending a random image
    and checking the response.
    """
    # API endpoint
    url = 'http://localhost:5000/predict'

    # Directory containing test images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_image_dir = os.path.join(project_root, 'data', 'test')

    print(f"Searching for images in: {os.path.abspath(test_image_dir)}")

    # Get all subdirectories (Alzheimer's stages)
    stage_dirs = [d for d in os.listdir(test_image_dir) if os.path.isdir(
        os.path.join(test_image_dir, d))]

    if not stage_dirs:
        print("No stage directories found in the test directory.")
        return

    # Randomly select a stage directory
    random_stage = random.choice(stage_dirs)
    stage_dir = os.path.join(test_image_dir, random_stage)

    # Get a list of all image files in the selected stage directory
    image_files = [f for f in os.listdir(
        stage_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in the {random_stage} directory.")
        return

    # Randomly select an image
    random_image = random.choice(image_files)
    image_path = os.path.join(stage_dir, random_image)

    print(f"Testing with image: {image_path}")
    print(f"True label (directory name): {random_stage}")

    # Open the image file
    with open(image_path, 'rb') as img:
        # Prepare the file for the request
        files = {'file': (random_image, img, 'image/jpeg')}

        # Send POST request to the API
        response = requests.post(url, files=files)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['result']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_alzheimers_api()
