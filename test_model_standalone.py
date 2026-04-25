import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.alzheimers_model import AlzheimerNet

def test_model():
    print("Loading model...")
    model = AlzheimerNet(num_classes=4)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'saved_models', 'alzheimer_model.pth')
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image
    print("Creating dummy image...")
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(arr)
    
    # Transform
    print("Transforming image...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_tensor = transform(image).unsqueeze(0)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        
    _, predicted = torch.max(output.data, 1)
    print(f"Prediction index: {predicted.item()}")
    
    classes = ['nondemented', 'very mild', 'mild demented', 'moderate demented']
    print(f"Prediction label: {classes[predicted.item()]}")

if __name__ == "__main__":
    test_model()
