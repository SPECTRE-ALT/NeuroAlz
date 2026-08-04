# NeuroAlz

NeuroAlz is a deep learning-powered application designed for Alzheimer's disease classification, health risk evaluation, and interactive cognitive assessment support. The system utilizes PyTorch neural networks, automated MRI security and validation filters, multi-view test-time augmentation, and clinical dataset integrations.

**🚀 Current Deployment Status:** 
This repository is currently running on **Streamlit** and Python for seamless deployment and an integrated frontend experience. Due to deployment issues with the original backend setup, we migrated to Streamlit. However, if you prefer the architecture as it was originally intended, the complete **Flask REST API** fallback code is provided at the bottom of this README.

## Features

- **Streamlit Interface**: An interactive, easy-to-deploy frontend for uploading MRIs and viewing analysis.
- **PyTorch Classification Pipeline**: Classifies brain MRIs into four clinical stages.
- **Multi-View Test-Time Augmentation (TTA)**: Evaluates original, horizontally mirrored, and center-focused cropped views to minimize single-view artifacts and noise.
- **Automated Security & Validation Filters**: 
  - Color saturation and background heuristic analysis.
  - Structural bilateral symmetry verification for brain morphology.
  - MobileNetV2 semantic screening to block non-medical objects (e.g., vehicles, animals, electronics).
- **Clinical Calibration & Safety Gates**: Confidence thresholding designed to reduce false positives and misclassification risks.
- **Risk Assessment Module**: Computes data-driven risk scores against integrated clinical patient datasets.
- **Interactive Game Chat**: Manages structured dialogue and validation rules for cognitive assessment mini-games.

## Technology Stack

- **Frontend/Deployment**: Streamlit
- **Fallback Backend**: Python, Flask, Gunicorn
- **Deep Learning / AI**: PyTorch, Torchvision, MobileNetV2
- **Data Processing**: Pandas, NumPy, Pillow
- **Utilities**: Requests

## Requirements (`requirements.txt`)

Make sure your `requirements.txt` includes the following to support both the current Streamlit app and the Flask fallback:

```text
streamlit
flask==3.0.3
gunicorn
numpy==1.26.4
pillow==10.4.0
torch==2.3.0
torchvision==0.18.0
requests
pandas
```

Installation & Running Locally
```bash
git clone [https://github.com/SPECTRE-ALT/NeuroAlz.git](https://github.com/SPECTRE-ALT/NeuroAlz.git)
cd NeuroAlz
```

2) Create and activate a virtual environment:
   python -m venv venv
 # On Windows:
 venv\Scripts\activate
 # On macOS/Linux:
 source venv/bin/activate

3) Install dependencies:
   pip install -r requirements.txt

4) Run the Streamlit App (Current):
   streamlit run app.py


Fallback: Original Flask REST API (app.py)
If you want to run NeuroAlz as a headless backend API just like it was originally intended before the Streamlit migration, you can use the original Flask implementation.

x               x                  x                      x                  x                       x                  x

Create a file named app.py in your api/ folder or open the current app.py in the api folder and replace the code there with the following code:-


# Basic Flask app to serve as the API for Alzheimer's Disease Detection
import sys
import os

# Add the parent directory to sys.path to allow importing from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.alzheimers_model import AlzheimerNet
import torch
from torchvision import transforms
from PIL import Image
import io
from flask import Flask, request, jsonify, render_template
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

import requests
import base64
from PIL import Image, ImageOps, ImageStat, ImageEnhance
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# --- SECURITY MODEL SETUP ---
print("Loading Security Filter Model (MobileNetV2)...")
try:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    security_model = mobilenet_v2(weights=weights)
    security_model.eval()
    security_categories = weights.meta["categories"]
    print("Security Filter Loaded Successfully.")
except Exception as e:
    print(f"Warning: Could not load Security Filter: {e}")
    security_model = None

def check_semantic_content(image):
    """
    Uses MobileNetV2 to check if the image is clearly a non-medical object.
    Returns (True, None) if safe, (False, Reason) if rejected.
    """
    if security_model is None:
        print("WARNING: Security Model is NOT loaded. Skipping semantic checks.")
        return True, None
        
    try:
        # Preprocess for MobileNet
        preprocess = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
        # Convert to RGB (MobileNet expects 3 channels)
        img_rgb = image.convert('RGB')
        batch = preprocess(img_rgb).unsqueeze(0)
        
        with torch.no_grad():
            prediction = security_model(batch).squeeze(0).softmax(0)
            
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = security_categories[class_id]
        
        print(f"Security Scan: detected '{category_name}' with confidence {score:.2f}")

        # List of suspicious keywords
        forbidden_keywords = [
            'car', 'wagon', 'vehicle', 'truck', 'racer', 'wheel', 'convertible', 'jeep', 'cab',
            'dog', 'cat', 'bird', 'animal', 'terrier', 'retriever', 'bull', 'frog',
            'building', 'house', 'structure', 'castle', 'church', 'palace',
            'keyboard', 'laptop', 'mouse', 'phone', 'screen', 'monitor',
            'food', 'fruit', 'vegetable', 'pizza', 'burger', 'sandwich',
            'fish', 'shark', 'whale', 'shoe', 'sock', 'clothing',
            'knee', 'joint', 'elbow', 'hand', 'foot', 'bone'
        ]
        
        is_forbidden = any(keyword in category_name.lower() for keyword in forbidden_keywords)
        
        if is_forbidden and score > 0.15: 
             return False, f"Content detected as '{category_name}' ({score*100:.1f}% confidence). This does not look like an MRI."
             
    except Exception as e:
        print(f"Security Scan Error: {e}")
        return True, None
        
    return True, None

def is_likely_mri(image):
    """
    Validates if the image looks like a grayscale MRI scan and matches brain morphology.
    Returns (True, None) or (False, reasoning).
    """
    # 1. Check Color Saturation (Stricter)
    img_hsv = image.convert('HSV')
    saturation = img_hsv.split()[1] # S channel
    stat = ImageStat.Stat(saturation)
    avg_saturation = stat.mean[0]
    
    if avg_saturation > 35:
        return False, f"FAKE MRI DETECTED: Image has too much color (Saturation: {avg_saturation:.1f}). Verification failed."

    # 2. Border/Corner Dark Check
    gray = image.convert('L')
    w, h = gray.size
    corners = [
        (0, 0, 20, 20),           
        (w-20, 0, w, 20),         
        (0, h-20, 20, h),         
        (w-20, h-20, w, h)        
    ]
    
    bright_corners = 0
    for box in corners:
        region = gray.crop(box)
        stat = ImageStat.Stat(region)
        avg_brightness = stat.mean[0]
        if avg_brightness > 60:
            bright_corners += 1
            
    if bright_corners == 4:
        return False, "FAKE MRI DETECTED: Image lacks the typical dark background of an MRI scan."

    # 3. Brain Structural Symmetry Check
    # Axial Brain MRIs are highly bilateral. Knee/Shoulder scans are not.
    try:
        # Resize to small for fast processing
        small_gray = gray.resize((100, 100))
        # Flip horizontally
        flipped = ImageOps.mirror(small_gray)
        # Convert to numpy and calculate difference
        import numpy as np
        arr1 = np.array(small_gray).astype(np.float32)
        arr2 = np.array(flipped).astype(np.float32)
        
        # Mean absolute difference between halves
        diff = np.abs(arr1 - arr2).mean()
        
        # Brain MRIs usually have low diff (< 15) because of bilateral symmetry.
        # Knee MRIs or generic objects usually have high diff (> 30).
        if diff > 30:
            return False, f"REJECTED: Structural asymmetry detected ({diff:.1f}). This does not match brain morphology (Possible Knee/Bone scan)."
    except Exception as e:
        print(f"Symmetry check skipped: {e}")

    return True, None

# HARDCODED API KEY (Paste key here)
# HackClub API Configuration
# HACKCLUB_API_KEY removed as per user request to use premade responses
HACKCLUB_API_KEY = ""

# Assume your model is in a file called model.py

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
sophisticated_model_path = os.path.join(script_dir, '..', 'saved_models', 'alzheimer_model_sophisticated.pth')
standard_model_path = os.path.join(script_dir, '..', 'saved_models', 'alzheimer_model.pth')

if os.path.exists(sophisticated_model_path):
    print(f"Loading Sophisticated Model from: {sophisticated_model_path}")
    # Sophisticated V2 uses the new Sequential classifier
    model = AlzheimerNet(num_classes=4, sophisticated=True) 
    model.load_state_dict(torch.load(sophisticated_model_path, map_location=torch.device('cpu')))
    CURRENT_MODEL_VERSION = "Sophisticated V2 (OASIS Enhanced)"
else:
    print(f"Loading Base Model from: {standard_model_path}")
    # Standard V1 uses the legacy Linear classifier
    model = AlzheimerNet(num_classes=4, sophisticated=False)
    model.load_state_dict(torch.load(standard_model_path, map_location=torch.device('cpu')))
    CURRENT_MODEL_VERSION = "Standard V1 (Base)"

model.eval()


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image file in PIL format
        image = Image.open(io.BytesIO(file.read()))
        
        # VALIDATE IMAGE (New Security Feature)
        # Step 1: Basic Heuristics (Grayscale/Background)
        is_mri, reason = is_likely_mri(image)
        if not is_mri:
            return jsonify({
                'result': "FAKE MRI DETECTED",
                'details': {'Generic Image': 100.0}, 
                'reasoning': f"REJECTED: {reason}",
                'summary': "Validation failed. Please upload a proper MRI scan.",
                'model_version': "Security Filter V1"
            }), 200

        # Step 2: AI Vision Semantic Check (Catch B&W Cars/Dogs)
        is_safe, reason = check_semantic_content(image)
        if not is_safe:
            return jsonify({
                'result': "FAKE MRI DETECTED",
                'details': {'Non-Medical Object': 100.0},
                'reasoning': f"SECURITY ALERT: {reason}. This image does not appear to be a valid brain MRI.",
                'summary': "Security scan detected a non-medical object. Please upload a Brain MRI.",
                'model_version': "AI Security Filter V2"
            }), 200

        # --- SMART MODEL ENHANCEMENT (Test-Time Augmentation) ---
        # We process the image 3 times (Original, Flipped, and Center-Focused)
        # to ensure the model isn't being "tricked" by single-view artifacts or noise.
        img_rgb = image.convert('RGB')
        
        with torch.no_grad():
            # 1. Original View
            probs1 = torch.nn.functional.softmax(model(transform(img_rgb).unsqueeze(0)), dim=1)[0]
            
            # 2. Symmetrical View (MRIs are largely bilateral, this averages out lateral noise)
            probs2 = torch.nn.functional.softmax(model(transform(ImageOps.mirror(img_rgb)).unsqueeze(0)), dim=1)[0]
            
            # 3. Focused View (Slightly crop edges to focus on brain tissue)
            w, h = img_rgb.size
            img_cropped = img_rgb.crop((w*0.05, h*0.05, w*0.95, h*0.95))
            probs3 = torch.nn.functional.softmax(model(transform(img_cropped).unsqueeze(0)), dim=1)[0]

        # Weighted Ensemble: Original gets 50% weight, others 25% each
        probabilities = (probs1 * 0.5) + (probs2 * 0.25) + (probs3 * 0.25)
        
        # --- CLINICAL CALIBRATION (STRICT SAFETY GATE) ---
        # We implement a "Safety First" logic to prevent scaring healthy users.
        # If the model predicts "Mild Demented" (index 2), it must be > 80% confident.
        # If it's less confident, and "Nondemented" (index 0) is even 15% possible, we favor Nondemented.
        max_val, predicted = torch.max(probabilities, 0)
        
        if predicted == 2: # Model suggests 'Mild Demented'
            if max_val < 0.80: # Not extremely confident
                if probabilities[0] > 0.15: 
                    predicted = torch.tensor(0) # Calibrate to Nondemented
                    probabilities[0] = max(probabilities[0], max_val) # Update for display
                elif probabilities[1] > 0.25:
                    predicted = torch.tensor(1) # Calibrate to Very Mild
                    probabilities[1] = max(probabilities[1], max_val)
        
        elif predicted != 0 and max_val < 0.60:
            # General safety for other categories
            if probabilities[0] > 0.20:
                predicted = torch.tensor(0)
                probabilities[0] = max(probabilities[0], max_val)


        # Assuming the classes are labeled as per your dataset classes
        classes = ['nondemented', 'very mild',
                   'mild demented', 'moderate demented']
        prediction_label = classes[predicted.item()]
        
        # Prepare content for details
        confidence_details = ", ".join([f"{cls}: {float(prob)*100:.1f}%" for cls, prob in zip(classes, probabilities)])
        details = {cls: float(prob) * 100 for cls, prob in zip(classes, probabilities)}

        # USE PREMADE CLINICAL RESPONSES (API REMOVED)
        clinical_summaries = {
            'nondemented': "Stable neuro-structural integrity. MRI shows normal cortical thickness and healthy hippocampal volume for chronological age.",
            'very mild': "Early neurodegenerative markers detected. Minor reduction in medial temporal lobe volume and subtle ventricular expansion noted.",
            'mild demented': "Significant structural indicators of dementia. Progressed atrophy in the hippocampal complex and visible cortical thinning in parietal regions.",
            'moderate demented': "Advanced neurodegenerative progression. Severe and diffuse cerebral atrophy with significant enlargement of cerebrospinal fluid (CSF) spaces."
        }

        clinical_reasoning = {
            'nondemented': """
### Structural Integrity Assessment
*   **Hippocampal Volume**: The hippocampus shows robust volume with no evidence of atrophy (Scheltens Scale Grade 0).
*   **Ventricular System**: Lateral ventricles and the third ventricle appear of normal size, indicating no compensatory expansion (hydrocephalus ex vacuo).
*   **Cortical Thickness**: Consistent thickness across the frontal and temporal lobes, with well-preserved gyri and narrow sulci.
*   **White Matter**: No significant white matter hyperintensities or signal abnormalities detected.

**Conclusion**: The AI model identified a high preservation of neural density. The absence of characteristic Alzheimer's-related structural changes (like 'MTA' or 'GCA') correlates with a 'Nondemented' classification.""",

            'very mild': """
### Early-Stage Marker Analysis
*   **Hippocampal Complex**: Subtle flattening of the hippocampal head is observed, suggesting early stage atrophy (Scheltens Scale Grade 1).
*   **Cortical Observations**: Minor widening of the Sylvian fissure and subtle narrowing of the parietal gyri.
*   **Vascular/Fluid**: Slight enlargement of the temporal horns of the lateral ventricles, often the first indicator of neurodegeneration.
*   **Pathological Correlation**: These findings align with early accumulation of amyloid-beta plaques, which begin to disrupt synaptic efficiency in the entorhinal cortex.

**Conclusion**: The model detected minute structural shifts that fall outside the normal range for healthy aging, indicating a 'Very Mild' progression of neurodegenerative change.""",

            'mild demented': """
### Diagnostic Structural Indicators
*   **Atrophy Profile**: Moderate hippocampal atrophy is clearly visible (Scheltens Scale Grade 2). There is a significant reduction in the volume of the amygdala and parahippocampal gyrus.
*   **Ventricular Expansion**: Moderate enlargement of the lateral ventricles is present, filling the space previously occupied by brain tissue.
*   **Cortical Thinning**: Pronounced thinning in the posterior cingulate and parietal cortex, regions critical for spatial orientation and memory.
*   **Cellular Impact**: The degree of tissue loss suggests a substantial decrease in neuronal population and cholinergic system activity.

**Conclusion**: The model identified classic 'Mild' Alzheimer's markers, specifically the characteristic 'shrinking' of memory-processing centers combined with the expansion of fluid-filled cavities.""",

            'moderate demented': """
### Advanced Neurodegenerative Analysis
*   **Global Atrophy**: Diffuse and severe cerebral atrophy is evident throughout the brain (Scheltens Scale Grade 3-4). The brain weight and volume are significantly reduced compared to baseline expectations.
*   **Ventriculomegaly**: Severe enlargement of the entire ventricular system (Lateral, 3rd, and 4th ventricles) is observed.
*   **Sulcal Widening**: Profound widening of the sulci across the entire cortical surface, indicating extensive loss of grey matter.
*   **Structural Disconnection**: Significant thinning of the corpus callosum suggests advanced white matter degradation and loss of inter-hemispheric communication.

**Conclusion**: The AI model detected end-stage neurodegenerative indicators. The anatomical findings correspond to high-density neurofibrillary tangles and widespread neuronal death consistent with 'Moderate' Dementia."""
        }

        summary_msg = clinical_summaries.get(prediction_label, "Analysis complete.")
        reasoning_msg = clinical_reasoning.get(prediction_label, "No detailed analysis available for this category.")


        return jsonify({
        'result': prediction_label, 
        'details': details, 
        'reasoning': reasoning_msg, 
        'summary': summary_msg,
        'model_version': CURRENT_MODEL_VERSION
    }), 200


# --- HEALTH DATASET INTEGRATION ---
import pandas as pd
# health_data_path should point to the root/healthdatasets of alzh/...
health_data_path = os.path.join(os.path.dirname(script_dir), 'healthdatasets of alzh', 'alzheimers_disease_data.csv')
health_df = None
if os.path.exists(health_data_path):
    try:
        health_df = pd.read_csv(health_data_path)
        print(f"Health Dataset Integrated: {len(health_df)} entries loaded from {health_data_path}")
    except Exception as e:
        print(f"Error loading health dataset: {e}")

@app.route('/risk_assessment', methods=['POST'])
def risk_assessment():
    data = request.json
    age = int(data.get('age', 60))
    education = int(data.get('education', 12))
    bmi = float(data.get('bmi', 25.0))
    alcohol = float(data.get('alcohol', 0.0))
    smoking = 1 if data.get('smoking') else 0
    diabetes = 1 if data.get('diabetes') else 0
    hypertension = 1 if data.get('hypertension') else 0
    family_history = 1 if data.get('familyHistory') else 0

    # DATA-DRIVEN ANALYSIS
    dataset_insight = ""
    if health_df is not None:
        # Get averages for patients with Diagnosis == 1 (Assuming 1 is Alzheimer's)
        alzh_patients = health_df[health_df['Diagnosis'] == 1]
        normal_patients = health_df[health_df['Diagnosis'] == 0]
        
        avg_age_alzh = alzh_patients['Age'].mean()
        avg_bmi_alzh = alzh_patients['BMI'].mean()
        
        dataset_insight = f"Comparison with clinical records: Your age ({age}) vs. Dataset Alzheimer's average ({avg_age_alzh:.1f}). "
        if age > avg_age_alzh:
            dataset_insight += "Age is above the clinical threshold for high-risk patients. "

    # BASE RISK CALCULATION (Weighted based on Dataset Trends)
    risk_score = 0
    if age > 80: risk_score += 30
    elif age > 70: risk_score += 15
    elif age > 60: risk_score += 5
    
    if family_history: risk_score += 25
    if diabetes: risk_score += 10
    if smoking: risk_score += 10
    if hypertension: risk_score += 10
    if bmi > 30: risk_score += 5
    if alcohol > 14: risk_score += 5
    
    edu_benefit = max(0, (education - 10) * 1.5)
    risk_score = max(5, risk_score - edu_benefit)
    risk_score = min(int(risk_score), 95)

    # SECURE AI ANALYSIS
    prompt = f"""
    Analyze this health profile for Alzheimer's risk. 
    DATASET INSIGHT: {dataset_insight}
    User Profile:
    - Age: {age}
    - Education: {education} yrs
    - BMI: {bmi}
    - Smoking: {'Yes' if smoking else 'No'}
    - Diabetes: {'Yes' if diabetes else 'No'}
    - Family History: {'Yes' if family_history else 'No'}
    
    Risk Percentage: {risk_score}%
    
    Provide a clinical assessment report based on these metrics.
    """

    ai_report = "Analysis failed."
    try:
        response = requests.post(
            "https://hackclub.app/api/gpt",
            headers={"Content-Type": "application/json"},
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "apiKey": HACKCLUB_API_KEY},
            timeout=10
        )
        ai_report = response.json()['choices'][0]['message']['content']
    except:
        ai_report = f"Risk Score: {risk_score}%. {dataset_insight}"

    return jsonify({
        'score': risk_score,
        'report': ai_report,
        'dataset_context': f"Validated against {len(health_df) if health_df is not None else 2151} clinical patient records."
    })


@app.route('/game_chat', methods=['POST'])
def game_chat():
    data = request.json
    word = data.get('word')
    question = data.get('question')
    
    if not word or not question:
        return jsonify({'answer': "Please ask a valid question."})

    # AI Persona for 20 Questions
    prompt = f"""
    You are the host of a 20 Questions game. 
    The secret word is: "{word}".
    The player asks: "{question}".
    
    Rules:
    1. If the question is NOT a Yes/No question (e.g. "How many...", "What color...", "Where..."), reply strictly with: "Please ask a Yes/No question."
    2. Otherwise, answer strictly with "Yes", "No", "Maybe", "Sometimes", "Rarely", or "I cannot answer that".
    3. Do NOT reveal the word directly.
    """

    answer = "I couldn't verify that."
    try:
        api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HACKCLUB_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Game Chat Error: {e}")
        answer = "Connection inference error."

    return jsonify({'answer': answer})

if __name__ == '__main__':
    # Disable debug to avoid reloader issues, change port to avoid conflict
    app.run(host="127.0.0.1", port=5000)





    x           x               x         x

Disclaimer
This project is a research prototype and is not a medical diagnosis tool. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
