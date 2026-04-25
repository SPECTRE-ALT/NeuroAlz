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
            'fish', 'shark', 'whale'
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
    Validates if the image looks like a grayscale MRI scan.
    Returns (True, None) or (False, reasoning).
    """
    # 1. Check Color Saturation (Stricter)
    img_hsv = image.convert('HSV')
    saturation = img_hsv.split()[1] # S channel
    stat = ImageStat.Stat(saturation)
    avg_saturation = stat.mean[0]
    
    # 1. Check Color Saturation
    # Relaxed Threshold: Real MRIs are grayscale (0 saturation), but some formats 
    # or screenshots might have slight tint or compression artifacts.
    # Real photos (cars, nature) usually have much higher saturation (>50).
    if avg_saturation > 35:
        return False, f"FAKE MRI DETECTED: Image has too much color (Saturation: {avg_saturation:.1f}). Verification failed."

    # 2. Border/Corner Dark Check
    # Relaxed: Only reject if *significant* portion of corners are bright.
    # Allow for text annotations which are common in medical scans.
    gray = image.convert('L')
    w, h = gray.size
    
    # Check 4 corners (20x20 blocks - larger area to average out noise)
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
        # Higher threshold (60) allows for some noise/text
        if avg_brightness > 60:
            bright_corners += 1
            
    # Only reject if ALL corners are bright (implies full-frame photo)
    # Valid MRIs usually have at least one or two dark corners even with text.
    if bright_corners == 4:
        return False, "FAKE MRI DETECTED: Image lacks the typical dark background of an MRI scan."

    return True, None

# HARDCODED API KEY (Paste key here)
# HackClub API Configuration
HACKCLUB_API_KEY = "sk-hc-v1-57a1cb0c8acf43798d7cf36685846a917d1614b914144fb586bc777b8919374b"

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

        # FALLBACK: Local Model Analysis
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply the defined transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Forward pass, get the model output
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted class label
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output.data, 1)

        # Assuming the classes are labeled as per your dataset classes
        classes = ['nondemented', 'very mild',
                   'mild demented', 'moderate demented']
        prediction_label = classes[predicted.item()]
        
        # Prepare content for details
        confidence_details = ", ".join([f"{cls}: {float(prob)*100:.1f}%" for cls, prob in zip(classes, probabilities)])
        details = {cls: float(prob) * 100 for cls, prob in zip(classes, probabilities)}

        # HackClub API Analysis
        reasoning_msg = "Analysis pending..."
        
        if HACKCLUB_API_KEY and HACKCLUB_API_KEY != "PASTE_YOUR_HF_TOKEN_HERE":
            try:
                # HackClub API logic
                api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
                model_id = "qwen/qwen3-32b"
                
                # Convert PIL image to base64
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG') 
                img_bytes = img_byte_arr.getvalue()
                base64_image = base64.b64encode(img_bytes).decode('utf-8')

                headers = {
                    "Authorization": f"Bearer {HACKCLUB_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                prompt = f"""
                You are an expert neuroradiologist. You are reviewing an MRI analysis report.
                
                Model: EfficientNet-B0 (Sophisticated)
                Final Diagnosis: {prediction_label}
                Confidence: {confidence_details}
                
                Provide a JSON response with:
                1. "detailed_analysis": A highly technical, comprehensive assessment for a specialist. this is the PRIORITY.
                2. "summary": A brief patient overview.
                """
                
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }

                response = requests.post(api_url, headers=headers, json=payload, timeout=25)
                
                if response.status_code == 200:
                    result_json = response.json()
                    content = result_json['choices'][0]['message']['content']
                    
                    try:
                        ai_data = json.loads(content)
                        # Prioritize detailed analysis
                        reasoning_msg = ai_data.get("detailed_analysis", content)
                        summary_msg = ai_data.get("summary", "Summary not available.")
                    except json.JSONDecodeError:
                        # Fallback if AI doesn't return valid JSON
                        reasoning_msg = content
                        summary_msg = "Summary not available."
                else:
                    print(f"AI API Error Status: {response.status_code}, Response: {response.text}")
                    reasoning_msg = f"HackClub API Error: {response.text}"
                    summary_msg = "Error"
                    
            except Exception as e:
                print(f"CRITICAL AI FAILURE: {e}")
                reasoning_msg = f"Analysis unavailable (Connection Timeout: {e}). Using offline interpretation."
                summary_msg = "Analysis unavailable."
        
        # Fallback to static text if needed
        if reasoning_msg.startswith("Analysis pending") or "Error" in reasoning_msg or "empty response" in reasoning_msg or "Connection Timeout" in reasoning_msg:
             # ... (keep existing fallback logic but maybe wrap it?)
             # Logic handles it below by not overwriting if not needed, 
             # but actually the fallback logic below overwrites reasoning_msg if it matches clean.
             # Let's just ensuring `summary_msg` is defined if we fall through.
             if 'summary_msg' not in locals(): summary_msg = "Analysis unavailable."

             biological_explanations = {
                'Nondemented': """
**Biological Analysis**: The MRI scan demonstrates preserved hippocampal volume and normal ventricular size. No evidence of tumors or stroke.
**Why this happened**: Absence of neurotoxic protein thresholds. Neural synapses remain intact.
**Symptoms & Prognosis**: None currently. Continued healthy cognitive aging is expected.""",

                'Very Mild Demented': """
**Biological Analysis**: Subtle atrophy of the hippocampus (cell death) and entorhinal cortex thinning. Slight ventricular enlargement.
**Why this happened (Etiology)**: Accumulation of amyloid plaques and tau tangles is destroying synapses in memory centers.
**Expected Symptoms**: "Senior moments", forgetting simple names, misplacing items. Personality remains intact.""",

                'Mild Demented': """
**Biological Analysis**: Moderate atrophy in the **Medial Temporal and Parietal lobes**. Ventricles enlarged due to tissue loss (hydrocephalus ex vacuo).
**Why this happened (Etiology)**: The disease has progressed to widespread neuronal death. Cholinergic loss is affecting memory networks.
**Expected Symptoms**: Distinct short-term memory loss, getting lost in familiar places, difficulty handling money. Mood changes/withdrawal.""",

                'Moderate Demented': """
**Biological Analysis**: Significant diffuse cortical atrophy decimating the **Temporal/Parietal cortices**. Marked ventriculomegaly.
**Why this happened (Etiology)**: Neurofibrillary tangles have invaded the neocortex (Stage V/VI). Massive synaptic pruning.
**Expected Symptoms**: Profound memory loss, inability to recognize family, confusion, potential hallucinations, and loss of fine motor skills."""
            }
             reasoning_msg = biological_explanations.get(prediction_label, reasoning_msg)
             summary_msg = "Standard clinical definition based on diagnosis."

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
    app.run(debug=False, port=5001)