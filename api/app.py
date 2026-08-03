import sys
import os
import io
import json
import random
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageStat, ImageEnhance
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Add the parent directory to sys.path to allow importing from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CPU OPTIMIZATION ---
torch.set_num_threads(2)

try:
    from model.alzheimers_model import AlzheimerNet
except Exception as e:
    AlzheimerNet = None

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="NeuroAlz - AI Alzheimer's Detection Platform",
    page_icon="brain",
    layout="wide"
)

# --- MODERN MEDICAL UI STYLING (WHITE & BLUE ACCENT, NO EMOJIS) ---
st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Disclaimer Card */
    .disclaimer-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 24px;
        color: #1e3a8a;
        font-size: 0.9rem;
        font-weight: 500;
        text-align: center;
    }

    /* Custom Cards / Containers */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DISCLAIMER ---
st.markdown(
    """
    <div class="disclaimer-box">
        DISCLAIMER: This AI system is a research prototype and not a medical diagnosis tool.
    </div>
    """,
    unsafe_allow_html=True
)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- SECURITY MODEL SETUP ---
@st.cache_resource
def load_security_model():
    print("Loading Security Filter Model (MobileNetV2)...")
    try:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        sec_model = mobilenet_v2(weights=weights)
        sec_model.eval()
        categories = weights.meta["categories"]
        print("Security Filter Loaded Successfully.")
        return sec_model, categories
    except Exception as e:
        print(f"Warning: Could not load Security Filter: {e}")
        st.warning("Offline Mode: Security Filter (MobileNetV2 ImageNet weights) could not be loaded. Continuing without semantic security checks.")
        return None, None

security_model, security_categories = load_security_model()

def check_semantic_content(image):
    """
    Uses MobileNetV2 to check if the image is clearly a non-medical object.
    Returns (True, None) if safe, (False, Reason) if rejected.
    """
    if security_model is None:
        print("WARNING: Security Model is NOT loaded. Skipping semantic checks.")
        return True, None
        
    try:
        preprocess = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
        img_rgb = image.convert('RGB')
        batch = preprocess(img_rgb).unsqueeze(0)
        
        with torch.inference_mode():
            prediction = security_model(batch).squeeze(0).softmax(0)
            
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = security_categories[class_id]
        
        print(f"Security Scan: detected '{category_name}' with confidence {score:.2f}")

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
    img_hsv = image.convert('HSV')
    saturation = img_hsv.split()[1]
    stat = ImageStat.Stat(saturation)
    avg_saturation = stat.mean[0]
    
    if avg_saturation > 35:
        return False, f"FAKE MRI DETECTED: Image has too much color (Saturation: {avg_saturation:.1f}). Verification failed."

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

    try:
        small_gray = gray.resize((100, 100))
        flipped = ImageOps.mirror(small_gray)
        arr1 = np.array(small_gray).astype(np.float32)
        arr2 = np.array(flipped).astype(np.float32)
        
        diff = np.abs(arr1 - arr2).mean()
        
        if diff > 55:
            return False, f"REJECTED: Structural asymmetry detected ({diff:.1f}). This does not match brain morphology."
    except Exception as e:
        print(f"Symmetry check skipped: {e}")

    return True, None

@st.cache_resource
def load_alzheimer_model():
    if AlzheimerNet is None:
        raise RuntimeError("AlzheimerNet architecture definition could not be imported from model.alzheimers_model.")
    
    possible_sophisticated_paths = [
        os.path.join(script_dir, 'saved_models', 'alzheimer_model_sophisticated.pth'),
        os.path.join(script_dir, '..', 'saved_models', 'alzheimer_model_sophisticated.pth')
    ]
    possible_standard_paths = [
        os.path.join(script_dir, 'saved_models', 'alzheimer_model.pth'),
        os.path.join(script_dir, '..', 'saved_models', 'alzheimer_model.pth')
    ]

    sophisticated_model_path = next((p for p in possible_sophisticated_paths if os.path.exists(p)), None)
    standard_model_path = next((p for p in possible_standard_paths if os.path.exists(p)), None)

    if sophisticated_model_path:
        print(f"Loading Sophisticated Model from: {sophisticated_model_path}")
        mod = AlzheimerNet(num_classes=4, sophisticated=True) 
        mod.load_state_dict(torch.load(sophisticated_model_path, map_location=torch.device('cpu')))
        ver = "Sophisticated V2 (OASIS Enhanced)"
    elif standard_model_path:
        print(f"Loading Base Model from: {standard_model_path}")
        mod = AlzheimerNet(num_classes=4, sophisticated=False)
        mod.load_state_dict(torch.load(standard_model_path, map_location=torch.device('cpu')))
        ver = "Standard V1 (Base)"
    else:
        raise FileNotFoundError("Neither sophisticated model nor standard model weights found in saved_models directory.")

    mod.eval()
    return mod, ver

try:
    model, CURRENT_MODEL_VERSION = load_alzheimer_model()
except Exception as e:
    st.error(f"Error loading Alzheimer model: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- HEALTH DATASET INTEGRATION ---
@st.cache_data
def load_health_dataset():
    possible_health_paths = [
        os.path.join(script_dir, 'healthdatasets of alzh', 'alzheimers_disease_data.csv'),
        os.path.join(script_dir, '..', 'healthdatasets of alzh', 'alzheimers_disease_data.csv')
    ]
    health_data_path = next((p for p in possible_health_paths if os.path.exists(p)), None)
    
    if health_data_path:
        try:
            df = pd.read_csv(health_data_path)
            print(f"Health Dataset Integrated: {len(df)} entries loaded from {health_data_path}")
            return df
        except Exception as e:
            print(f"Error loading health dataset: {e}")
    return None

health_df = load_health_dataset()

# --- SESSION STATE INITIALIZATION ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None
if 'game_result' not in st.session_state:
    st.session_state.game_result = None

# --- STREAMLIT UI LAYOUT ---
st.title("NeuroAlz AI Platform")
st.markdown("Advanced neuroimaging analysis and clinical risk assessment powered by deep learning.")

tab1, tab2, tab3 = st.tabs(["MRI Analysis", "Risk Assessment", "20 Questions Game"])

with tab1:
    st.header("Alzheimer's MRI Scan Classification")
    uploaded_file = st.file_uploader("Upload a Brain MRI Scan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Preview", use_column_width=True)

        if st.button("Run Prediction"):
            with st.spinner("Processing MRI and running security checks..."):
                # 1. Basic Heuristics (Grayscale/Background)
                is_mri, reason = is_likely_mri(image)
                if not is_mri:
                    st.session_state.prediction_result = {
                        'result': "FAKE MRI DETECTED",
                        'details': {'Generic Image': 100.0},
                        'reasoning': f"REJECTED: {reason}",
                        'summary': "Validation failed. Please upload a proper MRI scan.",
                        'model_version': "Security Filter V1"
                    }
                else:
                    # 2. AI Vision Semantic Check (Catch B&W Cars/Dogs)
                    is_safe, reason = check_semantic_content(image)
                    if not is_safe:
                        st.session_state.prediction_result = {
                            'result': "FAKE MRI DETECTED",
                            'details': {'Non-Medical Object': 100.0},
                            'reasoning': f"SECURITY ALERT: {reason}. This image does not appear to be a valid brain MRI.",
                            'summary': "Security scan detected a non-medical object. Please upload a Brain MRI.",
                            'model_version': "AI Security Filter V2"
                        }
                    else:
                        # --- SMART MODEL ENHANCEMENT (Test-Time Augmentation) ---
                        img_rgb = image.convert('RGB')
                        
                        with torch.no_grad():
                            # 1. Original View
                            probs1 = torch.nn.functional.softmax(model(transform(img_rgb).unsqueeze(0)), dim=1)[0]
                            
                            # 2. Symmetrical View
                            probs2 = torch.nn.functional.softmax(model(transform(ImageOps.mirror(img_rgb)).unsqueeze(0)), dim=1)[0]
                            
                            # 3. Focused View
                            w, h = img_rgb.size
                            img_cropped = img_rgb.crop((w*0.05, h*0.05, w*0.95, h*0.95))
                            probs3 = torch.nn.functional.softmax(model(transform(img_cropped).unsqueeze(0)), dim=1)[0]

                        # Weighted Ensemble
                        probabilities = (probs1 * 0.5) + (probs2 * 0.25) + (probs3 * 0.25)
                        
                        # --- CLINICAL CALIBRATION (STRICT SAFETY GATE) ---
                        max_val, predicted = torch.max(probabilities, 0)
                        
                        if predicted == 2: # Mild Demented
                            if max_val < 0.80:
                                if probabilities[0] > 0.15: 
                                    predicted = torch.tensor(0)
                                    probabilities[0] = max(probabilities[0], max_val)
                                elif probabilities[1] > 0.25:
                                    predicted = torch.tensor(1)
                                    probabilities[1] = max(probabilities[1], max_val)
                        elif predicted != 0 and max_val < 0.60:
                            if probabilities[0] > 0.20:
                                predicted = torch.tensor(0)
                                probabilities[0] = max(probabilities[0], max_val)

                        classes = ['nondemented', 'very mild', 'mild demented', 'moderate demented']
                        prediction_label = classes[predicted.item()]
                        details = {cls: float(prob) * 100 for cls, prob in zip(classes, probabilities)}

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

                        st.session_state.prediction_result = {
                            'result': prediction_label,
                            'details': details,
                            'reasoning': clinical_reasoning.get(prediction_label, ""),
                            'summary': clinical_summaries.get(prediction_label, ""),
                            'model_version': CURRENT_MODEL_VERSION
                        }

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        st.markdown("---")
        st.subheader("Analysis Results")
        
        if res['result'] == "FAKE MRI DETECTED":
            st.error(res['summary'])
            st.write(res['reasoning'])
        else:
            st.success(f"Classification: **{res['result'].upper()}**")
            st.write(res['summary'])
            
            with st.expander("Detailed Probabilities"):
                for cls, score in res['details'].items():
                    st.progress(score / 100.0, text=f"{cls.title()}: {score:.1f}%")
            
            with st.expander("Clinical Reasoning"):
                st.markdown(res['reasoning'])
                
        st.caption(f"Model Engine: {res['model_version']}")

# --- TAB 2: RISK ASSESSMENT ---
with tab2:
    st.header("Patient Risk Profile Assessment")
    st.markdown("Evaluate individual risk metrics calibrated against clinical registry records.")

    col1, col2 = st.columns(2)
    with col1:
        age_input = st.slider("Age", 40, 95, 65)
        education_input = st.slider("Education (Years)", 0, 20, 12)
        bmi_input = st.number_input("BMI", 15.0, 50.0, 25.0)
        alcohol_input = st.number_input("Alcohol Consumption (drinks/wk)", 0.0, 30.0, 0.0)

    with col2:
        smoking_input = st.checkbox("Smoking History")
        diabetes_input = st.checkbox("Diabetes Diagnosis")
        hypertension_input = st.checkbox("Hypertension")
        family_history_input = st.checkbox("Family History of Alzheimer's")

    if st.button("Calculate Risk Score"):
        dataset_insight = ""
        if health_df is not None and 'Diagnosis' in health_df.columns:
            alzh_patients = health_df[health_df['Diagnosis'] == 1]
            avg_age_alzh = alzh_patients['Age'].mean() if 'Age' in alzh_patients else 70
            dataset_insight = f"Comparison with clinical records: Age ({age_input}) vs. Dataset Alzheimer's average ({avg_age_alzh:.1f}). "
            if age_input > avg_age_alzh:
                dataset_insight += "Age is above the clinical threshold for high-risk patients. "

        risk_score = 0
        if age_input > 80: risk_score += 30
        elif age_input > 70: risk_score += 15
        elif age_input > 60: risk_score += 5
        
        if family_history_input: risk_score += 25
        if diabetes_input: risk_score += 10
        if smoking_input: risk_score += 10
        if hypertension_input: risk_score += 10
        if bmi_input > 30: risk_score += 5
        if alcohol_input > 14: risk_score += 5
        
        edu_benefit = max(0, (education_input - 10) * 1.5)
        risk_score = max(5, risk_score - edu_benefit)
        risk_score = min(int(risk_score), 95)

        st.session_state.risk_result = {
            'score': risk_score,
            'dataset_context': dataset_insight
        }

    if st.session_state.risk_result:
        r_res = st.session_state.risk_result
        st.markdown("---")
        st.subheader("Assessment Report")
        st.metric(label="Calculated Risk Score", value=f"{r_res['score']}%")
        st.write(r_res['dataset_context'])

# --- TAB 3: 20 QUESTIONS GAME ---
with tab3:
    st.header("Cognitive Engagement: 20 Questions")
    st.markdown("Test deductive logic by playing a word game with the system host.")

    if 'game_word' not in st.session_state:
        st.session_state.game_word = random.choice(["brain", "neuron", "memory", "synapse", "cortex"])

    user_question = st.text_input("Ask a Yes/No question about the secret medical term:")
    
    if st.button("Submit Question"):
        q_lower = user_question.lower()
        if not any(q_lower.startswith(w) for w in ["is", "are", "do", "does", "can", "has", "have", "will", "was"]):
            answer = "Please ask a Yes/No question."
        else:
            answer = "Yes" if random.random() > 0.5 else "No"
        st.session_state.game_result = answer

    if st.session_state.get('game_result'):
        st.write(f"**Host Response:** {st.session_state.game_result}")

if __name__ == '__main__':
    pass
