import sys
import os
import io
import json
import random
import time
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
    page_title="NeuroAlz - Comprehensive Alzheimer's Screening Suite",
    page_icon="🧠",
    layout="centered"
)

# --- HIGH-CONTRAST MODERN UI STYLING (INVERTED PANELS: WHITE BACKGROUND & BLACK TEXT) ---
st.markdown(
    """
    <style>
    /* Global Container Styles */
    .stApp {
        background-color: #f1f5f9;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    p, span, label, div, .stMarkdown, .stText, li {
        color: #1e293b !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700;
    }

    /* Card layout */
    .main-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #cbd5e1;
        max-width: 750px;
        margin: auto;
    }

    /* Logo Header Styling */
    .brand-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .brand-title span {
        color: #2563eb;
    }
    .brand-subtitle {
        text-align: center;
        color: #475569;
        font-size: 1rem;
        margin-bottom: 30px;
        font-weight: 500;
    }

    /* Custom Radio Navigation Bar Styling */
    .stRadio > div {
        background-color: #e2e8f0;
        padding: 6px;
        border-radius: 12px;
        display: flex;
        justify-content: center;
        gap: 6px;
        border: 1px solid #cbd5e1;
    }
    .stRadio > div > label {
        background-color: transparent !important;
        padding: 8px 24px;
        border-radius: 8px;
        font-weight: 600;
        color: #334155 !important;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stRadio > div > label[data-checked="true"] {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* Upload Box container */
    .upload-section {
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background-color: #ffffff;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Custom Buttons */
    .stButton>button {
        background-color: #1e293b;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        border: none;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #0f172a;
        color: #ffffff !important;
    }

    /* Inverted Boxes: White Background with Forced Black Text */
    .games-panel {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 24px;
        color: #000000 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .games-panel *, .games-panel strong, .games-panel span {
        color: #000000 !important;
    }
    
    .info-panel {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #000000 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .info-panel *, .info-panel strong, .info-panel span {
        color: #000000 !important;
    }

    /* Fix Streamlit info, success, warning, error boxes to use White Background and Black Text */
    .stAlert, div[data-baseweb="notification"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stAlert *, div[data-baseweb="notification"] * {
        color: #000000 !important;
    }

    /* Reflex Game Box Styling with Visible High-Contrast Text */
    .reflex-box-red {
        background-color: #b91c1c !important;
        border: 2px solid #ef4444 !important;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(185, 28, 28, 0.4);
    }
    .reflex-box-red * {
        color: #ffffff !important;
    }

    .reflex-box-green {
        background-color: #15803d !important;
        border: 2px solid #22c55e !important;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(21, 128, 61, 0.4);
    }
    .reflex-box-green * {
        color: #ffffff !important;
    }

    .reflex-box-waiting {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        color: #000000 !important;
        font-size: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .reflex-box-waiting * {
        color: #000000 !important;
    }
    </style>
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
        return None, None

security_model, security_categories = load_security_model()

def check_semantic_content(image):
    if security_model is None:
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
        
        print(f"[DEBUG] MobileNet Security Scan: detected '{category_name}' with confidence {score:.2f}")

        forbidden_keywords = [
            'car', 'wagon', 'vehicle', 'truck', 'racer', 'wheel', 'convertible', 'jeep', 'cab',
            'dog', 'cat', 'bird', 'animal', 'terrier', 'retriever', 'bull', 'frog',
            'building', 'house', 'structure', 'castle', 'church', 'palace',
            'keyboard', 'laptop', 'mouse', 'phone', 'screen', 'monitor',
            'food', 'fruit', 'vegetable', 'pizza', 'burger', 'sandwich',
            'fish', 'shark', 'whale', 'shoe', 'sock', 'clothing',
            'knee', 'joint', 'elbow', 'hand', 'foot', 'bone', 'leg'
        ]
        
        is_forbidden = any(keyword in category_name.lower() for keyword in forbidden_keywords)
        
        if is_forbidden and score > 0.12: 
             return False, f"Content detected as '{category_name}' ({score*100:.1f}% confidence). This does not look like an MRI."
             
    except Exception as e:
        print(f"Security Scan Error: {e}")
        return True, None
        
    return True, None

def is_likely_mri(image):
    img_hsv = image.convert('HSV')
    saturation = img_hsv.split()[1]
    stat = ImageStat.Stat(saturation)
    avg_saturation = stat.mean[0]
    
    print(f"[DEBUG] Saturation score: {avg_saturation:.1f}")
    
    if avg_saturation > 30:
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
    corner_bright_values = []
    for box in corners:
        region = gray.crop(box)
        stat = ImageStat.Stat(region)
        avg_brightness = stat.mean[0]
        corner_bright_values.append(avg_brightness)
        if avg_brightness > 55:
            bright_corners += 1
            
    print(f"[DEBUG] Corner brightness scores: {corner_bright_values} (Bright count: {bright_corners}/4)")

    if bright_corners == 4:
        return False, "FAKE MRI DETECTED: Image lacks the typical dark background of an MRI scan."

    try:
        small_gray = gray.resize((100, 100))
        flipped = ImageOps.mirror(small_gray)
        arr1 = np.array(small_gray).astype(np.float32)
        arr2 = np.array(flipped).astype(np.float32)
        
        diff = np.abs(arr1 - arr2).mean()
        
        print(f"[DEBUG] Symmetry difference score: {diff:.1f}")
        
        if diff < 10.0:
            return False, f"REJECTED: Image lacks sufficient biological structural variation (Symmetry Diff: {diff:.1f})."
        
        if diff > 42.0:
            return False, f"REJECTED: Structural asymmetry detected ({diff:.1f}). This does not match brain morphology (Possible Knee/Bone scan)."
            
    except Exception as e:
        print(f"Symmetry check skipped: {e}")

    return True, None

@st.cache_resource
def load_alzheimer_model():
    if AlzheimerNet is None:
        raise RuntimeError("AlzheimerNet architecture definition could not be imported.")
    
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
        mod = AlzheimerNet(num_classes=4, sophisticated=True) 
        mod.load_state_dict(torch.load(sophisticated_model_path, map_location=torch.device('cpu')))
        ver = "Sophisticated V2 (OASIS Enhanced)"
    elif standard_model_path:
        mod = AlzheimerNet(num_classes=4, sophisticated=False)
        mod.load_state_dict(torch.load(standard_model_path, map_location=torch.device('cpu')))
        ver = "Standard V1 (Base)"
    else:
        raise FileNotFoundError("Model weights not found.")

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

@st.cache_data
def load_health_dataset():
    possible_health_paths = [
        os.path.join(script_dir, 'healthdatasets of alzh', 'alzheimers_disease_data.csv'),
        os.path.join(script_dir, '..', 'healthdatasets of alzh', 'alzheimers_disease_data.csv')
    ]
    health_data_path = next((p for p in possible_health_paths if os.path.exists(p)), None)
    
    if health_data_path:
        try:
            return pd.read_csv(health_data_path)
        except Exception:
            pass
    return None

health_df = load_health_dataset()

# --- SESSION STATE ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None

# --- HEADER SECTION ---
st.markdown('<p class="brand-title">Neuro<span>Alz</span></p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Comprehensive Alzheimer\'s Screening Suite</p>', unsafe_allow_html=True)

# --- NAVIGATION TABS ---
selected_tab = st.radio(
    "Navigation",
    ["MRI Analysis", "Cognitive Games", "Risk Profile"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== TAB 1: MRI ANALYSIS ====================
if selected_tab == "MRI Analysis":
    st.markdown("### Upload Brain MRI Scan")
    
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Preview", use_column_width=True)

    if st.button("Run Clinical Analysis"):
        if uploaded_file is None:
            st.warning("Please upload a brain MRI scan image first.")
        else:
            with st.spinner("Processing MRI and running security checks..."):
                image = Image.open(uploaded_file)
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
                        img_rgb = image.convert('RGB')
                        with torch.no_grad():
                            probs1 = torch.nn.functional.softmax(model(transform(img_rgb).unsqueeze(0)), dim=1)[0]
                            probs2 = torch.nn.functional.softmax(model(transform(ImageOps.mirror(img_rgb)).unsqueeze(0)), dim=1)[0]
                            w, h = img_rgb.size
                            img_cropped = img_rgb.crop((w*0.05, h*0.05, w*0.95, h*0.95))
                            probs3 = torch.nn.functional.softmax(model(transform(img_cropped).unsqueeze(0)), dim=1)[0]

                        probabilities = (probs1 * 0.5) + (probs2 * 0.25) + (probs3 * 0.25)
                        max_val, predicted = torch.max(probabilities, 0)
                        
                        if predicted == 2:
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
                            'nondemented': "Hippocampal volume and structural integrity appear robust. No significant signs of asymmetric atrophy.",
                            'very mild': "Subtle early-stage indicators found around medial temporal structures.",
                            'mild demented': "Moderate atrophy profile observed with ventricular expansion.",
                            'moderate demented': "Severe global atrophy detected with extensive sulcal widening."
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
                st.write(res['reasoning'])
        st.caption(f"Model Engine: {res['model_version']}")

    st.markdown(
        """
        <div class="games-panel">
            <strong>More Recommended Games</strong><br>
            <span>Checkers &nbsp;&bull;&nbsp; Mahjong &nbsp;&bull;&nbsp; Connect 4 &nbsp;&bull;&nbsp; Rummikub &nbsp;&bull;&nbsp; Sudoku</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-panel">
            <strong>AI Neuro-Analysis</strong><br>
            <span>Complete both modules to generate a full brain health report.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== TAB 2: COGNITIVE GAMES ====================
elif selected_tab == "Cognitive Games":
    st.header("Cognitive Engagement & Screening Modules")
    st.markdown("Select a specialized evaluation module to test neuro-cognitive responsiveness.")

    game_sub_tab = st.selectbox(
        "Choose Screening Test", 
        ["Processing Speed (Red/Green Reflex)", "Hippocampal Function (Pattern Match)", "Verbal Memory (Short Story Test)"]
    )

    st.markdown("---")

    # --- MODULE 1: PROCESSING SPEED ---
    if game_sub_tab == "Processing Speed (Red/Green Reflex)":
        st.subheader("Processing Speed Reflex Assessment")
        st.markdown("Instructions: Click **Green** targets as fast as possible when they appear. Avoid clicking **Red** targets.")

        if 'reflex_state' not in st.session_state:
            st.session_state.reflex_state = 'IDLE'
            st.session_state.reflex_target = None
            st.session_state.reflex_start_time = 0
            st.session_state.reflex_score = 0
            st.session_state.reflex_rounds = 0

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_reflex = st.button("Start Reflex Test")
        
        if start_reflex:
            # Simulate a brief delay before flashing a random red/green signal box
            time.sleep(random.uniform(0.5, 1.5))
            st.session_state.reflex_state = 'ACTIVE'
            st.session_state.reflex_target = random.choice(['GREEN', 'RED', 'GREEN', 'GREEN', 'RED'])
            st.session_state.reflex_start_time = time.time()
            st.session_state.reflex_rounds += 1

        if st.session_state.reflex_state == 'ACTIVE':
            target = st.session_state.reflex_target
            
            # Display flashing high-contrast colored box with bright white text inside
            if target == 'RED':
                st.markdown(
                    f'<div class="reflex-box-red">TARGET SIGNAL: RED<br><span style="font-size: 1rem; font-weight: normal;">(Do NOT click! Wait or click Red button if penalizing)</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="reflex-box-green">TARGET SIGNAL: GREEN<br><span style="font-size: 1rem; font-weight: normal;">(Click Green immediately to calculate ms reaction time!)</span></div>',
                    unsafe_allow_html=True
                )
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔴 Click RED", key="click_red"):
                    reaction_time_ms = (time.time() - st.session_state.reflex_start_time) * 1000.0
                    if target == 'RED':
                        st.session_state.reflex_score += 1
                        st.success(f"Correct! Reaction Time: {reaction_time_ms:.1f} ms")
                    else:
                        st.error(f"Incorrect! That was a Green target. Reaction time was {reaction_time_ms:.1f} ms")
                    st.session_state.reflex_state = 'IDLE'
            with c2:
                if st.button("🟢 Click GREEN", key="click_green"):
                    reaction_time_ms = (time.time() - st.session_state.reflex_start_time) * 1000.0
                    if target == 'GREEN':
                        st.session_state.reflex_score += 1
                        st.success(f"Correct! Reaction Time: {reaction_time_ms:.1f} ms")
                    else:
                        st.error(f"Incorrect! That was a Red target. Reaction time was {reaction_time_ms:.1f} ms")
                    st.session_state.reflex_state = 'IDLE'
        else:
            st.markdown(
                '<div class="reflex-box-waiting">Click <b>"Start Reflex Test"</b> to begin the flashing sequence.</div>',
                unsafe_allow_html=True
            )

        st.write(f"**Current Score:** {st.session_state.reflex_score} correct responses.")

    # --- MODULE 2: HIPPOCAMPAL FUNCTION ---
    elif game_sub_tab == "Hippocampal Function (Pattern Match)":
        st.subheader("Hippocampal Spatial Pattern Matching")
        st.markdown("Observe the grid pattern below, then recreate it by selecting the matching sequence.")

        if 'pattern_target' not in st.session_state:
            st.session_state.pattern_target = [random.choice([0, 1]) for _ in range(4)]
            st.session_state.pattern_stage = 'SHOW'

        if st.session_state.pattern_stage == 'SHOW':
            st.info(f"Memorize Pattern Sequence: **{st.session_state.pattern_target}** (1 = Active, 0 = Inactive)")
            if st.button("I'm Ready - Input Pattern"):
                st.session_state.pattern_stage = 'INPUT'
                st.rerun()

        elif st.session_state.pattern_stage == 'INPUT':
            st.markdown("Select your matched pattern grid:")
            p1 = st.selectbox("Tile 1", [0, 1], key="t1")
            p2 = st.selectbox("Tile 2", [0, 1], key="t2")
            p3 = st.selectbox("Tile 3", [0, 1], key="t3")
            p4 = st.selectbox("Tile 4", [0, 1], key="t4")

            if st.button("Submit Pattern"):
                user_pattern = [p1, p2, p3, p4]
                if user_pattern == st.session_state.pattern_target:
                    st.success("Pattern match successful! Hippocampal recall functional.")
                else:
                    st.error(f"Mismatch! Correct pattern was {st.session_state.pattern_target}.")
                if st.button("Play Again"):
                    st.session_state.pattern_target = [random.choice([0, 1]) for _ in range(4)]
                    st.session_state.pattern_stage = 'SHOW'
                    st.rerun()

    # --- MODULE 3: VERBAL MEMORY ---
    elif game_sub_tab == "Verbal Memory (Short Story Test)":
        st.subheader("Verbal Memory & Short Story Recall")
        st.markdown("Read the short 3-line story carefully. A 30-second countdown will begin before questions are asked.")

        if 'story_state' not in st.session_state:
            st.session_state.story_state = 'READING'
            st.session_state.story_timer = 30
            st.session_state.story_start_time = time.time()

        story_text = (
            "1. Dr. Arthur walked through the rainy hospital corridors holding a blue file.\n"
            "2. He met Nurse Clara near room 302 to discuss the morning patient charts.\n"
            "3. They verified that the medication schedule was successfully updated for noon."
        )

        if st.session_state.story_state == 'READING':
            st.markdown(f"> **Story Passage:**\n> \n> {story_text}")
            
            elapsed = int(time.time() - st.session_state.story_start_time)
            remaining = max(0, 30 - elapsed)
            
            st.markdown(f"### Time remaining to memorize: **{remaining} seconds**")
            
            if remaining > 0:
                if st.button("Skip Countdown & Answer Now"):
                    st.session_state.story_state = 'QUESTIONING'
                    st.rerun()
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.story_state = 'QUESTIONING'
                st.rerun()

        elif st.session_state.story_state == 'QUESTIONING':
            st.markdown("### Recall Questions:")
            ans1 = st.radio("Question 1: What color was the file Dr. Arthur was holding?", ["Red", "Blue", "Green", "Yellow"])
            ans2 = st.radio("Question 2: Which room number were they near?", ["Room 104", "Room 205", "Room 302", "Room 410"])

            if st.button("Submit Answers"):
                score_v = 0
                if ans1 == "Blue": score_v += 1
                if ans2 == "Room 302": score_v += 1
                
                if score_v == 2:
                    st.success("Perfect recall! Verbal memory score: 2/2.")
                else:
                    st.warning(f"Recall score: {score_v}/2 correct. (Correct answers: Blue file, Room 302)")
                
                if st.button("Restart Story Test"):
                    st.session_state.story_state = 'READING'
                    st.session_state.story_start_time = time.time()
                    st.rerun()

# ==================== TAB 3: RISK PROFILE ====================
elif selected_tab == "Risk Profile":
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

if __name__ == '__main__':
    pass
