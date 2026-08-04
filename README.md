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

### 📂 Backend Source Code Update

The original Flask implementation (`app.py`) for the backend API has been included as a plain text reference file named **`app_flask_backup.txt`** in the root repository. 

* **Why a `.txt` file?** This ensures clean formatting and prevents syntax highlighting or rendering issues in the GitHub web editor.
* **Streamlit Server Note:** The Streamlit server will completely ignore this `.txt` file, so it will not interfere with your active deployment or app execution.



For those who want to try the streamlit web version out here is the link - https://neuroalz-dx3e2tcbrvvdnrjdzhvqxf.streamlit.app/
x               x                  x                      x                  x                       x                  x


Disclaimer
This project is a research prototype and is not a medical diagnosis tool. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
