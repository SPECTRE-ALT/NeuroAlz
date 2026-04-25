**Alzheimer's Disease Stage Detection Using Deep Learning**

This repository contains the implementation of a deep learning model designed to detect various stages of Alzheimer's disease using MRI brain scans. The project employs a transfer learning approach, utilizing the EfficientNetB0 architecture, adapted to classify images into four distinct stages: nondemented, very mild, mild, and moderate dementia. Built with PyTorch, the model is trained on a structured dataset of MRI images and provides a robust framework for early and accurate diagnosis of Alzheimer’s progression.

### Key Features:

- **Model Architecture:** Utilizes the pre-trained EfficientNetB0 model, fine-tuned for the specific task of classifying Alzheimer's disease stages from brain MRIs.
- **Data Processing:** Includes comprehensive preprocessing and augmentation techniques to optimize model training and performance.
- **API Integration:** Features a Flask-based REST API that enables users to upload MRI images and receive diagnostic predictions, facilitating easy interaction with the model.
- **Containerization:** Docker support for easy setup and deployment, ensuring consistency across different environments.
- **Evaluation Metrics:** Implements various metrics for thorough evaluation and validation of the model, ensuring high accuracy and reliability.

### Project Structure:

- `/model`: Contains the model definitions and state.
- `/data`: Dataset directory with train and test splits.
- `/utils`: Utility functions for data loading and transformations.
- `/api`: Contains the Flask application for the REST API.
- `/tests/`: Includes test scripts
- `train.py`: Script for model training.
- `evaluate.py`: Script for model evaluation and performance metrics.

### Goals:

- To provide a reliable tool for early detection of Alzheimer's disease stages, aiding in better management and treatment planning.
- To contribute to the ongoing research in medical AI by demonstrating the application of advanced machine learning techniques in healthcare diagnostics.
- To offer an accessible platform for further development and validation by the research community, healthcare professionals, and technology enthusiasts.

## Usage:

1. Train the model using **`train.py`**.
2. Evaluate the model's performance with **`evaluate.py`**.
3. Start the API server by running **`python api/app.py`**.
4. Test the API using **`python tests/test_alzheimers_api.py`**.

This project is intended for educational and research purposes, aiming to bridge the gap between medical imaging and machine learning technologies.

## Detailed Module Descriptions

### API (app.py)

Located in the `/api` directory, `app.py` sets up a Flask-based REST API for the Alzheimer's Disease Detection model:

- `/predict` endpoint for MRI image upload and prediction
- Image preprocessing using PyTorch transformations
- Integration with the trained AlzheimerNet model

### Testing (test_alzheimers_api.py)

Found in the `/tests` directory, `test_alzheimers_api.py` provides API testing functionality:

- Random selection of test images
- API endpoint testing with selected images
- Comparison of true labels with model predictions

### Training (train.py)

Located in the root directory, `train.py` contains the main training logic for the Alzheimer's detection model:

- Defines the `train_model` function for model training
- Implements the main execution flow in the `main` function
- Utilizes PyTorch for model training and optimization
- Loads data using custom data loaders from `utils/data_utils.py`
- Initializes the AlzheimerNet model, optimizer, and loss function
- Trains the model for a specified number of epochs
- Saves the trained model to the `saved_models` directory

### Evaluation (evaluate.py)

Located in the root directory, `evaluate.py` contains the evaluation logic for the Alzheimer's detection model:

- Defines the `evaluate_model` function for model evaluation
- Implements the main execution flow in the `main` function
- Loads the trained model from the `saved_models` directory
- Utilizes the test data loader to evaluate model performance
- Computes and prints the accuracy of the model on the test set
- Handles device-agnostic evaluation (CPU or CUDA)
