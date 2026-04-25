"""
This module contains the evaluation logic for the Alzheimer's detection model.

It defines functions for evaluating the trained model on a test dataset.
"""

import torch
from model.alzheimers_model import AlzheimerNet
from utils.data_utils import get_data_loaders
import os


def evaluate_model(model, test_loader):
    """
    Evaluate the Alzheimer's detection model.

    Args:
        model (nn.Module): The trained neural network model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: The accuracy of the model on the test set.

    This function evaluates the model on the test set and computes the accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


def main():
    """
    Main function to set up and start the evaluation process.

    This function loads the trained model, initializes the test data loader,
    and starts the evaluation process.
    """

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths
    train_dir = os.path.join(script_dir, 'data', 'train')
    test_dir = os.path.join(script_dir, 'data', 'test')
    model_path = os.path.join(
        script_dir, 'saved_models', 'alzheimer_model.pth')

    # Load data
    _, test_loader = get_data_loaders(train_dir, test_dir)

    # Initialize and load the model
    model = AlzheimerNet(num_classes=4)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
