"""
This file contains the main training logic for the Alzheimer's detection model

It defines functions for training the model and the main execution flow.
"""

import torch
from model.alzheimers_model import AlzheimerNet
from utils.data_utils import get_data_loaders
from torch import optim, nn
import os


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    """
    Train the Alzheimer's detection model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimization algorithm.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 25.

    Returns:
        model (nn.Module): The trained model.

    This function trains the model for the specified number of epochs,
    updating the model parameters based on the computed loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs} completed.')

    return model


def main():
    """
    Main function to set up and start the training process.

    This function initializes the data loaders, model, optimizer, and loss function,
    then starts the training process.
    """

    # Point to the user's uploaded data
    data_dir = r"archive (1)\combined_images"
    
    # Use the modified data loader that handles splitting
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Data loaded from {data_dir}")
    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    model = AlzheimerNet(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(
        model, train_loader, criterion, optimizer, num_epochs=25)

    # Save the trained model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(trained_model.state_dict(), 'saved_models/alzheimer_model.pth')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
