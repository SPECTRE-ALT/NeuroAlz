import torch
from model.alzheimers_model import AlzheimerNet
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import os
import time
import copy
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    """
    Train the model with Early Stopping and sophisticated metrics tracking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    early_stopping_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Update scheduler after training batch
                if phase == 'train':
                    # Check if scheduler exists (it will be added in main)
                    try:
                        scheduler.step()
                    except NameError:
                        pass

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model & Early Stopping Logic
            if phase == 'val':
                if epoch_loss < best_loss: # Focusing on Loss for early stopping as per good practice
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    
        print()
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 1. Setup Directories
    # Try the merged dataset first, otherwise fallback
    data_dir = "final_dataset_merged"
    if not os.path.exists(data_dir):
        print(f"Merged dataset not found at '{data_dir}'. Checking for standard archive...")
        data_dir = r"archive (1)\combined_images"
        if not os.path.exists(data_dir):
            print("Error: No dataset found. Please upload data or run utils/merge_datasets.py")
            return

    print(f"Loading data from: {data_dir}")

    # 2. Data Transforms (Standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # EfficientNet standard input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Smart Splitting (Stratified split is better, but random split is decent for large data)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 4. Handle Class Imbalance with WeightedRandomSampler
    # Calculate weights for all samples in the training set
    print("Calculating class weights for balancing...")
    
    # Get labels from the underlying dataset (indices need to be mapped if using Subset/RandomSplit)
    # This is tricky with random_split. A cleaner way for weighting is to calculate on the full set indices.
    
    # Get all targets
    targets = [s[1] for s in full_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    
    # Assign weight to each sample
    sample_weights = [class_weights[t] for t in targets]
    
    # Now we need to filter these weights for only the training indices
    train_indices = train_dataset.indices
    train_sample_weights = [sample_weights[i] for i in train_indices]
    
    # Create Sampler
    sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))

    # 5. Data Loaders
    # Note: Shuffle must be False when using a sampler
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0) # Batch 16 from research
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Classes: {full_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 6. Initialize sophisticated model
    model = AlzheimerNet(num_classes=4)
    
    # 7. Optimizer, Loss & Scheduler
    # We use a weighted loss with Label Smoothing (0.1)
    # Label Smoothing prevents the model from being over-confident, which causes false positives.
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Cosine Annealing helps find the absolute global minimum for better accuracy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 8. Train
    # Extended training for "100% accuracy" goal - minimum 20 mins expected on CPU
    print("Starting intensive training session (Target: ~20+ mins)...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=15)

    # 9. Save
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    save_path = 'saved_models/alzheimer_model_sophisticated.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Sophisticated model saved to: {save_path}")
    print("Update api/app.py to load this new model file if you want to use it immediately.")

if __name__ == "__main__":
    main()
    ###done 
