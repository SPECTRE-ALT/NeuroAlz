import os
import sys

def validate_dataset_structure(dataset_path):
    """
    Validates that the dataset folder contains the required 4 class subfolders.
    """
    required_classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path '{dataset_path}' does not exist.")
        return False
        
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found folders in '{dataset_path}':")
    for d in subdirs:
        print(f" - {d}")
        
    missing = [c for c in required_classes if c not in subdirs]
    
    if missing:
        print("\nERROR: Missing required class folders for compatibility:")
        for m in missing:
            print(f" - {m}")
        print("\nPlease rename your folders to match the existing model classes exactly.")
        return False
        
    print("\nSUCCESS: Dataset structure appears correct!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <path_to_dataset>")
    else:
        validate_dataset_structure(sys.argv[1])
