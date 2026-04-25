import os
import shutil
import sys
from collections import Counter

def merge_datasets(source_dirs, target_dir):
    """
    Merges multiple datasets into a single target directory.
    Expects standard structure: Root/Class/Image.jpg
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    
    # Create class subfolders in target
    for class_name in CLASSES:
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

    total_images = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found: {source_dir}")
            continue
            
        print(f"Processing source: {source_dir}")
        
        # Walk through source dir
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Determine class from parent folder name
                    # Try to match loosely (case insensitive)
                    parent_folder = os.path.basename(root)
                    target_class = None
                    
                    # Normalize names
                    if 'non' in parent_folder.lower(): target_class = 'NonDemented'
                    elif 'very' in parent_folder.lower(): target_class = 'VeryMildDemented'
                    elif 'mild' in parent_folder.lower(): target_class = 'MildDemented'
                    elif 'moderate' in parent_folder.lower(): target_class = 'ModerateDemented'
                    
                    if target_class:
                        src_path = os.path.join(root, file)
                        # Create unique filename to prevent overwrite
                        unique_name = f"{os.path.basename(source_dir)}_{total_images}_{file}"
                        dst_path = os.path.join(target_dir, target_class, unique_name)
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            total_images += 1
                            if total_images % 1000 == 0:
                                print(f"  Copied {total_images} images...")
                        except Exception as e:
                            print(f"  Error copying {src_path}: {e}")
                            
    print(f"\nMerge Complete. Total images: {total_images}")
    
    # Print statistics
    stats = []
    for class_name in CLASSES:
        path = os.path.join(target_dir, class_name)
        count = len(os.listdir(path))
        stats.append((class_name, count))
        
    print("\nDataset Statistics:")
    print("-" * 30)
    for name, count in stats:
        print(f"{name}: {count}")
    print("-" * 30)

if __name__ == "__main__":
    # Define your source directories here
    # Assuming standard locations based on user discussion
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Potential source paths
    sources = [
        os.path.join(base_path, "archive (1)", "combined_images"),
        os.path.join(base_path, "archive no 2 OASIS", "Data")
    ]
    
    target = os.path.join(base_path, "final_dataset_merged")
    
    print("Beginning Dataset Merge...")
    merge_datasets(sources, target)
    print(f"\nReady for training! Dataset located at: {target}")
