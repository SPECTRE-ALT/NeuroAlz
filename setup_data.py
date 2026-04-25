import os
import shutil
import random
from pathlib import Path

# Config
SOURCE_DIR = r"archive (1)\combined_images"
DATA_DIR = "data"
TRAIN_RATIO = 0.8

# Map source names to target names (if they differ slightly, or just lowercased)
# Source: MildDemented, ModerateDemented, NonDemented, VeryMildDemented
# Target: mild_demented, moderate_demented, nondemented, very_mild
CLASS_MAPPING = {
    "MildDemented": "mild_demented",
    "ModerateDemented": "moderate_demented",
    "NonDemented": "nondemented",
    "VeryMildDemented": "very_mild"
}

def setup_data():
    if os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' already exists. Removing to ensure fresh setup...")
        shutil.rmtree(DATA_DIR)

    os.makedirs(os.path.join(DATA_DIR, "train"))
    os.makedirs(os.path.join(DATA_DIR, "test"))

    for source_name, target_name in CLASS_MAPPING.items():
        src_class_path = os.path.join(SOURCE_DIR, source_name)
        if not os.path.exists(src_class_path):
            print(f"Warning: Source folder '{src_class_path}' not found.")
            continue

        # Create target class folders
        os.makedirs(os.path.join(DATA_DIR, "train", target_name), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "test", target_name), exist_ok=True)

        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        print(f"Processing {source_name}: {len(train_imgs)} train, {len(test_imgs)} test")

        for img in train_imgs:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(DATA_DIR, "train", target_name, img))
            
        for img in test_imgs:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(DATA_DIR, "test", target_name, img))

    print("Data setup complete.")

if __name__ == "__main__":
    setup_data()
