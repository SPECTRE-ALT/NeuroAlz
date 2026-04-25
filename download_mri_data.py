import kagglehub
import os
import shutil

# Download the MRI dataset (approx 35MB, 6400 images)
print("Downloading MRI dataset (tourist55/alzheimers-dataset-4-class-of-images)...")
path = kagglehub.dataset_download("tourist55/alzheimers-dataset-4-class-of-images")

print("Path to dataset files:", path)

# Target directory
target_dir = os.path.join(os.getcwd(), 'data')

# The dataset structure is usually:
# /Alzheimer_s Dataset/
#    /train/
#    /test/
# We need to move these to 'data/train' and 'data/test'

source_dataset_dir = os.path.join(path, "Alzheimer_s Dataset")

if os.path.exists(source_dataset_dir):
    print(f"Copying files from {source_dataset_dir} to {target_dir}...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dataset_dir, target_dir)
    print("Dataset setup complete.")
else:
    print(f"Unexpected structure. Contents of {path}:", os.listdir(path))
