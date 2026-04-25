import kagglehub
import os

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")

print("Path to dataset files:", path)

# List files to see what we got
print("Files in dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
