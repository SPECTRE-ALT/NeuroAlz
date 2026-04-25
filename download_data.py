import opendatasets as od
import os

# Define the Kaggle dataset URL
dataset_url = 'https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images'

# Download the dataset
print("Downloading dataset... (You may be prompted for Kaggle credentials)")
od.download(dataset_url, data_dir='.')

# Check download
if os.path.exists('alzheimers-dataset-4-class-of-images'):
    print("Download complete.")
else:
    print("Download failed or folder name unexpected.")
