import os
import pandas as pd
script_dir = os.path.dirname(os.path.abspath('api/app.py'))
root_dir = os.path.dirname(script_dir)
health_data_path = os.path.join(root_dir, 'healthdatasets of alzh', 'alzheimers_disease_data.csv')

print(f"Script Dir: {script_dir}")
print(f"Root Dir: {root_dir}")
print(f"Target Path: {health_data_path}")
print(f"Exists: {os.path.exists(health_data_path)}")

if os.path.exists(health_data_path):
    df = pd.read_csv(health_data_path)
    print(f"Success! Loaded {len(df)} rows.")
else:
    print("Failed to find file.")
