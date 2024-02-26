import os
import cv2
import pandas as pd
import json
from test import test
import time

# Read the JSON file
with open('test_label.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient='index')

base_directory = '/Volumes/Transcend/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof'

total_rows = len(df)
progress_interval = 250
processed_count = 0
checkpoint_file = 'progress_checkpoint.csv'

# Check if a checkpoint file exists
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file, index_col=0)
    processed_count = checkpoint_df.shape[0]
    df.update(checkpoint_df)

start_time = time.time()

for row_name in df.index[processed_count:]:
    image_path = os.path.join(base_directory, row_name)
    label, value = test(image_path)
    df.loc[row_name, 'Real'] = label
    df.loc[row_name, 'Prediction Score'] = value
    processed_count += 1
    if processed_count % progress_interval == 0 or processed_count == total_rows:
        elapsed_time = time.time() - start_time
        average_time_per_sample = elapsed_time / processed_count
        estimated_remaining_time = average_time_per_sample * (total_rows - processed_count)
        print(f"Processed {processed_count}/{total_rows} samples. Elapsed time: {elapsed_time:.2f}s. Estimated remaining time: {estimated_remaining_time:.2f}s")
        # Save the DataFrame as a checkpoint
        df.iloc[:processed_count].to_csv(checkpoint_file)

# Remove the checkpoint file
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print(f"Total time taken: {elapsed_time:.2f}s")

