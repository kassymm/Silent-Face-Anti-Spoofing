import os
import pandas as pd
from embeddings import get_embeddings
import json
from tqdm import tqdm
import time

image_dir = "/Users/kassymmukhanbetiyar/Development/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof/"

def fill_embeddings_csv(model_path = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", checkpoint_path = "embeddings.csv"):
    # Get the number of processed images
    processed_count = 0
    if os.path.isfile(checkpoint_path):
        processed_count = sum(1 for _ in open(checkpoint_path)) - 1  # Subtract 1 to exclude the header row

    # Get the remaining image paths to process
    with open('test_label.json', 'r') as file:
        test_labels = json.load(file)

    remaining_image_names = list(test_labels.keys())[processed_count:]

    # Create a new DataFrame for the newly processed images
    new_embeddings_df = pd.DataFrame(columns=['Embedding1', 'Embedding2', 'Embedding3'])

    total_iterations = len(remaining_image_names)
    progress_bar = tqdm(total=total_iterations, ncols=80, dynamic_ncols=True, smoothing=0.5)
    # Initialize start time
    start_time = time.time()

    # Iterate over the remaining image paths and fill in the embeddings
    for i, image_name in enumerate(remaining_image_names, start=processed_count + 1):
        image_path = image_dir + image_name
        embeddings = get_embeddings(image_path, model_path)

        # Add the embeddings to the new DataFrame
        new_embeddings_df.loc[image_name] = embeddings.tolist()[0]  # Assuming the embedding shape is (1, 3)

        # Append the new embeddings to the existing DataFrame and CSV file every 250 iterations
        if i % 250 == 0 or i == len(remaining_image_names):
            new_embeddings_df.to_csv(checkpoint_path, mode='a', header=(i == 250))
            new_embeddings_df = pd.DataFrame(columns=['Embedding1', 'Embedding2', 'Embedding3'])
            # Update progress bar
            progress_bar.update(250 if i % 250 == 0 else i % 250)
            progress_bar.set_postfix({'Remaining': f'{total_iterations - i}', 'Elapsed': f'{time.time() - start_time:.2f}s'})
    return checkpoint_path

fill_embeddings_csv()