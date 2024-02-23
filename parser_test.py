import os
from test import test

def run_test_on_images(folder_path):
    # Iterate over all files in the folder
    for image_name in os.listdir(folder_path):
        # Check if the file is a PNG image
        if image_name.endswith('.png'):
            # Call the test function on the image
            
            test(image_name, model_dir="./resources/anti_spoof_models", device_id = 0)
            

run_test_on_images('/Volumes/Transcend/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof/Data/test/4930/spoof/')