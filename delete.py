import os

folder_path = "/Volumes/Transcend/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof/Data/test/4930/spoof/"
for image_name in os.listdir(folder_path):
    # Check if the file is a result.png image
    if image_name.endswith('result.png'):
        # Call the test function on the image
        imagePath = folder_path + image_name
        os.remove(imagePath)