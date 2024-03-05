import argparse
import os

import warnings

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# model_input_size = {
#     "2.7_80x80_MiniFASNetV2.pth": (80, 80, 'MiniFASNetV2', 2.7),
#     "4_0_0_80x80_MiniFASNetV1SE.pth" : (80, 80, 'MiniFASNetV1SE', 4.0)
# }


def get_embeddings(image_path, model_path = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_path)

    image_bbox = model_test.get_bbox(image)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    embedding = model_test.extract_embeddings(img, model_path)
    return embedding

# Example usage
image_path = "/Users/kassymmukhanbetiyar/Development/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof/Data/test/3613/spoof/511091.png"
# model_path = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
# model_name = "2.7_80x80_MiniFASNetV2.pth"
# device_id = 0

embeddings = get_embeddings(image_path)
print(embeddings)