import argparse
import os
import warnings
import torch
import cv2
import numpy as np
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.model_lib.MiniFASNet import MiniFASNetV2


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

keep_dict = {'1.8M': [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
                      64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
                      154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
                      52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
                      128, 26, 26, 128, 26, 26, 128, 512, 512],

             '1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                       13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                       52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                       128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
             }
def get_bn_embeddings(image_path, model_path="resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", device_id=0):
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

    # Load the weights of the pre-trained model
    model = MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.2, num_classes=3, img_channel=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        # Pass the image through the model
        embedding = model(img)

        # Apply batch normalization to the embedding
        embedding = model.bn(embedding)

    return embedding

def test_model(image_path, model_path="resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_path)

    image_bbox = model_test.get_bbox(image)
    # test_speed = 0
    # sum the prediction from single model's result

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
    prediction = model_test.predict(img, model_path)
    return prediction
image_path = "/Users/kassymmukhanbetiyar/Development/Verigram/CelebA/archive/CelebA_Spoof_/CelebA_Spoof/Data/test/3613/spoof/511091.png"


# embeddings = get_embeddings(image_path)
# print(embeddings)
# prediction = test_model(image_path)
# print(prediction)