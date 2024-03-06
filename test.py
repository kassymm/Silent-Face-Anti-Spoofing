import argparse
import os

import warnings

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def test(image_path, model_dir="./resources/anti_spoof_models", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_path)

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    # test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
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
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    return label, value, prediction


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--image_path", type=str, default="image_F1.jpg", help="image used to test"
    )
    args = parser.parse_args()
    test(args.image_path, args.model_dir, args.device_id)
