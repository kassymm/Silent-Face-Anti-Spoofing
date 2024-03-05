import torch
from torchsummary import summary
from src.anti_spoof_predict import AntiSpoofPredict

model_path = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
device_id = 0

model = AntiSpoofPredict(device_id)
model._load_model(model_path)

summary(model.model, input_size=(3, 80, 80))