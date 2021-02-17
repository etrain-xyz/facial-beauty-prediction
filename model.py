import config
import torchvision
import torch.nn as nn
import torch

def load_model(model_arch):
    model = getattr(torchvision.models, model_arch)(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_features=num_ftrs, out_features=1)
    model = model.to(torch.device(config.device))
    return model