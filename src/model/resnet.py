import torch.nn as nn
from torchvision import models

NUMBER_OF_CLASSES = 2

def initialize_resnet_classifier():
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  input_features = model.fc.in_features
  new_layer = nn.Linear(input_features, NUMBER_OF_CLASSES)
  model.fc = new_layer
  return model
