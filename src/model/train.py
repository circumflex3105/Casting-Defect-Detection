import os
import argparse
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.model.dataset import create_data_loader
from src.model.resnet import initialize_resnet_classifier
from src.utils.utils import get_computation_device

DEFAULT_EPOCH_COUNT = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
RAW_DATA_DIRECTORY = "data/raw"
PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
TRAIN_CSV_PATH = "data/splits/train.csv"
VALIDATION_CSV_PATH = "data/splits/validation.csv"
MODEL_SAVE_DIRECTORY = "models"
MODEL_SAVE_FILENAME = "best_model.pth"

def parse_command_line_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-preprocessed", action="store_true")
  parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCH_COUNT)
  parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
  parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
  return parser.parse_args()

def create_loss_function():
  return nn.CrossEntropyLoss()

def create_optimizer(model, learning_rate):
  return optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(predictions, labels):
  _, predicted_classes = torch.max(predictions, 1)
  correct_predictions = (predicted_classes == labels).sum().item()
  total_predictions = labels.size(0)
  return correct_predictions / total_predictions

def train_single_epoch(model, data_loader, loss_function, optimizer, device):
  model.train()
  total_loss = 0.0
  progress_bar = tqdm.tqdm(data_loader, desc="Training", leave=False)

  for images, labels in progress_bar:
    images = images.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_function(predictions, labels)
    
    loss.backward()
    optimizer.step()
    
    current_loss = loss.item()
    total_loss += current_loss
    progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
    
  return total_loss / len(data_loader)

def validate_single_epoch(model, data_loader, loss_function, device):
  model.eval()
  total_loss = 0.0
  total_accuracy = 0.0
  progress_bar = tqdm.tqdm(data_loader, desc="Validation", leave=False)

  with torch.no_grad():
    for images, labels in progress_bar:
      images = images.to(device)
      labels = labels.to(device)
      
      predictions = model(images)
      loss = loss_function(predictions, labels)
      
      current_loss = loss.item()
      total_loss += current_loss
      total_accuracy += calculate_accuracy(predictions, labels)
      progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
      
  average_loss = total_loss / len(data_loader)
  average_accuracy = total_accuracy / len(data_loader)
  return average_loss, average_accuracy

def save_model_weights(model, directory_path, file_name):
  Path(directory_path).mkdir(parents=True, exist_ok=True)
  full_save_path = os.path.join(directory_path, file_name)
  torch.save(model.state_dict(), full_save_path)

def execute_training_pipeline():
  arguments = parse_command_line_arguments()
  data_directory = PREPROCESSED_DATA_DIRECTORY if arguments.use_preprocessed else RAW_DATA_DIRECTORY
  device = get_computation_device()
  
  train_data_loader = create_data_loader(
    TRAIN_CSV_PATH, 
    data_directory, 
    arguments.batch_size, 
    is_training=True
  )
  
  validation_data_loader = create_data_loader(
    VALIDATION_CSV_PATH, 
    data_directory, 
    arguments.batch_size, 
    is_training=False
  )
  
  model = initialize_resnet_classifier()
  model = model.to(device)
  
  loss_function = create_loss_function()
  optimizer = create_optimizer(model, arguments.learning_rate)
  
  best_validation_loss = float("inf")

  epoch_progress_bar = tqdm.tqdm(range(arguments.epochs), desc="Overall Epochs")
  
  for epoch in epoch_progress_bar:
    training_loss = train_single_epoch(model, train_data_loader, loss_function, optimizer, device)
    validation_loss, validation_accuracy = validate_single_epoch(model, validation_data_loader, loss_function, device)
    
    epoch_progress_bar.set_postfix({
      "Train Loss": f"{training_loss:.4f}",
      "Val Loss": f"{validation_loss:.4f}",
      "Val Acc": f"{validation_accuracy:.4f}"
    })
    
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      save_model_weights(model, MODEL_SAVE_DIRECTORY, MODEL_SAVE_FILENAME)

if __name__ == "__main__":
  execute_training_pipeline()
