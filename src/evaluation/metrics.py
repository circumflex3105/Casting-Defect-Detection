import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.model.dataset import create_data_loader
from src.model.resnet import initialize_resnet_classifier
from src.utils.utils import get_computation_device

DEFAULT_BATCH_SIZE = 32
RAW_DATA_DIRECTORY = "data/raw"
PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
TEST_CSV_PATH = "data/splits/test.csv"
MODEL_SAVE_PATH = "models/best_model.pth"
POSITIVE_CLASS_LABEL = 1

def parse_command_line_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-preprocessed", action="store_true")
  parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
  return parser.parse_args()

def load_trained_model(device):
  model = initialize_resnet_classifier()
  model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
  model.to(device)
  model.eval()
  return model

def collect_predictions_and_labels(model, data_loader, device):
  all_true_labels = []
  all_predicted_labels = []
  progress_bar = tqdm(data_loader, desc="Evaluating Test Set")
  
  with torch.no_grad():
    for images, labels in progress_bar:
      images = images.to(device)
      
      predictions = model(images)
      _, predicted_classes = torch.max(predictions, 1)
      
      all_true_labels.extend(labels.cpu().numpy())
      all_predicted_labels.extend(predicted_classes.cpu().numpy())
      
  return all_true_labels, all_predicted_labels

def print_evaluation_metrics(true_labels, predicted_labels):
  accuracy = accuracy_score(true_labels, predicted_labels)
  precision = precision_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS_LABEL)
  recall = recall_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS_LABEL)
  f1 = f1_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS_LABEL)
  matrix = confusion_matrix(true_labels, predicted_labels)
  
  print("\n--- Final Test Set Evaluation ---")
  print(f"Accuracy:  {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall:    {recall:.4f}")
  print(f"F1-Score:  {f1:.4f}")
  print("\nConfusion Matrix:")
  print(f"                 Predicted OK    Predicted Defect")
  print(f"Actual OK        {matrix[0][0]:<15} {matrix[0][1]}")
  print(f"Actual Defect    {matrix[1][0]:<15} {matrix[1][1]}\n")

def execute_evaluation_pipeline():
  arguments = parse_command_line_arguments()
  data_directory = PREPROCESSED_DATA_DIRECTORY if arguments.use_preprocessed else RAW_DATA_DIRECTORY
  device = get_computation_device()
  
  test_data_loader = create_data_loader(
    TEST_CSV_PATH, 
    data_directory, 
    arguments.batch_size, 
    is_training=False
  )
  
  trained_model = load_trained_model(device)
  
  true_labels, predicted_labels = collect_predictions_and_labels(
    trained_model, 
    test_data_loader, 
    device
  )
  
  print_evaluation_metrics(true_labels, predicted_labels)

if __name__ == "__main__":
  execute_evaluation_pipeline()