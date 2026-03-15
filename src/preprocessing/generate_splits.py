import os
import random
import csv
from pathlib import Path

RAW_DATA_DIRECTORY = "data/raw"
SPLITS_DIRECTORY = "data/splits"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

def get_image_paths_and_labels(base_directory):
  dataset = []
  base_path = Path(base_directory)
  for class_directory in base_path.iterdir():
    if class_directory.is_dir():
      label = class_directory.name
      for image_path in class_directory.iterdir():
        if image_path.is_file():
          dataset.append((str(image_path), label))
  return dataset

def shuffle_dataset(dataset, seed):
  random.seed(seed)
  shuffled_dataset = dataset.copy()
  random.shuffle(shuffled_dataset)
  return shuffled_dataset

def calculate_split_indices(total_items, train_ratio, validation_ratio):
  train_end_index = int(total_items * train_ratio)
  validation_end_index = train_end_index + int(total_items * validation_ratio)
  return train_end_index, validation_end_index

def split_dataset(dataset, train_end_index, validation_end_index):
  train_dataset = dataset[:train_end_index]
  validation_dataset = dataset[train_end_index:validation_end_index]
  test_dataset = dataset[validation_end_index:]
  return train_dataset, validation_dataset, test_dataset

def write_dataset_to_csv(dataset, output_file_path):
  with open(output_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["file_path", "label"])
    csv_writer.writerows(dataset)

def generate_splits():
  dataset = get_image_paths_and_labels(RAW_DATA_DIRECTORY)
  shuffled_dataset = shuffle_dataset(dataset, RANDOM_SEED)
  
  total_items = len(shuffled_dataset)
  train_end_index, validation_end_index = calculate_split_indices(
    total_items, 
    TRAIN_RATIO, 
    VALIDATION_RATIO
  )
  
  train_dataset, validation_dataset, test_dataset = split_dataset(
    shuffled_dataset, 
    train_end_index, 
    validation_end_index
  )
  
  Path(SPLITS_DIRECTORY).mkdir(parents=True, exist_ok=True)
  
  write_dataset_to_csv(train_dataset, os.path.join(SPLITS_DIRECTORY, "train.csv"))
  write_dataset_to_csv(validation_dataset, os.path.join(SPLITS_DIRECTORY, "validation.csv"))
  write_dataset_to_csv(test_dataset, os.path.join(SPLITS_DIRECTORY, "test.csv"))

if __name__ == "__main__":
  generate_splits()
