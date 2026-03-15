import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION = [0.229, 0.224, 0.225]
LABEL_MAPPING = {"ok_front": 0, "def_front": 1}
DATALOADER_WORKER_COUNT = 4

def read_dataset_from_csv(csv_file_path):
  dataset_items = []
  with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
      if len(row) == 2:
        dataset_items.append((row[0], row[1]))
  return dataset_items

def extract_relative_image_path(original_file_path):
  path_components = os.path.normpath(original_file_path).split(os.sep)
  class_directory = path_components[-2]
  file_name = path_components[-1]
  return os.path.join(class_directory, file_name)

class CastingDataset(Dataset):
  def __init__(self, csv_file_path, base_image_directory, image_transform):
    self.dataset_items = read_dataset_from_csv(csv_file_path)
    self.base_image_directory = base_image_directory
    self.image_transform = image_transform

  def __len__(self):
    return len(self.dataset_items)

  def __getitem__(self, index):
    original_csv_path, label_string = self.dataset_items[index]
    
    relative_path = extract_relative_image_path(original_csv_path)
    full_image_path = os.path.join(self.base_image_directory, relative_path)
    
    image = Image.open(full_image_path).convert("RGB")
    transformed_image = self.image_transform(image)
    
    numeric_label = LABEL_MAPPING[label_string]
    tensor_label = torch.tensor(numeric_label, dtype=torch.long)
    
    return transformed_image, tensor_label

def create_image_transform():
  return transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STANDARD_DEVIATION)
  ])

def create_data_loader(csv_file_path, base_image_directory, batch_size, is_training):
  image_transform = create_image_transform()
  dataset = CastingDataset(csv_file_path, base_image_directory, image_transform)
  
  return DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=is_training,
    num_workers=DATALOADER_WORKER_COUNT,
    pin_memory=True
  )
