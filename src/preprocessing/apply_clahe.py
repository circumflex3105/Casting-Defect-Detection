import os
import cv2
from pathlib import Path

RAW_DATA_DIRECTORY = "data/raw"
PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

def get_all_image_paths(directory):
  image_paths = []
  for root, _, files in os.walk(directory):
    for file_name in files:
      if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_paths.append(os.path.join(root, file_name))
  return image_paths

def read_image_grayscale(image_path):
  return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def create_clahe_object(clip_limit, tile_grid_size):
  return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

def apply_clahe_to_image(image, clahe_object):
  return clahe_object.apply(image)

def construct_output_path(input_path, raw_directory, preprocessed_directory):
  relative_path = os.path.relpath(input_path, raw_directory)
  return os.path.join(preprocessed_directory, relative_path)

def ensure_directory_exists(file_path):
  directory = os.path.dirname(file_path)
  Path(directory).mkdir(parents=True, exist_ok=True)

def save_image(image, output_path):
  ensure_directory_exists(output_path)
  cv2.imwrite(output_path, image)

def apply_clahe():
  image_paths = get_all_image_paths(RAW_DATA_DIRECTORY)
  clahe_object = create_clahe_object(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
  
  for image_path in image_paths:
    image = read_image_grayscale(image_path)
    enhanced_image = apply_clahe_to_image(image, clahe_object)
    
    output_path = construct_output_path(
      image_path, 
      RAW_DATA_DIRECTORY, 
      PREPROCESSED_DATA_DIRECTORY
    )
    
    save_image(enhanced_image, output_path)

if __name__ == "__main__":
  apply_clahe()
