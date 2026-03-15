import os
import cv2
from pathlib import Path

RAW_DATA_DIRECTORY = "data/raw"
PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
VISUALIZATION_DATA_DIRECTORY = "data/visualization"

def get_relative_image_paths(base_directory):
  relative_paths = []
  for root, _, files in os.walk(base_directory):
    for file_name in files:
      if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(root, file_name)
        relative_paths.append(os.path.relpath(full_path, base_directory))
  return relative_paths

def concatenate_images_horizontally(left_image_path, right_image_path):
  left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
  right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
  
  if left_image is None or right_image is None:
    return None
    
  return cv2.hconcat([left_image, right_image])

def save_comparison_image(image, output_path):
  output_directory = os.path.dirname(output_path)
  Path(output_directory).mkdir(parents=True, exist_ok=True)
  cv2.imwrite(output_path, image)

def generate_comparisons():
  relative_paths = get_relative_image_paths(RAW_DATA_DIRECTORY)
  
  for relative_path in relative_paths:
    raw_path = os.path.join(RAW_DATA_DIRECTORY, relative_path)
    preprocessed_path = os.path.join(PREPROCESSED_DATA_DIRECTORY, relative_path)
    comparison_path = os.path.join(VISUALIZATION_DATA_DIRECTORY, relative_path)
    
    if os.path.exists(preprocessed_path):
      comparison_image = concatenate_images_horizontally(raw_path, preprocessed_path)
      if comparison_image is not None:
        save_comparison_image(comparison_image, comparison_path)

if __name__ == "__main__":
  generate_comparisons()
