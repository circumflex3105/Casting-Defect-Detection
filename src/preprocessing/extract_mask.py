import os
import cv2
import numpy as np

PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
BLUR_KERNEL_SIZE = (5, 5)
MORPHOLOGY_KERNEL_SIZE = (5, 5)
MORPHOLOGY_ITERATIONS = 2

def get_all_image_paths(directory):
  image_paths = []
  for root, _, files in os.walk(directory):
    for file_name in files:
      if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_paths.append(os.path.join(root, file_name))
  return image_paths

def generate_otsu_mask(image):
  _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return binary_mask

def refine_mask_morphology(mask, kernel_size, iterations):
  kernel = np.ones(kernel_size, np.uint8)
  closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
  opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
  return opened_mask

def extract_mask():
  image_paths = get_all_image_paths(PREPROCESSED_DATA_DIRECTORY)
  
  for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, 0)
    raw_mask = generate_otsu_mask(blurred_image)
    refined_mask = refine_mask_morphology(raw_mask, MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_ITERATIONS)
    masked_image = cv2.bitwise_and(image, image, refined_mask)
    
    cv2.imwrite(image_path, masked_image)

if __name__ == "__main__":
  extract_mask()
