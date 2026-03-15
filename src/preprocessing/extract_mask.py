import os
import cv2
import numpy as np

PREPROCESSED_DATA_DIRECTORY = "data/preprocessed"
BLUR_KERNEL_SIZE = (7, 7)
CANNY_THRESHOLD_LOWER = 20
CANNY_THRESHOLD_UPPER = 80
DILATION_KERNEL_SIZE = (11, 11)
DILATION_ITERATIONS = 3
WHITE_PIXEL_VALUE = 255
MINIMUM_POINTS_FOR_ELLIPSE = 5
ELLIPSE_MARGIN_OFFSET = 10

def get_all_image_paths(directory):
  image_paths = []
  for root, _, files in os.walk(directory):
    for file_name in files:
      if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_paths.append(os.path.join(root, file_name))
  return image_paths

def generate_elliptical_mask(image):
  blurred_image = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, 0)
  edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_LOWER, CANNY_THRESHOLD_UPPER)
  
  dilation_kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
  dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=DILATION_ITERATIONS)
  
  contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  silhouette_mask = np.zeros_like(image)
  if contours:
    largest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, closed=False))
    if len(largest_contour) >= MINIMUM_POINTS_FOR_ELLIPSE:
      center, axes, angle = cv2.fitEllipse(largest_contour)
      
      adjusted_axes = (
        max(0.0, axes[0] - ELLIPSE_MARGIN_OFFSET),
        max(0.0, axes[1] - ELLIPSE_MARGIN_OFFSET)
      )
      
      adjusted_ellipse = (center, adjusted_axes, angle)
      cv2.ellipse(silhouette_mask, adjusted_ellipse, WHITE_PIXEL_VALUE, thickness=cv2.FILLED)
      
  return silhouette_mask

def extract_mask():
  image_paths = get_all_image_paths(PREPROCESSED_DATA_DIRECTORY)
  
  for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      continue
      
    elliptical_mask = generate_elliptical_mask(image)
    masked_image = cv2.bitwise_and(image, image, mask=elliptical_mask)
    
    cv2.imwrite(image_path, masked_image)

if __name__ == "__main__":
  extract_mask()
