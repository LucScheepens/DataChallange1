# import cv2
# from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import time
import numpy as np
#
def apply_gaussian_blur(lst, sigma = 0.5):
  '''
  This is a function that applies gaussian blur to a list of images. To specify
  the amount of noise, use sigma. A new list with gaussian blur is returned

  :param lst: The list to convert
  :type lst: list
  :param sigma: The variance for the gaussian noise
  :type sigma: float
  '''

  

  print('Adding gaussian noise')
  tic = time.perf_counter()
  transformed_lst = []

  for i in lst:
    image_lst = i.tolist()
    transformed_lst.append(gaussian_filter(image_lst, sigma=sigma))
  
  toc = time.perf_counter()
  print(f"All images processed in {toc-tic:0.2f} seconds.")

  return np.array(transformed_lst)

def grayscale_arr(arr):
  gray = []
  for i in range(len(arr)):
    img = np.full((128,128,3), 12, np.uint8)
    gray.append(rgb2gray(arr[i]))
  return np.array(gray)

def gray_image(img):

  gray = rgb2gray(img)
  return gray

def preprocess_data(arr):
  X_train_blur = apply_gaussian_blur(arr)
  # X_train_blur = grayscale_df(X_train_blur)
  return X_train_gray_blur

