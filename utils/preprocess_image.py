import cv2
import numpy as np

def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size)) 
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img