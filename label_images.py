import os
import shutil
import cv2
import csv
import numpy as np
from tensorflow.keras.models import load_model
from utils.fix_old_keras_model import fix_old_keras_model
from utils.preprocess_image import preprocess_image

model_path = "models/keras_model.h5" # CHANGE TO YOUR MODEL
labels_path = "models/labels.txt" # CHANGE TO YOUR LABELS FILE
image_folder = "images/"
output_folder = "output/"
image_size = 224
predictions_array = []

np.set_printoptions(suppress=True)
fix_old_keras_model(model_path) # This does not do anything if the model works out of the box
model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").readlines()

for class_name in class_names:
    os.makedirs(f"{output_folder}/{class_name}", exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")): 
        image_path = os.path.join(image_folder, filename)
        print(f"Now predicting image {image_path}")
        img = preprocess_image(image_path, image_size)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        prediction_index = predicted_class[0]
        class_name = class_names[prediction_index]
        print(f"{filename}: Predicted Class - {prediction_index} - {class_name}")
        predictions_array.append({
            "file": filename,
            "prediction": class_name
        })
        shutil.copy(image_path, f"{output_folder}/{class_name}/{filename}")


with open(f"{output_folder}output.csv", "w") as csv_output:
    csv_writer = csv.writer(csv_output)
    csv_writer.writerow(["file_name", "prediction"])
    for item in predictions_array:
        csv_writer.writerow([item["file"], item["prediction"]])
