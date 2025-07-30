import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 32

def load_data(data_dir):
    images = []
    labels = []
    classes = os.listdir(data_dir)
    classes.sort()
    for label, class_folder in enumerate(classes):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path): continue
        for img_file in tqdm(os.listdir(class_path), desc=f"Loading {class_folder}"):
            try:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            except:
                continue
    X = np.array(images)
    y = np.array(labels)
    return X, y
