import os
import cv2
import shutil
import numpy as np
from src.faceDetection.centerface import CenterFace


features = []
labels = []

def copy_files(source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over files and directories in the source directory
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join(dest_dir, item)

        # If item is a file, copy it to the destination directory
        if os.path.isfile(source_item):
            with open(source_item, 'rb') as src_file:
                with open(dest_item, 'wb') as dest_file:
                    shutil.copyfileobj(src_file, dest_file)
            print(f"Copied file: {source_item} to {dest_item}")
        # If item is a directory, recursively copy its contents
        elif os.path.isdir(source_item):
            copy_files(source_item, dest_item)
            print(f"Copied directory: {source_item} to {dest_item}")

def read_images_from_folders(path):
    labels = []  # List to store folder names (labels)
    images = []  # List to store images

    # Iterate over each folder in the given path
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        # Check if the item in the path is a directory
        if os.path.isdir(folder_path):
            labels.append(folder_name)  # Add folder name as label

            # Iterate over each image file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)  # Read image using OpenCV
                    frame = cv2.imread('000388.jpg')
                    h, w = frame.shape[:2]
                    landmarks = True
                    centerface = CenterFace(landmarks=landmarks)
                    if landmarks:
                        dets, lms = centerface(frame, h, w, threshold=0.35)
                    else:
                        dets = centerface(frame, threshold=0.35)

                    for det in dets:
                        boxes, score = det[:4], det[4]
                        for (x, y, w, h) in boxes:
                            faces = frame[y:y + h, x:x + w]
                            gray = cv2.cvtColor(faces_roi, cv2.COLOR_BGR2GRAY)
                            features.append(gray)
                            labels.append(folder_name)
                    if image is not None:
                        images.append(image)  # Add image to list


    return labels, images




if __name__ == "__main__":
    # Input source and destination directories
    source_dir = input("Enter the source directory path: ")
    dest_dir = "/home/nulluser61/majorProject/data/faceDataset"

    # Copy files from source to destination
    copy_files(source_dir, dest_dir)
    read_images_from_folders(dest_dir)
    features = np.array(features, dtype='object')
    labels = np.array(labels)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features, labels)

    face_recognizer.save('face_trained.yml')
    np.save('../../data/faceDetection/Facefeatures.npy', features)
    np.save('../../data/faceDetection/Facelabels.npy', labels)








