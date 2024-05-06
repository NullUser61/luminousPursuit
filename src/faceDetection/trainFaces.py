import os
import cv2
import numpy as np
import shutil
from sklearn.preprocessing import LabelEncoder
from centerface import CenterFace

# Function to read images from folders and extract features
def read_images_from_folders(path):
    labels = []  # List to store folder names (labels)
    features = []  # List to store features
    label_encoder = LabelEncoder()  # Initialize LabelEncoder

    # Iterate over each folder in the given path
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        # Check if the item in the path is a directory
        if os.path.isdir(folder_path):
            # labels.append(folder_name)  # Add folder name as label
            # print(folder_name)
            # print(labels)

            # Iterate over each image file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)  # Read image using OpenCV
                    print(image_path)
                    # cv2.imshow("img",image)

                    # Extract face features
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        features.append(gray)  # Flatten image and add to features list
                        labels.append(folder_name)
                    else:
                        labels.pop()

    # Convert labels to integer IDs using LabelEncoder
    integer_labels = label_encoder.fit_transform(labels)

    return integer_labels, np.array(features, dtype='object')    # Return integer labels and features array

# Function to copy files from source directory to destination directory
def copy_files(source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over files and directories in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # Check if the item in the path is a directory
        if os.path.isdir(folder_path):
            # Create destination folder in the destination directory
            dest_folder_path = os.path.join(dest_dir, folder_name)
            os.makedirs(dest_folder_path, exist_ok=True)

            # Iterate over each image file in the folder
            for filename in os.listdir(folder_path):
                # Copy image file from source directory to destination directory
                source_image_path = os.path.join(folder_path, filename)
                dest_image_path = os.path.join(dest_folder_path, filename)
                shutil.copyfile(source_image_path, dest_image_path)
                print(f"Copied file: {source_image_path} to {dest_image_path}")

if __name__ == "__main__":
    # Input source directory path
    # source_dir = input("Enter the source directory path: ")
    source_dir = "/home/krizz/Downloads/MyFaceDataset/"
    dest_dir = "data/faceDetection/faceDataset"

    # Copy files from source directory to destination directory
    copy_files(source_dir, dest_dir)

    # Read images from folders in destination directory and extract features
    labels, features = read_images_from_folders(dest_dir)

    # Ensure that the number of samples (features) and labels match
    # if len(features) != len(labels):
    #     print("Error: Number of samples does not match the number of labels.")
    #     exit(1)

    # Train the Recognizer
    print(labels, features)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    # Save the trained recognizer
    face_recognizer.save('data/faceDetection/face_trained.yml/')
    np.save

    # Save features and labels
    np.save('data/faceDetection/Facefeatures.npy', features)
    np.save('data/faceDetection/Facelabels.npy', labels)
