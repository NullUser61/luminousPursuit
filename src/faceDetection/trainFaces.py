import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

class DatasetCreation:
    def __init__(self, directory):
        self.directory = directory
        self.targetSize = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filepath):
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        sface = cv2.resize(face, self.targetSize)
        return sface

    def face_list(self):
        for label in os.listdir(self.directory):
            label_path = os.path.join(self.directory, label)
            for face in os.listdir(label_path):
                self.X.append(self.extract_face(os.path.join(label_path, face)))
                self.Y.append(label)
        return np.array(self.X), np.array(self.Y)

class FaceRecognition:
    def __init__(self, embedded_X, Y):
        self.embedded_X = embedded_X
        self.Y = Y
        self.encoder = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True)

    def train_model(self):
        self.encoder.fit(self.Y)
        encoded_Y = self.encoder.transform(self.Y)
        X_train, X_test, Y_train, Y_test = train_test_split(self.embedded_X, encoded_Y, shuffle=True, random_state=17)
        self.model.fit(X_train, Y_train)
        return X_train, X_test, Y_train, Y_test

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def evaluate_model(self, X_test, Y_test):
        ypreds_test = self.model.predict(X_test)
        return accuracy_score(Y_test, ypreds_test)

def get_embeddings(embedder, face_imgs):
    embeddings = []
    for img in face_imgs:
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)
        embedding = embedder.embeddings(img)
        embeddings.append(embedding[0])
    return np.asarray(embeddings)

def main():
    dataset = DatasetCreation("/content/drive/MyDrive/MyFaceDataset/")  # Take input
    X, Y = dataset.face_list()
    print("Face extraction done.")

    embedder = FaceNet()
    embedded_X = get_embeddings(embedder, X)
    np.savez_compressed("/content/drive/MyDrive/EmbeddedGS", embedded_X, Y) # Take input
    print("Embeddings saved.")

    recognizer = FaceRecognition(embedded_X, Y)
    X_train, X_test, Y_train, Y_test = recognizer.train_model()
    print("Model trained.")

    accuracy = recognizer.evaluate_model(X_test, Y_test)
    print("Model accuracy:", accuracy)

    recognizer.save_model('/content/drive/MyDrive/svm_model_S_G_160x160.pkl') # Take input
    print("Model saved.")

if __name__ == "__main__":
    main()
