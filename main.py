import cv2 as cv
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from src.faceDetection.faceDetection import camera
from src.faceDetection.centerface import CenterFace
from src.faceDetection.faceDetection import recognize_faces
import time
from src.UI.main_ui import runUI
# Initialize the serial port for communication with the laser module
# serialInst = serial.Serial("/dev/ttyUSB0", baudrate=9600)


def initialize():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify the GPU device index
    # Explicitly allow memory growth, which can help to reduce memory fragmentation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def load_models():
    global  facenet, model, encoder, haarcascade , centerface

    # Load FaceNet model
    facenet = FaceNet()

    # Load SVM model for face recognition
    model = pickle.load(open("data/svm_model_S_G_160x160.pkl", 'rb'))

    # Load label encoder for face recognition
    faces_embeddings = np.load("data/EmbeddedGS.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)

    centerface = CenterFace()
    # Load Haar cascade classifier for face detection
    # haarcascade = cv.CascadeClassifier("../../data/haarcascade_frontalface_default.xml")

def main():
    # initialize()
    # load_models()
    # camera( facenet, model, encoder, centerface, Recognise=1)
    # recognize_faces()
    runUI()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
