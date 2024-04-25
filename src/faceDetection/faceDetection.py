import cv2 as cv
import numpy as np
import os
import pickle
import serial
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from centerface import CenterFace
import time
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
    global detector, facenet, model, encoder, haarcascade
    # Initialize MTCNN detector
    detector = MTCNN()

    # Load FaceNet model
    facenet = FaceNet()

    # Load SVM model for face recognition
    model = pickle.load(open("../../data/svm_model_S_G_160x160.pkl", 'rb'))

    # Load label encoder for face recognition
    faces_embeddings = np.load("../../data/EmbeddedGS.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)

    # Load Haar cascade classifier for face detection
    haarcascade = cv.CascadeClassifier("../../data/haarcascade_frontalface_default.xml")

def laser_movement(x, y, w, h):
    dotx = x + (w // 2)
    doty = y + (h // 2)
    xan = 34.0383 - 0.0770748 * doty
    yan = 122.18 - 0.0910204 * dotx
    xAngle = f"{xan},{xan},{yan}\n"
    serialInst.write(xAngle.encode('utf-8'))

def recognize_faces(faces,img):
    # print("faces = " , faces)
    # print(faces[0])
    # print(type(faces))

    x, y, w, h = faces
    x, y, w, h = int(x), int(y), int(w), int(h)
    # print(x,y,w,h)

    img = img[y-2:y + h-y+2, x-2:x + w-x+2]
    # cv.imshow('Image', img)
    time.sleep(1)
    img = cv.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    ypred = facenet.embeddings(img)
    similarity_threshold = 2
    face_name = model.predict(ypred)
    decision_score = model.decision_function(ypred)
    final_name = encoder.inverse_transform(face_name)[0] if decision_score > 0.7 else "Unidentified"
    return final_name

def camera():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    while True:
        ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        # print("dets=",dets , "lms" , lms)
        for det in dets:
            boxes, score = det[:4], det[4]
            # print("boxes=" , boxes)
            final_name = recognize_faces(boxes,frame)
            cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            print(final_name)
            # cv.circle(frame, (x + (w // 2), y + (h // 2)), radius=2, color=(255, 0, 0), thickness=2)
            # cv.putText(frame, str(final_name), (boxes[0], boxes[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(frame, str(final_name), (int(boxes[0]), int(boxes[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)


            # laser_movement(x, y, w, h)
        for lm in lms:
            for i in range(0, 5):
                cv.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        cv.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def main():
    initialize()
    load_models()
    camera()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
