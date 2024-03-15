import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow_gpu as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

def recognize_faces():
    # INITIALIZE
    facenet = FaceNet()
    faces_embeddings = np.load("../../data/Embedded.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    haarcascade = cv.CascadeClassifier("../../data/haarcascade_frontalface_default.xml")
    model = pickle.load(open("../../data/svm_model_160x160.pkl", 'rb'))

    cap = cv.VideoCapture(0)

    known_faces = []
    unknown_faces = []

    while cap.isOpened():
        _, frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv.resize(img, (160, 160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            similarity_threshold = 2
            # face_name = model.predict(ypred)
            decision_score = model.decision_function(ypred)
            x_pointer = x + (w // 2)
            y_pointer = y + (h // 2)
            # if decision_score[0][np.argmax(decision_score[0])] > 2.15:
            #     final_name = encoder.inverse_transform(face_name)[0]
            #     x_pointer = x+(w//2)
            #     y_pointer = y+(h//2)
            #     known_faces.append(final_name)
            #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
            #     cv.circle(frame, (x_pointer, y_pointer), radius=2, color=(255, 0, 0), thickness=2)
            #     cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
            #                1, (0, 0, 255), 3, cv.LINE_AA)
            # else:
            #     final_name = "Unidentified"
            #     unknown_faces.append(final_name)
            #     x_pointer = x + (w // 2)y_pointer
            #     y_pointer = y + (h // 2)
            #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            #     cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
            #                1, (0, 0, 255), 3, cv.LINE_AA)
            print("X:", x_pointer)
            print("Y:", y_pointer)
            # return x_pointer, y_pointer
            # cv.imshow("Face Recognition:", frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

    # cap.release()
    # cv.destroyAllWindows()

    # return x_pointer, y_pointer

if __name__ == "__main__":
    while 1:
        xPointer, yPointer = recognize_faces()
        print("X:", xPointer)
        print("Y:", yPointer)
