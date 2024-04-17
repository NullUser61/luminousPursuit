
# IMPORT
import cv2 as cv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify the GPU device index

import tensorflow as tf
import time
# Explicitly allow memory growth, which can help to reduce memory fragmentation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import serial.tools.list_ports
from mtcnn.mtcnn import MTCNN
detector=MTCNN()
# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("../../data/Embedded.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
# haarcascade = cv.CascadeClassifier("../../data/haarcascade_frontalface_default.xml")
model = pickle.load(open("../../data/svm_model_160x160.pkl", 'rb'))

cap = cv.VideoCapture(0)
# WHILE LOOP

# while cap.isOpened():
#    _, frame = cap.read()
# serialInst = serial.Serial("/dev/ttyUSB0", baudrate=9600)
# xan=120
# yan=90
# xAngle = f"{xan},{xan},{yan}\n"
# serialInst.write(xAngle.encode('utf-8'))

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector.detect_faces(rgb_img)[0]['box']
    print(faces)
    # faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        # ypred = facenet.embeddings(img)
        # similarity_threshold = 2.1
        # face_name = model.predict(ypred)
        # print(face_name)
        # decision_score = model.decision_function(ypred)
        # face_name=np.argmax(decision_score[0])
        # print("\n",decision_score[0])
        # print("\n", decision_score[0][face_name])
        # if (decision_score[0][np.argmax(decision_score[0])] > 2.1):
        #     final_name = encoder.inverse_transform(face_name)[0]
        #     print(x,y)
        dotx=x+(w//2)
        doty=y+(h//2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
        cv.circle(frame, (x+(w//2), y+(h//2)), radius=2, color=(255, 0, 0), thickness=2)
        # print("x=",x+(w//2),"y=",y+(h//2))
        # time.sleep(0.5)
        # xan=0.0703125*dotx+15
        xan=34.0383-0.0770748*doty
        # xan=xan//1
        # xan=180-xan
        # yan=100+doty*0.132609
        yan=122.18-0.0910204*dotx
        # print(xan)
        # yan=90
        # x=int(n)
        xAngle = f"{xan},{xan},{yan}\n"
        # serialInst.write(xAngle.encode('utf-8'))
        # cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
        #                1, (0, 0, 255), 3, cv.LINE_AA)
        # else:
            # final_name = "Unidentified"
        # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
        # cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
        #                1, (0, 0, 255), 3, cv.LINE_AA)
    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows

