# face recognition part II
# IMPORT
import cv2 as cv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin'


from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("src/data/Embedded.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("src/data/haarcascade_frontalface_default.xml")
model = pickle.load(open("src/data/svm_model_160x160.pkl", 'rb'))

cap = cv.VideoCapture(0)
# WHILE LOOP

# while cap.isOpened():
#    _, frame = cap.read()

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
        face_name = model.predict(ypred)
        print(face_name)
        decision_score = model.decision_function(ypred)
        # face_name=np.argmax(decision_score[0])
        # print("\n",decision_score[0])
        print("\n", decision_score[0][face_name])
        if (decision_score[0][np.argmax(decision_score[0])] > 2.15):
            final_name = encoder.inverse_transform(face_name)[0]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)
        else:
            final_name = "Unidentified"
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows

# ypred

#        print("\n",decision_score[0][2])
#        if np.argmax(decision_score[0]):
#            final_name = encoder.inverse_transform(face_name)[0]
#            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 10)
#            cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
#                   1, (0,0,255), 3, cv.LINE_AA)
#        else:
#            final_name="Unidentified"
#            cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 10)
#            cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
#                   1, (0,0,255), 3, cv.LINE_AA)

#    cv.imshow("Face Recognition:", frame)
#    if cv.waitKey(1) & ord('q') ==27:
#        break

# cap.release()
# cv.destroyAllWindows