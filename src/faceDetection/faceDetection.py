import cv2 as cv
import numpy as np
import time
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet

from src.faceDetection.centerface import CenterFace
from keras_facenet import FaceNet
# Initialize the serial port for communication with the laser module
# serialInst = serial.Serial("/dev/ttyUSB0", baudrate=9600)


def load_models():
    global  facenet, model, encoder , centerface

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

def laser_movement(face):
    x, y, w, h = face
    dotx = x + (w // 2)
    doty = y + (h // 2)
    xan = 34.0383 - 0.0770748 * doty
    yan = 122.18 - 0.0910204 * dotx
    xAngle = f"{xan},{xan},{yan}\n"
    serialInst.write(xAngle.encode('utf-8'))

def recognize_faces(faces,img):
    print("faces = " , faces)
    # print(faces[0])
    # print(type(faces))

    x, y, w, h = faces
    x, y, w, h = int(x), int(y), int(w), int(h)
    # print(x,y,w,h)

    img = img[y-2:y + h-y+2, x-2:x + w-x+2]
    # cv.imshow('Image', img)
    time.sleep(1)
    try:
        img = cv.resize(img, (160, 160))
    except:
        return "Unidentified"
    img = np.expand_dims(img, axis=0)
    ypred = facenet.embeddings(img)
    similarity_threshold = 2
    face_name = model.predict(ypred)
    decision_score = model.decision_function(ypred)
    print(decision_score)
    final_name = encoder.inverse_transform(face_name)[0] if decision_score > 0.7 else "Unidentified"
    return final_name

def camera(frame):
    # cap = cv.VideoCapture(0)
    # ret, frame = cap.read()
    # try:
    # except:
    #     camera(facenet, model, encoder, centerface)
    load_models()
    centerface=CenterFace()
    h, w = frame.shape[:2]
    while True:
    #     ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        # print("dets=",dets , "lms" , lms)
        for det in dets:
            boxes, score = det[:4], det[4]
            final_name = "random"
            # print("boxes=" , boxes)
            # if(Recognise==1):
            final_name = recognize_faces(boxes,frame)
            # else:
                # final_name=final_name="random"
            cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            print(final_name)
            # cv.circle(frame, (x + (w // 2), y + (h // 2)), radius=2, color=(255, 0, 0), thickness=2)
            # cv.putText(frame, str(final_name), (boxes[0], boxes[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

            if(final_name=="Unidentified"):
                cv.putText(frame, str(final_name), (int(boxes[0]), int(boxes[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 255), 3, cv.LINE_AA)
                cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 225), 1)
            else:
                cv.putText(frame, str(final_name), (int(boxes[0]), int(boxes[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 225, 0), 3, cv.LINE_AA)
                cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            # laser_movement(boxes)


            # laser_movement(x, y, w, h)
        for lm in lms:
            # laser_movement(lm[])
            for i in range(2, 3):
                cv.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
                # if(i==2):
                    # laser_movement()
        return frame
    #     cv.imshow('out', frame)
    #     # Press Q on keyboard to stop recording
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # cap.release()


def main():
    camera()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
