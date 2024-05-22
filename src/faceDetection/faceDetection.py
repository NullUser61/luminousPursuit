import cv2 as cv
import numpy as np
import time
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
# from centerface import CenterFace
from src.faceDetection.centerface import CenterFace
from keras_facenet import FaceNet
# Initialize the serial port for communication with the laser module
import serial.tools.list_ports
try:
    serialInst = serial.Serial("/dev/ttyUSB0", baudrate=9600)
except:
    print("Unable to open port")
# label_encoder = LabelEncoder()
# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('data/faceDetection/face_trained.yml')
# labels = np.load('data/faceDetection/Facelabels.npy')
# features = np.load('data/faceDetection/Facefeatures.npy', allow_pickle=True)

def load_models():
    global  facenet, model, encoder , centerface

    # Load FaceNet model
    facenet = FaceNet()

    # Load SVM model for face recognition
    model = pickle.load(open("data/faceDetection/svm_model_S_G_160x160.pkl", 'rb'))

    # Load label encoder for face recognition
    faces_embeddings = np.load("data/faceDetection/EmbeddedGS.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)

    centerface = CenterFace()

def laser_movement(face):
    x, y, w, h = face
    dotx = x + (w // 2)
    doty = y + (h // 2)
    xan = 25.4131 - 0.0790737 * doty
    yan = 116.849 - 0.0900342 * dotx
    yan=(yan-180)*-1
    yan=yan+10
    xAngle = f"{xan},{xan},{yan}\n"
    try:
        serialInst.write(xAngle.encode('utf-8'))
    except:
        print("Unable to write to serial output")
def laser_movement2(dotx,doty):
    # x, y, w, h = face
    xan = 25.4131 - 0.0790737 * doty
    yan = 116.849 - 0.0900342 * dotx
    yan=(yan-180)*-1
    xan=xan-3
    yan=yan-2
    # yan=yan-22
    xAngle = f"{xan},{xan},{yan}\n"
    try:
        serialInst.write(xAngle.encode('utf-8'))
    except:
        print("Unable to write to serial output")


def recognize_faces(boxes, frame, faces, names,face_recognizer):
    x, y, w, h = boxes
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = frame[y:h, x:w]  # Extract ROI with a margin of 2 pixels
    # cv.imshow('Face ROI', img)  # Display the face ROI
    # cv.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv.destroyAllWindows()  # Close all OpenCV windows

    faces.append(img)
    # cv.imshow("face",img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Ensure that the face ROI dimensions are valid
    if gray.shape[0] > 0 and gray.shape[1] > 0:
        faces_roi = cv.resize(gray, (100, 100))  # Resize the face ROI if necessary
        # Debugging output to inspect input data
        # print("Input data shape:", faces_roi.shape)
        # print("Input data dtype:", faces_roi.dtype)
        label, confidence = face_recognizer.predict(faces_roi)
        # print(confidence)
        # time.sleep(3)
        if(confidence>105):
            if(label==1):
                return "Sonu"
            else:
                return "Gautham"
        else:
            return "unidentified"
    else:
        print("Invalid face ROI dimensions")
        return None




def recognize_faces_facenet(boxes,img, faces, names):
    print("faces = " , boxes)
    # print(faces[0])
    # print(type(faces))

    x, y, w, h = boxes
    x, y, w, h = int(x), int(y), int(w), int(h)
    # print(x,y,w,h)

    img = img[y-2:y + h-y+2, x-2:x + w-x+2]
    faces.append(img)
    # cv.imshow('Image', img)
    # time.sleep(1)
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
    final_name = encoder.inverse_transform(face_name)[0] if decision_score > 0.7 else "unidentified"
    if final_name==1:
        final_name="Sonu"
    else:
        final_name="Gautham"
    # final_name = name[final_name]
    names.append(final_name)
    print(final_name)
    return final_name

def camera(frame, face_recognizer):
    # cap = cv.VideoCapture(0)
    # ret, frame = cap.read()
    # try:
    # except:
    #     camera(facenet, model, encoder, centerface)
    # load_models()
    centerface=CenterFace()
    h, w = frame.shape[:2]
    # face_recognizer = cv.face.LBPHFaceRecognizer_create()
    # face_recognizer.read('data/faceDetection/face_trained.yml')
    # labels = np.load('data/faceDetection/Facelabels.npy')
    # labels = label_encoder.inverse_transform(labels)
    # print("Reversed labels:", labels)
    while True:
    #     ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        unauth_count = 0
        auth_count = 0
        names=[]
        faces=[]
        global boxes, pointer, dotx,doty
        # print("dets=",dets , "lms" , lms)
        for det in dets:
            boxes, score = det[:4], det[4]
            final_name = "random"
            # print("boxes=" , boxes)
            # if(Recognise==1):
            final_name = recognize_faces(boxes,frame,faces,names,face_recognizer)

            # else:
                # final_name=final_name="random"
            cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            # print(final_name)
            # cv.circle(frame, (x + (w // 2), y + (h // 2)), radius=2, color=(255, 0, 0), thickness=2)
            # cv.putText(frame, str(final_name), (boxes[0], boxes[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

            if(final_name=="Unidentified"):
                cv.putText(frame, str(final_name), (int(boxes[0]), int(boxes[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 255), 3, cv.LINE_AA)
                cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 225), 1)
                unauth_count+= 1

            else:
                cv.putText(frame, str(final_name), (int(boxes[0]), int(boxes[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 225, 0), 3, cv.LINE_AA)
                cv.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
                auth_count+= 1
                names.append(final_name)
        # laser_movement(boxes)
        try:
            laser_movement2(lms[0][2])
        except:
            print()

            # laser_movement(x, y, w, h)
        for lm in lms:
            # laser_movement(lm[])
            for i in range(2, 3):
                cv.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)

                # laser_movement2(int(lm[i * 2]),int(lm[i * 2 + 1]))
                # if(i==2):
                    # laser_movement()
                dotx=int(lm[i * 2])
                doty=int(lm[i * 2 + 1])
                laser_movement2(dotx,doty)
        return frame , faces, names , auth_count, unauth_count
    #     cv.imshow('out', frame)
    #     # Press Q on keyboard to stop recording
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # cap.release()


def main():
    cap = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Loop to capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        camera(frame)


if __name__ == "__main__":
    main()
