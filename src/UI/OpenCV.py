import os
import cv2 as cv
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from src.faceDetection.faceDetection import camera
from src.faceDetection.centerface import CenterFace
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

# img = cv.imread('images/IMG_20230827_130752156.jpg')

# cv.imshow('Cat', img)

class AuthorizedPersonDetails:
    def __init__(self, name, age, image):
        self.name = name
        self.age = age
        self.image = image

class UnauthorizedPersonDetails:
    def __init__(self, image):
        self.image = image

class PersonDetails:
    def __init__(self, authorizedPersons, unauthorizedPersons, authorizedCount, unauthorizedCount):
        self.authorizedPersons = authorizedPersons
        self.unauthorizedPersons = unauthorizedPersons
        self.authorizedCount = authorizedCount
        self.unauthorizedCount = unauthorizedCount

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

class CamFeedThread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    personDetails = pyqtSignal(PersonDetails)
    def run(self):
        self.ThreadActive = True
        self.capture = cv.VideoCapture(0)
        if (self.capture.isOpened() == False):
            print("Error opening the file")
        else:
            while self.ThreadActive:
                isTrue, frame = self.capture.read()

                if (not isTrue):
                    print("Capture Stream Closed")
                else:
                    frame , faces, names , auth_count, unauth_count =camera(frame)
                    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    flippedImage = cv.flip(image, 1)
                    QtFormattedImage = QImage(flippedImage.data, flippedImage.shape[1], flippedImage.shape[0], QImage.Format_RGB888) 
                    pic = QtFormattedImage.scaled(640, 480, Qt.KeepAspectRatio)

                    authorized = []
                    authorized.append(AuthorizedPersonDetails("Sonu", 21, "Authorized"))

                    unauthorized = []
                    unauthorized.append(UnauthorizedPersonDetails("Unauthorized"))

                    persons = PersonDetails(authorized, unauthorized, 5, 3)

                    self.ImageUpdate.emit(pic)
                    self.personDetails.emit(persons)

    
    def stop(self):
        self.ThreadActive = False
        self.capture.release()
        self.quit()
