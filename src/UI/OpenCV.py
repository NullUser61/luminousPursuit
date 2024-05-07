import os
import cv2 as cv
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from src.faceDetection.faceDetection import camera
from src.faceDetection.centerface import CenterFace
import numpy

class AuthorizedPersonDetails:
    def __init__(self, name, image):
        self.name = name
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

def rescaleFrame(frame, width, height):
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

class PersonCamFeedThread(QThread):
    ImageUpdate = pyqtSignal(numpy.ndarray)
    personDetails = pyqtSignal(PersonDetails)
    def __init__(self):
        super().__init__()
        self.cameraIndex = 0
        self.ThreadActive = False
        self.capture = None

    def run(self):
        self.ThreadActive = True
        self.capture = cv.VideoCapture(self.cameraIndex)
        if (self.capture.isOpened() == False):
            print("Unable to open the camera")
        else:
            while self.ThreadActive:
                isTrue, frame = self.capture.read()

                if (not isTrue):
                    print("Capture Stream Closed")
                else:
                    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    flippedImage = cv.flip(image, 1)
                    flippedImage, faces, names , auth_count, unauth_count = camera(flippedImage)
                    
                    authorized = []
                    unauthorized = []

                    for i in range(0, len(faces)):
                        if (names[i] != "unidentified"):
                            person = AuthorizedPersonDetails(names[i], faces[i])
                            authorized.append(person)
                        else:
                            person = UnauthorizedPersonDetails(faces[i])
                            unauthorized.append(person)

                    persons = PersonDetails(authorized, unauthorized, len(authorized), len(unauthorized))

                    self.ImageUpdate.emit(flippedImage)
                    self.personDetails.emit(persons)
                    

    
    def stop(self):
        self.ThreadActive = False
        if (self.capture != None):
            self.capture.release()
        self.quit()


class AnimalCamFeedThread(QThread):
    ImageUpdate = pyqtSignal(numpy.ndarray)
    def __init__(self):
        super().__init__()
        self.cameraIndex = 0
        self.ThreadActive = False
        self.capture = None

    def run(self):
        self.ThreadActive = True
        self.capture = cv.VideoCapture(self.cameraIndex)
        if (self.capture.isOpened() == False):
            print("Unable to open the camera")
        else:
            while self.ThreadActive:
                isTrue, frame = self.capture.read()

                if (not isTrue):
                    print("Capture Stream Closed")
                else:
                    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    flippedImage = cv.flip(image, 1)

                    self.ImageUpdate.emit(flippedImage)
    
    def stop(self):
        self.ThreadActive = False
        if (self.capture != None):
            self.capture.release()
        self.quit()


# def convertToQtFormat(frame, width, height):
#     image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     flippedImage = cv.flip(image, 1)
#     QtFormattedImage = QImage(flippedImage.data, flippedImage.shape[1], flippedImage.shape[0], QImage.Format_RGB888) 
#     # pic = QtFormattedImage.scaled(width, height, Qt.KeepAspectRatio)
#     pic = QtFormattedImage.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio)

#     return pic

def convertToQtFormat(frame):
    QtFormattedImage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

    return QtFormattedImage