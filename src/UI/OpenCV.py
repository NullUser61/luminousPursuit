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

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# capture = cv.VideoCapture('videos/VID_20230820_181931749.mp4')

class Worker(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        capture = cv.VideoCapture(0)
        if (capture.isOpened() == False):
            print("Error opening the file")
        else:
            # i = 0

            while self.ThreadActive:
                isTrue, frame = capture.read()

                if (not isTrue):
                    print("Error")
                else:
                    frame=camera(frame)
                    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    flippedImage = cv.flip(image, 1)
                    QtFormattedImage = QImage(flippedImage.data, flippedImage.shape[1], flippedImage.shape[0], QImage.Format_RGB888) 
                    pic = QtFormattedImage.scaled(640, 480, Qt.KeepAspectRatio)

                    self.ImageUpdate.emit(pic)
                    # newFrame = rescaleFrame(frame, 0.75)
                    # cv.imshow('Video', newFrame)
                    # key = cv.waitKey(20)
            
                #     if key == ord('q'):
                #         break

                # i += 1

            capture.release()
        # cv.destroyAllWindows()
    
    def stop(self):
        self.ThreadActive = False
        self.quit()
# cv.waitKey(0)