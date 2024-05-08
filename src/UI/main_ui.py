import sys
from src.UI import page_1, page_2, page_3
from PyQt5.QtWidgets import QMainWindow, QStackedWidget,QLabel
from PyQt5 import QtGui
from src.UI.OpenCV import PersonCamFeedThread, AnimalCamFeedThread
from src.UI.datetimeupdater import TimeUpdateThread

class MainWindow(QMainWindow):
    def __init__(self, app, face_recognizer):
        super().__init__()
        self.centralWidget = QStackedWidget()
        self.app = app
        self.face_recognizer = face_recognizer
        self.personCamFeedThread = PersonCamFeedThread(self.face_recognizer)
        self.animalCamFeedThread = AnimalCamFeedThread()
        self.timeThread = TimeUpdateThread()
        self.timeThread.start()

        # Adding Page 1 to the stacked widget
        page1 = page_1.Page1()
        page1.setupUi(self)
        page1.proceedButton.clicked.connect(self.loadFirstPage)
        self.centralWidget.addWidget(page1.centralwidget)

        # Adding Page 2 to the stacked widget
        page2 = page_2.Page2()
        page2.setupUi(self)
        page2.applyButton.clicked.connect(lambda: self.modeChange(page2.personcomboBox.currentText(), page2.cameracomboBox.currentIndex()))
        page2.exitButton.clicked.connect(lambda: self.exitButtonClicked())
        page2.shootButton.clicked.connect(page2.shootUnidentifiedPerson)
        page2.authorisedLeftArrow.clicked.connect(page2.authLeftArrow)
        page2.authorisedRightArrow.clicked.connect(page2.authRightArrow)
        page2.unauthorisedLeftArrow.clicked.connect(page2.unauthLeftArrow)
        page2.unauthorisedRightArrow.clicked.connect(page2.unauthRightArrow)
        self.personCamFeedThread.ImageUpdate.connect(page2.imageUpdateSlot)
        self.personCamFeedThread.personDetails.connect(page2.personDetailsSlot)
        self.timeThread.DateUpdate.connect(page2.timeUpdater)
        self.centralWidget.addWidget(page2.centralwidget)
        
        # Adding Page 2 to the stacked widget
        page3 = page_3.Page3()
        page3.setupUi(self)
        self.animalCamFeedThread.ImageUpdate.connect(page3.imageUpdateSlot)
        page3.exitButton.clicked.connect(lambda: self.exitButtonClicked())
        page3.applyButton.clicked.connect(lambda: self.modeChange(page3.animalcomboBox.currentText(), page3.cameracomboBox.currentIndex()))
        page3.shootButton.clicked.connect(page3.shootAnimal)
        self.timeThread.DateUpdate.connect(page3.timeUpdater)
        self.centralWidget.addWidget(page3.centralwidget)

        self.centralWidget.show()

    # def closeEvent(self):

    def loadFirstPage(self):
        self.personCamFeedThread.start()
        self.centralWidget.setCurrentIndex(1)

    def modeChange(self, modeValue, cameraValue):
        if modeValue == "Animal":
            self.personCamFeedThread.stop()
            self.centralWidget.setCurrentIndex(2)
            self.animalCamFeedThread.start()
        else:
            self.animalCamFeedThread.stop()
            self.centralWidget.setCurrentIndex(1)
            self.personCamFeedThread.start()

        if cameraValue == 0:
            if (self.personCamFeedThread.isRunning()):
                self.personCamFeedThread.stop()
                self.personCamFeedThread.cameraIndex = 0
                self.animalCamFeedThread.cameraIndex = 0
                self.personCamFeedThread.start()
            else:
                self.animalCamFeedThread.stop()
                self.personCamFeedThread.cameraIndex = 0
                self.animalCamFeedThread.cameraIndex = 0
                self.animalCamFeedThread.start()
        else:
            if (self.personCamFeedThread.isRunning()):
                self.personCamFeedThread.stop()
                self.personCamFeedThread.cameraIndex = 2
                self.animalCamFeedThread.cameraIndex = 2
                self.personCamFeedThread.start()
            else:
                self.animalCamFeedThread.stop()
                self.personCamFeedThread.cameraIndex = 2
                self.animalCamFeedThread.cameraIndex = 2
                self.animalCamFeedThread.start() 

    def exitButtonClicked(self):
        # Stopping personCamFeedThread if it is active
        if (self.personCamFeedThread.isRunning()):
            self.personCamFeedThread.stop()
        
        # Stopping animalCamFeedThread if it is active
        if (self.animalCamFeedThread.isRunning()):
            self.animalCamFeedThread.stop()

        if (self.timeThread.isRunning()):
            self.timeThread.stop()
    
        self.app.closeAllWindows()
        self.app.exit()
