import sys
from src.UI import page_1, page_2, page_3
from PyQt5.QtWidgets import QMainWindow, QStackedWidget,QLabel
from PyQt5 import QtGui
from src.UI.OpenCV import CamFeedThread

class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.centralWidget = QStackedWidget()
        self.app = app
    
        self.camFeedThread = CamFeedThread()

        # Adding Page 1 to the stacked widget
        page1 = page_1.Page1()
        page1.setupUi(self)
        page1.proceedButton.clicked.connect(self.loadFirstPage)
        self.centralWidget.addWidget(page1.centralwidget)

        # Adding Page 2 to the stacked widget
        page2 = page_2.Page2()
        page2.setupUi(self)
        page2.applyButton.clicked.connect(lambda: self.modeChange(page2.personcomboBox.currentText()))
        page2.exitButton.clicked.connect(lambda: self.exitButtonClicked(page2.cameraFeed))
        self.camFeedThread.ImageUpdate.connect(page2.imageUpdateSlot)
        self.camFeedThread.personDetails.connect(page2.personDetailsSlot)
        self.centralWidget.addWidget(page2.centralwidget)
        
        # Adding Page 2 to the stacked widget
        page3 = page_3.Page3()
        page3.setupUi(self)
        self.camFeedThread.ImageUpdate.connect(page3.imageUpdateSlot)
        page3.exitButton.clicked.connect(lambda: self.exitButtonClicked(page3.cameraFeed))
        self.centralWidget.addWidget(page3.centralwidget)
        page3.applyButton.clicked.connect(lambda: self.modeChange(page3.animalcomboBox.currentText()))

        self.centralWidget.show()

    def loadFirstPage(self):
        self.camFeedThread.start()
        self.centralWidget.setCurrentIndex(1)

    def modeChange(self, val):
        if val == "Animal":
            self.centralWidget.setCurrentIndex(2)
        else:
            self.centralWidget.setCurrentIndex(1)

    def exitButtonClicked(self, camFeed):
        self.camFeedThread.stop()
        self.app.closeAllWindows()