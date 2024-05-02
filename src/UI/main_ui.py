import sys
from src.UI import page_2, page_3
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5 import QtGui
from src.UI.OpenCV import Worker

def runUI():
    app = QApplication(sys.argv)
    window = QMainWindow()
    stackedWidget = QStackedWidget()

    # Adding Page 2 to the stacked widget
    page2 = page_2.Page2()
    page2.setupUi(window)

    worker = Worker()
    worker.start()
    worker.ImageUpdate.connect(page2.imageUpdateSlot)
    stackedWidget.addWidget(page2.centralwidget)
    # page2.changeMode()

    # Adding Page 2 to the stacked widget
    page3 = page_3.Page3()
    page3.setupUi(window)
    stackedWidget.addWidget(page3.centralwidget)

    stackedWidget.show()

    app.exec_()

