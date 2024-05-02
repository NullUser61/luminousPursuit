import sys
from src.UI import page_1, page_2, page_3
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5 import QtGui
from src.UI.OpenCV import Worker

app = QApplication(sys.argv)
stackedWidget = QStackedWidget()

def runUI():
    window = QMainWindow()

    worker = Worker()
    worker.start()

    # Adding Page 1 to the stacked widget
    page1 = page_1.Page1()
    page1.setupUi(window)
    page1.proceedButton.clicked.connect(lambda: stackedWidget.setCurrentIndex(1))
    stackedWidget.addWidget(page1.centralwidget)

    # Adding Page 2 to the stacked widget
    page2 = page_2.Page2()
    page2.setupUi(window)
    page2.applyButton.clicked.connect(lambda: modeChange(page2.personcomboBox.currentText()))
    page2.exitButton.clicked.connect(lambda: worker.stop())
    worker.ImageUpdate.connect(page2.imageUpdateSlot)
    stackedWidget.addWidget(page2.centralwidget)

    # Adding Page 2 to the stacked widget
    page3 = page_3.Page3()
    page3.setupUi(window)
    worker.ImageUpdate.connect(page3.imageUpdateSlot)
    page3.exitButton.clicked.connect(lambda: worker.stop())
    stackedWidget.addWidget(page3.centralwidget)

    page3.applyButton.clicked.connect(lambda: modeChange(page3.animalcomboBox.currentText()))

    stackedWidget.show()

    sys.exit(app.exec_())

def modeChange(val):
    if val == "Animal":
        stackedWidget.setCurrentIndex(2)
    else:
        stackedWidget.setCurrentIndex(1)