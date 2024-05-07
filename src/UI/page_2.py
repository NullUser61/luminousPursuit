# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'page2.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from src.UI.OpenCV import convertToQtFormat

class Page2(object):
    def setupUi(self, MainWindow):
        self.authorizedPersonIndex = 0
        self.unauthorizedPersonIndex = 0
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1236, 814)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_17 = QtWidgets.QFrame(self.centralwidget)
        self.frame_17.setStyleSheet("background-color: rgb(18, 30, 52);")
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_17)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame = QtWidgets.QFrame(self.frame_17)
        self.frame.setStyleSheet("background: QLinearGradient(\n"
"    x1: 0, y1: 0,\n"
"    x2: 1, y2: 0, \n"
"\n"
"    stop: 0 #111727, \n"
"    stop: 0.5 #4B5B98,\n"
"    stop: 1 #111727\n"
");")
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setStyleSheet("background: none;")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/Icons/Icons/lup.jpg"))
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.homeButton = QtWidgets.QPushButton(self.frame)
        self.homeButton.setStyleSheet("\n"
"background-color: rgb(50, 64, 111);")
        self.homeButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Icons/Icons/🦆 icon _home_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.homeButton.setIcon(icon)
        self.homeButton.setIconSize(QtCore.QSize(25, 25))
        self.homeButton.setObjectName("homeButton")
        self.horizontalLayout.addWidget(self.homeButton)
        self.trainingbutton = QtWidgets.QPushButton(self.frame)
        self.trainingbutton.setStyleSheet("\n"
"background-color: rgb(50, 64, 111);")
        self.trainingbutton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Icons/Icons/🦆 icon _film_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.trainingbutton.setIcon(icon1)
        self.trainingbutton.setIconSize(QtCore.QSize(25, 25))
        self.trainingbutton.setObjectName("trainingbutton")
        self.horizontalLayout.addWidget(self.trainingbutton)
        self.unauthorisedHistory = QtWidgets.QPushButton(self.frame)
        self.unauthorisedHistory.setStyleSheet("background-color: rgb(50, 64, 111);")
        self.unauthorisedHistory.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Icons/Icons/history 1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.unauthorisedHistory.setIcon(icon2)
        self.unauthorisedHistory.setIconSize(QtCore.QSize(25, 25))
        self.unauthorisedHistory.setObjectName("unauthorisedHistory")
        self.horizontalLayout.addWidget(self.unauthorisedHistory)
        self.exitButton = QtWidgets.QPushButton(self.frame)
        self.exitButton.setStyleSheet("background-color: rgb(50, 64, 111);")
        self.exitButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Icons/Icons/🦆 icon _log in_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exitButton.setIcon(icon3)
        self.exitButton.setIconSize(QtCore.QSize(25, 25))
        self.exitButton.setObjectName("exitButton")
        self.horizontalLayout.addWidget(self.exitButton)
        self.verticalLayout_5.addWidget(self.frame)
        self.frame_16 = QtWidgets.QFrame(self.frame_17)
        self.frame_16.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_16.setObjectName("frame_16")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame_16)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.frame_9 = QtWidgets.QFrame(self.frame_16)
        self.frame_9.setStyleSheet("background: QLinearGradient(\n"
"    x1: 0, y1: 0,\n"
"    x2: 1, y2: 1, \n"
"    stop: 0 #414F80, \n"
"    stop: 0.57 #262F4F, \n"
"    stop: 1 #1E2541\n"
");\n"
"\n"
"border-radius: 25px;")
        self.frame_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout.setContentsMargins(20, 20, 20, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame_9)
        self.frame_3.setStyleSheet("background-color: rgb(79, 96, 155);\n"
"border-radius: 15px;")
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 11)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(150, 0))
        self.frame_6.setStyleSheet("background-color: rgb(79, 96, 155);\n"
"border:none;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_8 = QtWidgets.QLabel(self.frame_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setMaximumSize(QtCore.QSize(30, 30))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/Icons/Icons/security-camera(1) 1.png"))
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.cameracomboBox = QtWidgets.QComboBox(self.frame_6)
        self.cameracomboBox.setStyleSheet("background-color: rgb(79, 96, 155);\n"
"\n"
"font: 9pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);\n"
"")
        self.cameracomboBox.setFrame(False)
        self.cameracomboBox.setObjectName("cameracomboBox")
        self.cameracomboBox.addItem("")
        self.cameracomboBox.addItem("")
        self.horizontalLayout_4.addWidget(self.cameracomboBox)
        self.horizontalLayout_2.addWidget(self.frame_6)
        spacerItem1 = QtWidgets.QSpacerItem(235, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.timeStamp = QtWidgets.QLabel(self.frame_3)
        self.timeStamp.setStyleSheet("color: rgb(255, 255, 255);")
        self.timeStamp.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.timeStamp.setObjectName("timeStamp")
        self.horizontalLayout_2.addWidget(self.timeStamp)
        self.horizontalLayout_2.setStretch(0, 5)
        self.horizontalLayout_2.setStretch(1, 10)
        self.horizontalLayout_2.setStretch(2, 14)
        self.verticalLayout.addWidget(self.frame_3)
        self.frame_2 = QtWidgets.QFrame(self.frame_9)
        self.frame_2.setStyleSheet("background-color: none;")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setObjectName("gridLayout")
        self.cameraFeed = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(16)
        sizePolicy.setVerticalStretch(9)
        sizePolicy.setHeightForWidth(self.cameraFeed.sizePolicy().hasHeightForWidth())
        self.cameraFeed.setSizePolicy(sizePolicy)
        self.cameraFeed.setMinimumSize(QtCore.QSize(800, 450))
        self.cameraFeed.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-radius: 25px;")
        self.cameraFeed.setAlignment(QtCore.Qt.AlignCenter)
        self.cameraFeed.setObjectName("cameraFeed")
        self.gridLayout.addWidget(self.cameraFeed, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_7 = QtWidgets.QFrame(self.frame_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setStyleSheet("background-color: none;")
        self.frame_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_8 = QtWidgets.QFrame(self.frame_7)
        self.frame_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.frame_5 = QtWidgets.QFrame(self.frame_8)
        self.frame_5.setStyleSheet("background-color: rgb(49, 162, 242);\n"
"border-radius: 15px;")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_6 = QtWidgets.QLabel(self.frame_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMaximumSize(QtCore.QSize(35, 50))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap(":/Icons/Icons/Group_fill.png"))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_3.addWidget(self.label_6)
        self.personcomboBox = QtWidgets.QComboBox(self.frame_5)
        self.personcomboBox.setStyleSheet("background-color: rgb(49, 162, 242);\n"
"\n"
"\n"
"font: 9pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);\n"
"")
        self.personcomboBox.setFrame(False)
        self.personcomboBox.setObjectName("personcomboBox")
        self.personcomboBox.addItem("")
        self.personcomboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.personcomboBox)
        self.horizontalLayout_6.addWidget(self.frame_5)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem2)
        self.applyButton = QtWidgets.QPushButton(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.applyButton.sizePolicy().hasHeightForWidth())
        self.applyButton.setSizePolicy(sizePolicy)
        self.applyButton.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 9pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(50, 64, 111);\n"
"\n"
"border-radius: 15px;\n"
"border: 2px solid #414F80;")
        self.applyButton.setObjectName("applyButton")
        self.horizontalLayout_6.addWidget(self.applyButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem3)
        self.horizontalLayout_6.setStretch(0, 7)
        self.horizontalLayout_6.setStretch(1, 2)
        self.horizontalLayout_6.setStretch(2, 7)
        self.horizontalLayout_6.setStretch(3, 2)
        self.horizontalLayout_5.addWidget(self.frame_8)
        self.frame_4 = QtWidgets.QFrame(self.frame_7)
        self.frame_4.setStyleSheet("background-color: rgb(18, 30, 52);\n"
"QFrame { box-shadow: 5px 5px 5px 0px rgba(0, 0, 0, 0.1); }")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        self.label_5.setStyleSheet("color:rgb(255, 255, 255);\n"
"background-color: rgb(39, 51, 94);\n"
"border-radius: 7px;")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.frame_4)
        self.label_7.setStyleSheet("color:rgb(255, 255, 255);\n"
"background-color: rgb(39, 51, 94);\n"
"border-radius: 7px;")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 1, 0, 1, 1)
        self.unauthorisedCount = QtWidgets.QLabel(self.frame_4)
        self.unauthorisedCount.setStyleSheet("color:rgb(255, 255, 255);\n"
"border: 2px solid #414F80;\n"
"")
        self.unauthorisedCount.setAlignment(QtCore.Qt.AlignCenter)
        self.unauthorisedCount.setObjectName("unauthorisedCount")
        self.gridLayout_2.addWidget(self.unauthorisedCount, 1, 1, 1, 1)
        self.authorisedCount = QtWidgets.QLabel(self.frame_4)
        self.authorisedCount.setStyleSheet("color:rgb(255, 255, 255);\n"
"border: 2px solid #414F80;")
        self.authorisedCount.setAlignment(QtCore.Qt.AlignCenter)
        self.authorisedCount.setObjectName("authorisedCount")
        self.gridLayout_2.addWidget(self.authorisedCount, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 5)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.horizontalLayout_5.addWidget(self.frame_4)
        self.horizontalLayout_5.setStretch(0, 6)
        self.horizontalLayout_5.setStretch(1, 3)
        self.verticalLayout.addWidget(self.frame_7)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 10)
        self.verticalLayout.setStretch(2, 2)
        self.horizontalLayout_9.addWidget(self.frame_9)
        self.frame_15 = QtWidgets.QFrame(self.frame_16)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_15)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_12 = QtWidgets.QFrame(self.frame_15)
        self.frame_12.setStyleSheet("background-color: rgb(17, 23, 39);\n"
"border-radius:25px;\n"
"border: 2px solid #414F80;")
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_19 = QtWidgets.QFrame(self.frame_12)
        self.frame_19.setStyleSheet("border: none;")
        self.frame_19.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_19.setObjectName("frame_19")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame_19)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_11 = QtWidgets.QLabel(self.frame_19)
        self.label_11.setStyleSheet("color: rgb(164, 175, 215);\n"
"font: 25 10pt \"Calibri Light\";\n"
"border: none;")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_11.addWidget(self.label_11)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem4)
        self.authorisedLeftArrow = QtWidgets.QPushButton(self.frame_19)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.authorisedLeftArrow.sizePolicy().hasHeightForWidth())
        self.authorisedLeftArrow.setSizePolicy(sizePolicy)
        self.authorisedLeftArrow.setStyleSheet("border: 2px solid #414F80;")
        self.authorisedLeftArrow.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/Icons/Icons/Arrow_left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.authorisedLeftArrow.setIcon(icon4)
        self.authorisedLeftArrow.setObjectName("authorisedLeftArrow")
        self.horizontalLayout_11.addWidget(self.authorisedLeftArrow)
        self.authorisedRightArrow = QtWidgets.QPushButton(self.frame_19)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.authorisedRightArrow.sizePolicy().hasHeightForWidth())
        self.authorisedRightArrow.setSizePolicy(sizePolicy)
        self.authorisedRightArrow.setStyleSheet("border: 2px solid #414F80;")
        self.authorisedRightArrow.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/Icons/Icons/Arrow_left(1).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.authorisedRightArrow.setIcon(icon5)
        self.authorisedRightArrow.setObjectName("authorisedRightArrow")
        self.horizontalLayout_11.addWidget(self.authorisedRightArrow)
        self.horizontalLayout_11.setStretch(0, 4)
        self.horizontalLayout_11.setStretch(1, 1)
        self.horizontalLayout_11.setStretch(2, 1)
        self.horizontalLayout_11.setStretch(3, 1)
        self.verticalLayout_2.addWidget(self.frame_19)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem5)
        self.frame_11 = QtWidgets.QFrame(self.frame_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy)
        self.frame_11.setStyleSheet("border: none;")
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.frame_10 = QtWidgets.QFrame(self.frame_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy)
        self.frame_10.setStyleSheet("background-color: rgb(39, 51, 94);\n"
"border-radius: 15px;")
        self.frame_10.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_10)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_9 = QtWidgets.QLabel(self.frame_10)
        self.label_9.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 0, 0, 1, 1)
        self.authorisedNameLabel = QtWidgets.QLabel(self.frame_10)
        self.authorisedNameLabel.setStyleSheet("color: rgb(255, 255, 255);")
        self.authorisedNameLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.authorisedNameLabel.setObjectName("authorisedNameLabel")
        self.gridLayout_3.addWidget(self.authorisedNameLabel, 0, 1, 1, 1)
        self.horizontalLayout_7.addWidget(self.frame_10)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem6)
        self.authorisedPersonImage = QtWidgets.QLabel(self.frame_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.authorisedPersonImage.sizePolicy().hasHeightForWidth())
        self.authorisedPersonImage.setSizePolicy(sizePolicy)
        self.authorisedPersonImage.setMinimumSize(QtCore.QSize(110, 100))
        self.authorisedPersonImage.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"\n"
"border: none;")
        self.authorisedPersonImage.setText("")
        self.authorisedPersonImage.setTextFormat(QtCore.Qt.PlainText)
        self.authorisedPersonImage.setScaledContents(False)
        self.authorisedPersonImage.setObjectName("authorisedPersonImage")
        self.horizontalLayout_7.addWidget(self.authorisedPersonImage)
        self.horizontalLayout_7.setStretch(0, 9)
        self.horizontalLayout_7.setStretch(1, 1)
        self.horizontalLayout_7.setStretch(2, 7)
        self.verticalLayout_2.addWidget(self.frame_11)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 7)
        self.verticalLayout_4.addWidget(self.frame_12)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem7)
        self.frame_14 = QtWidgets.QFrame(self.frame_15)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy)
        self.frame_14.setStyleSheet("background-color: rgb(17, 23, 39);\n"
"border-radius: 25px;\n"
"\n"
"border: 2px solid #414F80;")
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_14)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_20 = QtWidgets.QFrame(self.frame_14)
        self.frame_20.setStyleSheet("border: none;")
        self.frame_20.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_20.setObjectName("frame_20")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.frame_20)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_15 = QtWidgets.QLabel(self.frame_20)
        self.label_15.setStyleSheet("color: rgb(165, 4, 4);\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"border: none;")
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_14.addWidget(self.label_15)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem8)
        self.unauthorisedLeftArrow = QtWidgets.QPushButton(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.unauthorisedLeftArrow.sizePolicy().hasHeightForWidth())
        self.unauthorisedLeftArrow.setSizePolicy(sizePolicy)
        self.unauthorisedLeftArrow.setStyleSheet("border: 2px solid #414F80;")
        self.unauthorisedLeftArrow.setText("")
        self.unauthorisedLeftArrow.setIcon(icon4)
        self.unauthorisedLeftArrow.setObjectName("unauthorisedLeftArrow")
        self.horizontalLayout_14.addWidget(self.unauthorisedLeftArrow)
        self.unauthorisedRightArrow = QtWidgets.QPushButton(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.unauthorisedRightArrow.sizePolicy().hasHeightForWidth())
        self.unauthorisedRightArrow.setSizePolicy(sizePolicy)
        self.unauthorisedRightArrow.setStyleSheet("border: 2px solid #414F80;")
        self.unauthorisedRightArrow.setText("")
        self.unauthorisedRightArrow.setIcon(icon5)
        self.unauthorisedRightArrow.setObjectName("unauthorisedRightArrow")
        self.horizontalLayout_14.addWidget(self.unauthorisedRightArrow)
        self.horizontalLayout_14.setStretch(0, 4)
        self.horizontalLayout_14.setStretch(1, 1)
        self.horizontalLayout_14.setStretch(2, 1)
        self.horizontalLayout_14.setStretch(3, 1)
        self.verticalLayout_3.addWidget(self.frame_20)
        spacerItem9 = QtWidgets.QSpacerItem(17, 19, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem9)
        self.frame_18 = QtWidgets.QFrame(self.frame_14)
        self.frame_18.setStyleSheet("border: none;")
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.frame_18)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem10 = QtWidgets.QSpacerItem(97, 17, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem10)
        self.unauthorisedPersonImage = QtWidgets.QLabel(self.frame_18)
        self.unauthorisedPersonImage.setStyleSheet("color: rgb(164, 175, 215);\n"
"background-color: rgb(39, 51, 94);\n"
"border-radius: 15px;\n"
"border:none;")
        self.unauthorisedPersonImage.setLineWidth(1)
        self.unauthorisedPersonImage.setText("")
        self.unauthorisedPersonImage.setAlignment(QtCore.Qt.AlignCenter)
        self.unauthorisedPersonImage.setObjectName("unauthorisedPersonImage")
        self.horizontalLayout_10.addWidget(self.unauthorisedPersonImage)
        spacerItem11 = QtWidgets.QSpacerItem(16, 17, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem11)
        self.horizontalLayout_10.setStretch(0, 1)
        self.horizontalLayout_10.setStretch(1, 2)
        self.horizontalLayout_10.setStretch(2, 1)
        self.verticalLayout_3.addWidget(self.frame_18)
        spacerItem12 = QtWidgets.QSpacerItem(17, 23, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem12)
        self.frame_13 = QtWidgets.QFrame(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy)
        self.frame_13.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_13.setStyleSheet("border: none;")
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_13)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem13)
        self.shootButton = QtWidgets.QPushButton(self.frame_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shootButton.sizePolicy().hasHeightForWidth())
        self.shootButton.setSizePolicy(sizePolicy)
        self.shootButton.setMinimumSize(QtCore.QSize(100, 100))
        self.shootButton.setStyleSheet("background-color: rgb(165, 4, 4);\n"
"color:rgb(255, 255, 255);\n"
"\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"border-radius: 15px;")
        self.shootButton.setObjectName("shootButton")
        self.horizontalLayout_8.addWidget(self.shootButton)
        spacerItem14 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem14)
        self.horizontalLayout_8.setStretch(0, 3)
        self.horizontalLayout_8.setStretch(1, 5)
        self.horizontalLayout_8.setStretch(2, 3)
        self.verticalLayout_3.addWidget(self.frame_13)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 61)
        self.verticalLayout_3.setStretch(3, 4)
        self.verticalLayout_4.addWidget(self.frame_14)
        self.verticalLayout_4.setStretch(0, 2)
        self.verticalLayout_4.setStretch(1, 1)
        self.verticalLayout_4.setStretch(2, 3)
        self.horizontalLayout_9.addWidget(self.frame_15)
        self.horizontalLayout_9.setStretch(0, 7)
        self.horizontalLayout_9.setStretch(1, 3)
        self.verticalLayout_5.addWidget(self.frame_16)
        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 15)
        self.verticalLayout_6.addWidget(self.frame_17)
        self.verticalLayout_6.setStretch(0, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cameracomboBox.setItemText(0, _translate("MainWindow", "Camera 1"))
        self.cameracomboBox.setItemText(1, _translate("MainWindow", "Camera 2"))
        self.timeStamp.setText(_translate("MainWindow", "TextLabel"))
        self.cameraFeed.setText(_translate("MainWindow", "TextLabel"))
        self.personcomboBox.setItemText(0, _translate("MainWindow", "Person"))
        self.personcomboBox.setItemText(1, _translate("MainWindow", "Animal"))
        self.applyButton.setText(_translate("MainWindow", "Apply"))
        self.label_5.setText(_translate("MainWindow", "Authorised"))
        self.label_7.setText(_translate("MainWindow", "Unauthorised"))
        self.unauthorisedCount.setText(_translate("MainWindow", "1"))
        self.authorisedCount.setText(_translate("MainWindow", "1"))
        self.label_11.setText(_translate("MainWindow", "AUTHORISED PERSON"))
        self.label_9.setText(_translate("MainWindow", "Name:"))
        self.authorisedNameLabel.setText(_translate("MainWindow", "TextLabel"))
        self.label_15.setText(_translate("MainWindow", "UNAUTHORISED PERSON"))
        self.shootButton.setText(_translate("MainWindow", "SHOOT"))

    def imageUpdateSlot(self, image):
        width = self.cameraFeed.width()
        height = self.cameraFeed.height()
        scaledImage = convertToQtFormat(image)
        self.cameraFeed.setPixmap(QtGui.QPixmap.fromImage(scaledImage))
        self.cameraFeed.setScaledContents(True)

    def personDetailsSlot(self, personDetails):
        self.authCount = personDetails.authorizedCount
        self.unauthCount = personDetails.unauthorizedCount

        if personDetails.authorizedCount != 0 and self.authorizedPersonIndex < personDetails.authorizedCount:
            self.authorizedPersons = personDetails.authorizedPersons
            self.authorisedNameLabel.setText(self.authorizedPersons[self.authorizedPersonIndex].name)
            self.authorisedCount.setNum(personDetails.authorizedCount)
            self.authorisedPersonImage.setPixmap(QtGui.QPixmap.fromImage(self.authorizedPersons[self.authorizedPersonIndex].image))
            self.authorisedPersonImage.setScaledContents(True)
        
        if personDetails.unauthorizedCount != 0 and self.unauthorizedPersonIndex < personDetails.unauthorizedCount:
            self.unauthorizedPersons = personDetails.unauthorizedPersons
            self.unauthorisedCount.setNum(personDetails.unauthorizedCount)
            self.unauthorisedPersonImage.setPixmap(QtGui.QPixmap.fromImage(self.unauthorizedPersons[self.unauthorizedPersonIndex].image))
            self.unauthorisedPersonImage.setScaledContents(True)

    def authLeftArrow(self):
        # print("authLeftArrow")
        if (self.authorizedPersonIndex > 0):
            self.authorizedPersonIndex -= 1

    def authRightArrow(self):
        # print("authRightArrow")
        if (self.authorizedPersonIndex < self.authCount - 1):
            self.authorizedPersonIndex += 1

    def unauthLeftArrow(self):
        # print("unauthLeftArrow")
        if (self.unauthorizedPersonIndex > 0):
            self.unauthorizedPersonIndex -= 1

    def unauthRightArrow(self):
        # print("unauthRightArrow")
        if (self.unauthorizedPersonIndex < self.unauthCount - 1):
            self.unauthorizedPersonIndex += 1

    def timeUpdater(self, time):
        self.timeStamp.setText(time)

    def shootUnidentifiedPerson(self):
        print("Shooting unidentified person")

from src.UI import Images_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
