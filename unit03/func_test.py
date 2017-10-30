# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/edison/mainwindow2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator,FuncFormatter
from fractions import Fraction
import io
from PyQt5.Qt import QPixmap

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(666, 432)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 90, 631, 300))
        self.label.setText("")
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(50, 10, 301, 41))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(40, 10, 211, 20))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.widget)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.widget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout.addWidget(self.radioButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(450, 10, 113, 32))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 666, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.label.clear)
        self.pushButton.clicked.connect(self.functions_A)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图表测试"))
        self.radioButton.setText(_translate("MainWindow", "RadioButton"))
        self.radioButton_2.setText(_translate("MainWindow", "RadioButton"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))

    def functions_A(self):
        x = np.arange(0,4*np.pi,0.01)
        fig , ax = plt.subplots(figsize=(8,4))
        plt.plot(x,np.sin(x),x,np.cos(x))
        def pi_formatter(x,pos):
            frac = Fraction(int(np.round(x/(np.pi/4))),4)
            d,n = frac.denominator,frac.numerator
            if frac == 0 :
                return "0"
            elif frac == 1:
                return "$\pi$"  
            elif d == 1 :
                return r"${%d} \pi$" %n
            elif n == 1:
                return r"$\frac{\pi}{%d}$"%d
            return r"$\frac{%d\pi}{%d}$"%(n,d)

        plt.ylim(-1.5,1.5)
        plt.xlim(0,np.max(x))

        plt.subplots_adjust(bottom=0.15)
        plt.grid()
#设置x轴刻度
        ax.xaxis.set_major_locator(MultipleLocator(np.pi/4))
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
        ax.xaxis.set_minor_locator(MultipleLocator(np.pi/20))
#设置刻度文本的大小
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        plt.savefig("test.png",dpi=80)
        self.label.setPixmap(QPixmap("test.png"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())