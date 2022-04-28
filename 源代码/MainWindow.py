
import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import math
import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QMessageBox
from pytesseract import pytesseract


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(1470, 370, 201, 71))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(1470, 470, 201, 71))
        self.radioButton_2.setObjectName("radioButton_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(100, 170, 1311, 801))
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(330, 70, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 70, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(800, 70, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1030, 70, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1470, 730, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1470, 790, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1470, 850, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1470, 910, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(1610, 730, 256, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(1610, 790, 256, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.textBrowser_2.setFont(font)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(1610, 850, 256, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.textBrowser_3.setFont(font)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(1610, 910, 256, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.textBrowser_4.setFont(font)
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(100, 70, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(1610, 670, 256, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.textBrowser_5.setFont(font)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1470, 670, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_6.clicked.connect(self.LoadImage)
        self.pushButton.clicked.connect(self.CircleDetection)
        self.pushButton_2.clicked.connect(self.SplitPanel)
        self.pushButton_3.clicked.connect(self.PointerDetection)
        self.pushButton_4.clicked.connect(self.CalculateValue)
        self.radioButton.setChecked(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "检测圆"))
        self.pushButton_2.setText(_translate("MainWindow", "表盘分割"))
        self.pushButton_3.setText(_translate("MainWindow", "检测指针"))
        self.pushButton_4.setText(_translate("MainWindow", "求出参数"))
        self.label.setText(_translate("MainWindow", "转速表"))
        self.label_2.setText(_translate("MainWindow", "速度表"))
        self.label_3.setText(_translate("MainWindow", "油量表"))
        self.label_4.setText(_translate("MainWindow", "水温表"))
        self.pushButton_6.setText(_translate("MainWindow", "打开图片"))
        self.label_5.setText(_translate("MainWindow", "总里程"))
        self.radioButton.setText(_translate("MainWindow", "尼桑蓝鸟"))
        self.radioButton_2.setText(_translate("MainWindow", "大众帕萨特"))

    # 打开图片
    def LoadImage(self):
        # 获得路径
        path = QFileDialog.getOpenFileName(None, '选择图片', 'c:\\', 'All Files (*)')
        global ImgName
        ImgName = path[0]
        if len(ImgName):
            # 加载图片
            img = cv2.imread(ImgName)
            # 更换颜色通道以保证界面上显示图片的颜色正确
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.medianBlur(img, 3)
            # 获取图片长和宽
            height = img.shape[0]
            width = img.shape[1]
            # 显示图片
            widthStep = width * 3 
            frame = QImage(img, width, height, widthStep, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.graphicsView.setScene(self.scene)

    # 排序
    def takeSecond(self, elem):
        return elem[0]

    # 检测圆
    def CircleDetection(self):
        # 读取图片，转为灰度图,高斯平滑
        img = cv2.imread(ImgName)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # huogh圆检测,具体参数意义百度一下
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=250, param2=70, minRadius=0, maxRadius=0)
        # 输出每个圆的圆心坐标、半径
        circles = np.uint16(np.around(circles))
        # 检测图生成
        identify_img = img
        for i in circles[0, :]:
            cv2.circle(identify_img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 绿色圆
            cv2.circle(identify_img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 红色圆心
        # 图像展示
        cv2.imshow("identify_img", identify_img)

    # 表盘分割
    def SplitPanel(self):
        # 读取图片，转为灰度图
        img = cv2.imread(ImgName)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # huogh圆检测,具体参数意义百度一下
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=250, param2=50, minRadius=0, maxRadius=0)
        # 输出每个圆的圆心坐标、半径
        circles = np.uint16(np.around(circles))
        # 检测图生成
        identify_img = img
        cropped0 = identify_img[circles[0][0][1] - circles[0][0][2] - 5:circles[0][0][1] + circles[0][0][2] + 5,
                   circles[0][0][0] - circles[0][0][2] - 5:circles[0][0][0] + circles[0][0][2] + 5]
        cropped1 = identify_img[circles[0][1][1] - circles[0][1][2] - 5:circles[0][1][1] + circles[0][1][2] + 5,
                   circles[0][1][0] - circles[0][1][2] - 5:circles[0][1][0] + circles[0][1][2] + 5]
        cropped2 = identify_img[circles[0][2][1] - circles[0][2][2] - 5:circles[0][2][1] + circles[0][2][2] + 5,
                   circles[0][2][0] - circles[0][2][2] - 5:circles[0][2][0] + circles[0][2][2] + 5]
        cropped3 = identify_img[circles[0][3][1] - circles[0][3][2] - 5:circles[0][3][1] + circles[0][3][2] + 5,
                   circles[0][3][0] - circles[0][3][2] - 5:circles[0][3][0] + circles[0][3][2] + 5]
        cv2.imwrite("cropped0.jpg", cropped0)
        cv2.imwrite("cropped1.jpg", cropped1)
        cv2.imwrite("cropped2.jpg", cropped2)
        cv2.imwrite("cropped3.jpg", cropped3)
        QMessageBox.information(None, "Information", "完成！", QMessageBox.Yes | QMessageBox.No)
        img1 = cv2.imread('cropped0.jpg')
        img2 = cv2.imread('cropped1.jpg')
        img3 = cv2.imread('cropped2.jpg')
        img4 = cv2.imread('cropped3.jpg')
        # 图像展示
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        cv2.imshow("3", img3)
        cv2.imshow("4", img4)

    # 指针识别
    def PointerDetect1(self, img):
        house = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, house = cv2.threshold(house, 40, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(house, 75, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=5)
        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            for x1, y1, x2, y2 in line:
                lenth = self.len(x1, y1, x2, y2)
                xian = x1, y1, x2, y2, lenth
            xian2.append(xian)

        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)
        cv2.line(img, (xian2[0][0], xian2[0][1]), (xian2[0][2], xian2[0][3]), (0, 0, 255), 2)
        return img

    # 指针识别
    def PointerDetect2(self, img):
        height, width, channel = img.shape
        new = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                if img[i][j][2] > 128 and img[i][j][0] < 128 and img[i][j][1] < 128:
                    new[i][j] = 255
                else:
                    new[i][j] = 0

        canny_img = cv2.Canny(new, 75, 150)
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 20,
                                minLineLength=10, maxLineGap=2.5)

        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            for x1, y1, x2, y2 in line:
                lenth = self.len(x1, y1, x2, y2)
                xian = x1, y1, x2, y2, lenth
            xian2.append(xian)

        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)
        cv2.line(img, (xian2[0][0], xian2[0][1]), (xian2[0][2], xian2[0][3]), (0, 255, 0), 2)
        return img

    # 指针识别入口函数
    def PointerDetection(self):
        # 打开图片
        img1 = cv2.imread('cropped0.jpg')
        img2 = cv2.imread('cropped1.jpg')
        img3 = cv2.imread('cropped2.jpg')
        img4 = cv2.imread('cropped3.jpg')
        if self.radioButton.isChecked():
            # 指针检测
            nimg1 = self.PointerDetect1(img1)
            nimg2 = self.PointerDetect1(img2)
            nimg3 = self.PointerDetect1(img3)
            nimg4 = self.PointerDetect1(img4)
            # 图像展示
            cv2.imshow("1", nimg1)
            cv2.imshow("2", nimg2)
            cv2.imshow("3", nimg3)
            cv2.imshow("4", nimg4)
        else:
            # 指针检测
            nimg1 = self.PointerDetect2(img1)
            nimg2 = self.PointerDetect2(img2)
            nimg3 = self.PointerDetect2(img3)
            nimg4 = self.PointerDetect2(img4)
            # 图像展示
            cv2.imshow("1", nimg1)
            cv2.imshow("2", nimg2)
            cv2.imshow("3", nimg3)
            cv2.imshow("4", nimg4)

    # 传递坐标信息
    def Coordinates(self, img, T, Min, Max):
        house = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(house, 255, 255)  # 255

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, T, minLineLength=Min, maxLineGap=Max)
        lines = lines[:, 0, :]
        for x1, y1, x2, y2 in lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return lines

    # 传递坐标信息
    def Coordinates2(self, img):
        height, width, channel = img.shape
        new = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                if img[i][j][2] > 128 and img[i][j][0] < 128 and img[i][j][1] < 128:
                    new[i][j] = 255
                else:
                    new[i][j] = 0

        canny_img = cv2.Canny(new, 75, 150)
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 20,
                                minLineLength=10, maxLineGap=2.5)
        return lines

    # 两点间距离
    def len(self, x1, y1, x2, y2):
        lenth = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return lenth

    # 计算水温表示数
    def CalculateWaterTable1(self, img):
        a = self.Coordinates(img, 26, 5, 5)
        x1, y1, x2, y2 = a[0][0], a[0][1], a[0][2], a[0][3]
        k = (y1 - y2) / (x1 - x2)
        angle = np.rad2deg(np.arctan(k))
        val = (angle - 48) / 96
        val *= 100
        val = np.round(val, 2)
        self.textBrowser_4.setText(str(val) + '%')

    # 计算油量表示数
    def CalculateOilTable1(self, img):
        a = self.Coordinates(img, 30, 10, 1.45)
        x1, y1, x2, y2 = a[0][0], a[0][1], a[0][2], a[0][3]
        k = (y1 - y2) / (x1 - x2)
        angle = np.rad2deg(np.arctan(k))
        val = (angle + 45) / 90
        val *= 100
        val = np.round(val, 2)
        self.textBrowser_3.setText(str(val) + '%')

    def binary(self, img, T):
        height, width = img.shape
        newImg = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                if img[i, j] > T:
                    newImg[i, j] = 255
                else:
                    newImg[i, j] = 0
        return newImg

    def Calculate1(self, img):
        height, width, depth = img.shape
        r = height / 2 - 5
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = self.binary(gray_img, 150)

        lines = self.Coordinates(img, 30, 60, 6)
        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            x1, y1, x2, y2 = line
            lenth = self.len(x1, y1, x2, y2)
            xian = x1, y1, x2, y2, lenth
            xian2.append(xian)
        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)

        x1, y1, x2, y2 = xian2[0][0], xian2[0][1], xian2[0][2], xian2[0][3]
        x0, y0 = r + 5, r + 5  # 圆心
        len1 = self.len(x0, y0, x1, y1)
        len2 = self.len(x0, y0, x2, y2)
        if len1 < len2:
            x1 = x2
            y1 = y2

        k = -(y0 - y1) / (x0 - x1)  # 计算斜率
        degree = math.degrees(math.atan(k))
        if x0 > x1:
            degree += 180

        judge = []  # 建立判断数组
        for i in range(360):
            x2 = int(x0 + r * math.cos(math.radians(i + degree)))
            y2 = int(y0 - r * math.sin(math.radians(i + degree)))
            flag = False
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if binary_img[y2 + j][x2 + k] == 255:
                        flag = True
            if flag:
                judge.append(1)
            else:
                judge.append(0)

        maxNum = 0
        angle = 0
        angles = [0, 0]
        while angle < 360:  # 寻找最长的0串
            if judge[angle] == 0:
                start = angle
                angle += 1
                while angle < 360:
                    if judge[angle] == 1:
                        lens = angle - start
                        if lens > maxNum:
                            maxNum = lens
                            angles[0] = start
                            angles[1] = angle - 1
                        break
                    angle += 1
            angle += 1

        angles[0] -= 3
        angles[1] += 3
        val = angles[0] / (360 - angles[1] + angles[0])
        if val < 0:
            val = 0
        return val

    def Calculate2(self, img):
        height, width, depth = img.shape
        r = height / 2 - 5

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = self.binary(gray_img, 150)

        lines = self.Coordinates2(img)
        lines = lines[:, 0, :]
        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            x1, y1, x2, y2 = line
            lenth = self.len(x1, y1, x2, y2)
            xian = x1, y1, x2, y2, lenth
            xian2.append(xian)
        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)

        x1, y1, x2, y2 = xian2[0][0], xian2[0][1], xian2[0][2], xian2[0][3]
        x0, y0 = r + 5, r + 5  # 圆心
        len1 = self.len(x0, y0, x1, y1)
        len2 = self.len(x0, y0, x2, y2)
        if len1 < len2:
            x1 = x2
            y1 = y2

        k = -(y0 - y1) / (x0 - x1)  # 计算斜率
        degree = math.degrees(math.atan(k))
        if x0 > x1:
            degree += 180

        judge = []  # 建立判断数组
        r = self.len(xian2[0][0], xian2[0][1], x0, y0)

        for i in range(360):
            x2 = int(x0 + r * math.cos(math.radians(i + degree)))
            y2 = int(y0 - r * math.sin(math.radians(i + degree)))
            flag = False
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if binary_img[y2 + j][x2 + k] == 255:
                        flag = True
            if flag:
                judge.append(1)
            else:
                judge.append(0)

        maxNum = 0
        angle = 0
        angles = [0, 0]
        while angle < 360:  # 寻找最长的0串
            if judge[angle] == 0:
                start = angle
                angle += 1
                while angle < 360:
                    if judge[angle] == 1:
                        lens = angle - start
                        if lens > maxNum:
                            maxNum = lens
                            angles[0] = start
                            angles[1] = angle - 1
                        break
                    angle += 1
            angle += 1

        angles[0] -= 3
        angles[1] += 3
        val = angles[0] / (360 - angles[1] + angles[0])
        if val < 0:
            val = 0
        return val

    def find(self, num, circles):
        m = 1000
        index = 0
        for i in range(circles.shape[0]):
            if (abs(circles[i][2] - num) < m):
                m = circles[i][2] - num
                index = i
        return index

    def getLongestZero(self, judge):
        angle = 0
        angles = [0, 0]
        maxNum = 0
        while (angle < len(judge)):
            if (judge[angle] == 0):
                start = angle
                angle += 1
                while (angle < len(judge)):
                    if (judge[angle] == 1 or angle == len(judge) - 1):
                        lens = angle - start
                        if (lens > maxNum):
                            maxNum = lens
                            angles[0] = start
                            angles[1] = angle - 1
                        break
                    angle += 1
            angle += 1
        return angles

    def Cal11(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x0 = int(img.shape[0] / 2)
        y0 = int(img.shape[1] / 2)
        r = x0 - 5

        # canny变换
        ret, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        canny_img = cv2.Canny(binary_img, 75, 150)

        # 直线检测
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 10,
                               minLineLength=10, maxLineGap=3)

        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            for x1, y1, x2, y2 in line:
                lenth = self.len(x1, y1, x2, y2)
                xian = x1, y1, x2, y2, lenth
            xian2.append(xian)

        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)

        x1 = xian2[0][0]
        y1 = xian2[0][1]
        # 找出离圆心最远的点为指针的端点
        if (self.len(x1, y1, x0, y0) < self.len(xian2[0][2], xian2[0][3], x0, y0)):
            x1 = xian2[0][2]
            y1 = xian2[0][3]

        k = -(y0 - y1) / (x0 - x1)  # 计算斜率
        degree = math.degrees(math.atan(k))
        if (x1 < x0):
            degree += 180

        judge = []
        for i in range(360):
            x2 = int(x0 + r * math.cos(math.radians(i + degree)))
            y2 = int(y0 - r * math.sin(math.radians(i + degree)))
            flag = False
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (binary_img[y2 + k][x2 + j] == 255):
                        flag = True
            if (flag):
                judge.append(1)
            else:
                judge.append(0)
        angles = [0, 0]
        angles = self.getLongestZero(judge)
        val = 0
        if (judge[0] == 0):
            if (angles[0] == 0):
                oil,wtr = 0,1
            else:
                oil,wtr = 1,0
        else:
            oil = (360 - angles[1]) / (360 - angles[1] + angles[0])
            wtr = angles[0] / (360 - angles[1] + angles[0])
        return oil, wtr

    def Cal222(self, img):
        r = img.shape[0] / 2 - 5
        x0 = r + 5
        y0 = r + 5
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

        # huogh圆检测,具体参数意义百度一下
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, r / 2 - 2,
                                   param1=250, param2=20, minRadius=0, maxRadius=0)

        # 输出每个圆的圆心坐标、半径
        circles = np.uint16(np.around(circles))
        circles = circles[0]  # 降低维度

        x_small = circles[self.find(r / 4, circles)][0]  # 小圆圆心
        y_small = circles[self.find(r / 4, circles)][1]

        # 消除大圆圆环
        for i in range(360):
            x1 = int(x0 + r * math.cos(math.radians(i)))
            y1 = int(y0 - r * math.sin(math.radians(i)))
            for j in range(-3, 4):
                for k in range(-3, 4):
                    binary_img[y1 + j][x1 + k] = 0

        # 找指针
        height, width, channel = img.shape
        new = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                if (img[i][j][2] > 128 and img[i][j][0] < 128 and img[i][j][1] < 128):
                    new[i][j] = 255
                else:
                    new[i][j] = 0

        canny_img = cv2.Canny(new, 75, 150)
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 20,
                                minLineLength=10, maxLineGap=2.5)

        xian = (0, 0, 0, 0, 0)  # 用元组存储
        xian2 = []  # 降低维度
        for line in lines:
            for x1, y1, x2, y2 in line:
                lenth = self.len(x1, y1, x2, y2)
                xian = x1, y1, x2, y2, lenth
            xian2.append(xian)

        xian2 = sorted(xian2, key=lambda x: (x[4]), reverse=True)
        x2 = xian2[0][0]
        y2 = xian2[0][1]
        # 找出离圆心最远的点为指针的端点
        if (self.len(x2, y2, x_small, y_small) < self.len(xian2[0][2], xian2[0][3], x_small, y_small)):
            x2 = xian2[0][2]
            y2 = xian2[0][3]

        r_new = self.len(x2, y2, x_small, y_small)
        k = -(y_small - y2) / (x_small - x2)  # 计算斜率
        degree = math.degrees(math.atan(k))
        if (x2 < x_small):
            degree += 180
        r_new -= 2
        # 算开始角度

        judge = []
        for i in range(360):
            x1 = int(x_small + r_new * math.cos(math.radians(i + degree)))
            y1 = int(y_small - r_new * math.sin(math.radians(i + degree)))
            if (y1 >= img.shape[0] or x1 >= img.shape[1] or y1 < 0 or x1 < 0):
                break;
            if (binary_img[y1][x1] == 255):
                judge.append(1)
            else:
                judge.append(0)

        angles = [0, 0]
        angles = self.getLongestZero(judge)
        # 算出最大角度
        judge.clear()
        judge = []
        for i in range(0, 360):
            x1 = int(x_small + r_new * math.cos(math.radians(-i + degree)))
            y1 = int(y_small - r_new * math.sin(math.radians(-i + degree)))
            if (y1 >= img.shape[0] or x1 >= img.shape[1] or y1 < 0 or x1 < 0):
                break;
            if (binary_img[y1][x1] == 255):
                judge.append(1)
            else:
                judge.append(0)

        angles2 = [0, 0]
        angles2 = self.getLongestZero(judge)

        angles[0] -= 4
        # 检测图生成
        oil = 1 * angles[0] / (angles[0] + angles2[0])
        wtr = 50 + 100 * angles[0] / (angles[0] + angles2[0])
        return oil, wtr

    # 计算水温表示数
    def CalculateWaterTable2(self, img):
        a = self.Coordinates2(img)
        a = a[:, 0, :]
        x1, y1, x2, y2 = a[0][0], a[0][1], a[0][2], a[0][3]
        k = (y1 - y2) / (x1 - x2)
        angle = np.rad2deg(np.arctan(k))
        val = (angle - 45) / 90
        val *= 100
        val = np.round(val, 2)
        self.textBrowser_4.setText(str(val) + '%')

    # 计算油量表示数
    def CalculateOilTable2(self, img):
        a = self.Coordinates2(img)
        a = a[:, 0, :]
        x1, y1, x2, y2 = a[0][0], a[0][1], a[0][2], a[0][3]
        k = (y1 - y2) / (x1 - x2)
        angle = np.rad2deg(np.arctan(k))
        val = angle / (-60)
        val *= 100
        val = np.round(val, 2)
        self.textBrowser_3.setText(str(val) + '%')

    # 计算里程
    def CalculateMileage(self, img):
        if self.radioButton.isChecked():
            width = 790
            height = 450
            box = (width, height, width + 50, height + 36)
        else:
            width = 690
            height = 485
            box = (width, height, width + 64, height + 26)
        newImg = img.crop(box)
        time = 1
        width = int(newImg.size[0]) * time  # 可以根据自己需求扩大倍数
        height = int(newImg.size[1]) * time
        newImg = newImg.resize((width, height), Image.ANTIALIAS)
        newImg = newImg.convert('L')
        threshold = 150
        table = []
        for i in range(256):
            if i < threshold:
                table.append(1)
            else:
                table.append(0)
        newImg = newImg.point(table, '1')
        newImg.save('trip.png')
        trip = pytesseract.image_to_string(newImg, lang='num', config='--psm 6 ddd')
        # trip = pytesseract.image_to_string(newImg)
        print(trip)
        self.textBrowser_5.setText(trip+'km')

    # 参数计算
    def CalculateValue(self):
        img = Image.open(ImgName)
        self.CalculateMileage(img)
        if self.radioButton.isChecked():
            img1 = cv2.imread('cropped0.jpg')
            val1 = self.Calculate1(img1) * 8
            val1 = np.round(val1, 2)
            self.textBrowser.setText(str(val1)+'r/min')

            img2 = cv2.imread('cropped1.jpg')
            val2 = self.Calculate1(img2) * 160
            val2 = np.round(val2, 2)
            self.textBrowser_2.setText(str(val2)+'km/h')

            img3 = cv2.imread('cropped2.jpg')
            # self.CalculateWaterTable1(img3)
            ans = self.Cal11(img3)[1]*100
            ans = np.round(ans, 2)
            self.textBrowser_4.setText(str(ans)+'%')

            img4 = cv2.imread('cropped3.jpg')
            # self.CalculateOilTable1(img4)
            ans = self.Cal11(img4)[0]*100
            ans = np.round(ans, 2)
            self.textBrowser_3.setText(str(ans)+'%')
        else:
            img1 = cv2.imread('cropped0.jpg')
            val1 = self.Calculate2(img1) * 8
            val1 = np.round(val1, 2)
            self.textBrowser.setText(str(val1)+'r/min')

            img2 = cv2.imread('cropped1.jpg')
            val2 = self.Calculate2(img2) * 260
            val2 = np.round(val2, 2)
            self.textBrowser_2.setText(str(val2)+'km/h')

            img3 = cv2.imread('cropped2.jpg')
            # self.CalculateWaterTable2(img3)
            ans = self.Cal222(img3)[1]
            ans = np.round(ans, 2)
            self.textBrowser_4.setText(str(ans)+'℃')

            img4 = cv2.imread('cropped3.jpg')
            # self.CalculateOilTable2(img4)
            ans = self.Cal222(img4)[0]
            ans = np.round(ans, 2)
            self.textBrowser_3.setText(str(ans)+'%')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
