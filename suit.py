import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PIL import Image

class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # 创建 UI 界面
        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        self.label3 = QLabel(self)
        self.button1 = QPushButton('打开图像1', self)
        self.button2 = QPushButton('打开图像2', self)
        self.button3 = QPushButton('2幅图像匹配', self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        self.setLayout(self.layout)

        # 为按钮添加点击事件
        self.button1.clicked.connect(self.open_image1)
        self.button2.clicked.connect(self.open_image2)
        self.button3.clicked.connect(self.match)

    def open_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            self.image1 = Image.open(file_path)
            self.image_qt1 = ImageDisplay.PIL2QImage(self.image1)
            self.image_pixmap1 = QPixmap.fromImage(self.image_qt1)
            self.label1.setPixmap(self.image_pixmap1)

    def open_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            self.image2= Image.open(file_path)
            self.image_qt2 = ImageDisplay.PIL2QImage(self.image2)
            self.image_pixmap2 = QPixmap.fromImage(self.image_qt2)
            self.label2.setPixmap(self.image_pixmap2)

    def match(self):
        self.image1=np.array(self.image1)
        self.image2=np.array(self.image2)
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        # 初始化ORB检测器
        orb = cv2.ORB_create()
        # 找到关键点和描述符
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        # 初始化匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 匹配关键点
        matches = bf.match(des1, des2)
        # 按照距离排序并保留前50个匹配
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        # 绘制匹配结果
        img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches, None, flags=2)
        #显示结果
        cv2.imshow('result',img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def PIL2QImage(image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)
        return qimage



if __name__ == '__main__':
    # 创建应用程序对象
    app = QApplication(sys.argv)

    # 创建图像显示窗口并显示它
    image_display = ImageDisplay()
    image_display.show()

    # 运行应用程序
    sys.exit(app.exec_())