import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QFileDialog, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import cv2
import numpy as np
import math
import subprocess

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理")
        self.setGeometry(0, 0, 3600, 1800)

        #添加文本输入框1
        # 添加打开图片按钮
        self.button_1 = QPushButton("用逗号分割", self)
        self.button_1.setGeometry(300, 0, 250, 50)
        self.button_2 = QPushButton("备用框", self)
        self.button_2.setGeometry(560, 0, 200, 50)
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setGeometry(300, 50, 250, 100)
        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setGeometry(560, 50, 200, 100)
        self.lineedit1.returnPressed.connect(self.on_return_pressed)


        # 添加打开图片按钮
        self.open_button = QPushButton("打开图片", self)
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setGeometry(50, 50, 200, 100)

        # 添加图像显示区域
        #显示处理后的图像区域
        self.button_3 = QPushButton("处理后", self)
        self.button_3.setGeometry(800, 50, 250, 50)
        self.image_label = QLabel(self)
        self.image_label.setGeometry(800, 100, 1200, 1800)

####    #显示原图区域
        self.button_4 = QPushButton("原图", self)
        self.button_4.setGeometry(2050, 50, 250, 50)
        self.image_label1=QLabel(self)
        self.image_label1.setGeometry(2050,100, 1200, 1800)
        #self.image_label1.setPixmap(self.image)

        #(1)
        self.gray_button = QPushButton("图像灰度变换", self)
        self.gray_button.setGeometry(50, 200, 200, 90)
        self.gray_button.clicked.connect(self.on_button_clicked)


        # 添加图像处理功能选择
        self.processing_options1= ["彩色转灰度图", "调暗", "调亮","直方图均衡化"]
        self.processing_combo1=QComboBox(self)
        self.processing_combo1.addItems(self.processing_options1)
        self.processing_combo1.setGeometry(300, 200, 250, 90)

        #(2)
        self.jihe_button = QPushButton("图像几何变换", self)
        self.jihe_button.setGeometry(50, 400, 200, 90)
        self.jihe_button.clicked.connect(self.on_button_clicked)
        # 添加图像处理功能选择
        self.processing_options2 = ["图像平移", "旋转", "仿射错切", "插值缩放","透视"]
        self.processing_combo2=QComboBox(self)
        self.processing_combo2.addItems(self.processing_options2)
        self.processing_combo2.setGeometry(300, 400, 250, 90)

        # (3)
        self.noisy_button = QPushButton("图像去噪", self)
        self.noisy_button.setGeometry(50, 600, 200, 90)
        # 添加图像处理功能选择
        self.processing_options3 = ["加高斯噪声", "加椒盐噪声","均值滤波去噪","中值滤波去噪"]
        self.processing_combo3=QComboBox(self)
        self.processing_combo3.addItems(self.processing_options3)
        self.processing_combo3.setGeometry(300, 600, 250, 90)

        # (4)
        self.edge_button = QPushButton("图像边缘检测", self)
        self.edge_button.setGeometry(50, 800, 200, 90)
        # 添加图像处理功能选择
        self.processing_options4 = ["Sobel", "Prewitt", "Laplacian", "Scharr","Canny","Roberts","Log"]
        self.processing_combo4=QComboBox(self)
        self.processing_combo4.addItems(self.processing_options4)
        self.processing_combo4.setGeometry(300, 800, 250, 90)

        # (5)
        self.cut_button = QPushButton("图像分割", self)
        self.cut_button.setGeometry(50, 1000, 200, 90)
        # 添加图像处理功能选择
        self.processing_options5= ["分割前景","分割背景", "提取人眼区域"]
        self.processing_combo5=QComboBox(self)
        self.processing_combo5.addItems(self.processing_options5)
        self.processing_combo5.setGeometry(300, 1000, 250, 90)

        # (6)
        self.suit_button = QPushButton("特征点匹配", self)
        self.suit_button.setGeometry(50, 1200, 200, 90)
        # 添加图像处理功能选择
        self.processing_options6 = ["SIFT特征检测与描述", "2幅图像匹配"]
        self.processing_combo6=QComboBox(self)
        self.processing_combo6.addItems(self.processing_options6)
        self.processing_combo6.setGeometry(300, 1200, 250, 90)



        #(1)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image1)
        self.process_button.setGeometry(560, 200, 200, 90)

        # (2)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image2)
        self.process_button.setGeometry(560, 400, 200, 90)

        # (3)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image3)
        self.process_button.setGeometry(560, 600, 200, 90)

        # (4)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image4)
        self.process_button.setGeometry(560, 800, 200, 90)

        # (5)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image5)
        self.process_button.setGeometry(560, 1000, 200, 90)

        # (6)
        # 添加图像处理按钮
        self.process_button = QPushButton("处理图像", self)
        self.process_button.clicked.connect(self.process_image6)
        self.process_button.setGeometry(560, 1200, 200, 90)

    # 打开图片
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image files (*.jpg *.jpeg *.png *.jfif *.webp)")
        if file_path:
            self.image = Image.open(file_path)
            self.image_qt = ImageProcessor.PIL2QImage(self.image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
            self.image_label1.setPixmap(self.image_pixmap)

    def on_button_clicked(self):
        # 在文本输入框1中添加文本并设置为焦点
        self.lineedit1.setFocus()
    def on_return_pressed(self):
        # 当第一个文本框输入完成时，将焦点设置到第二个文本框上
        self.lineedit2.setFocus()
    # 处理图像
    #(1)图像灰度变换：
    def process_image1(self):
        processing = self.processing_combo1.currentText()
        if processing == "彩色转灰度图":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image_qt = ImageProcessor.PIL2QImage(gray)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "调暗":
            p = self.lineedit1.text()
            self.image = np.array(self.image)
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)  # 将图像转变为BGR格式
            img1 = cv2.convertScaleAbs(img_bgr, alpha=1, beta=-int(p))
            an= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 将图像转变为RGB格式
            self.image_qt = ImageProcessor.PIL2QImage(an)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "调亮":
            p = self.lineedit1.text()
            self.image = np.array(self.image)
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)  # 将图像转变为BGR格式
            img1 = cv2.convertScaleAbs(img_bgr, alpha=1, beta=int(p))
            liang= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 将图像转变为RGB格式
            self.image_qt = ImageProcessor.PIL2QImage(liang)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "直方图均衡化":
            self.image = np.array(self.image)
            # 将RGB图像转换为HSV颜色空间
            img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            # 对亮度通道进行直方图均衡化
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # 将处理后的图像转换回RGB颜色空间
            hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            self.image_qt = ImageProcessor.PIL2QImage(hsv)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)#将经过处理的HSV图像转换为Qt中的图像格式(QImage)，然后将其转换为Pixmap格式，并在GUI界面的一个标签控件(QLabel)中显示。

    #(2)图像几何变换
    def process_image2(self):
        processing = self.processing_combo2.currentText()
        if processing == "图像平移":
            #提取文本框中的内容
            p = self.lineedit1.text()
            x,y=p.split(",")
            # 定义平移矩阵
            M = np.float32([[1, 0, x], [0, 1, y]])#沿x轴平移x像素，沿y轴平移y像素
            img_array = np.array(self.image)
            # 进行平移变换
            img_trans = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
            self.image_qt = ImageProcessor.PIL2QImage(img_trans)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "旋转":
            p = self.lineedit1.text()
            self.image = np.array(self.image)
            # 定义旋转角度和缩放比例
            angle = float(p)
            scale = 1.0
            # 计算旋转后的图像大小
            (h, w) = self.image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            new_w = int(abs(M[0, 0]) * w + abs(M[0, 1]) * h)
            new_h = int(abs(M[1, 0]) * w + abs(M[1, 1]) * h)
            # 缩放旋转后的图像
            M[0, 2] += (new_w - w) // 2
            M[1, 2] += (new_h - h) // 2
            rotated = cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_AREA)
            #self.image = self.image.rotate(int(p))
            self.image_qt = ImageProcessor.PIL2QImage(rotated)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "仿射错切":
            p = self.lineedit1.text()
            angle_x,angle_y = p.split(",")
            angle_x=int(angle_x)
            angle_y=int(angle_y)
            self.image = np.array(self.image)
            h, w = self.image.shape[:-1]

            shear_matrix_horizontal = np.float32(
                [[1, math.tan(angle_x * math.pi / 180), 0], [0, 1, 0], [0, 0, 1]])
            shear_matrix_vertical = np.float32(
                [[1, 0, 0], [math.tan(angle_y * math.pi / 180), 1, 0], [0, 0, 1]])

            # 将水平方向和垂直方向的错切矩阵合并为一个矩阵
            shear_matrix = np.matmul(shear_matrix_horizontal, shear_matrix_vertical)

            # 对图像进行水平方向和垂直方向的错切
            img_sheared = cv2.warpPerspective(self.image, shear_matrix, (w, h))

            self.image_qt = ImageProcessor.PIL2QImage(img_sheared)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "插值缩放":
            img_array = np.array(self.image)
            # 定义缩放比例
            p = self.lineedit1.text()
            scale_percent = int(p)
            # 计算缩放后的图像大小
            width = int(img_array.shape[1] * scale_percent / 100)
            height = int(img_array.shape[0] * scale_percent / 100)
            # 定义目标图像大小
            dim = (width, height)            # 进行插值缩放
            img_resized = cv2.resize(img_array, dim, interpolation=cv2.INTER_AREA)
            self.image_qt = ImageProcessor.PIL2QImage(img_resized)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "透视":
            self.image = np.array(self.image)
            trans_img=self.image
            self.perspective_transform()
            # self.image_qt = ImageProcessor.PIL2QImage(self.image)
            # self.image_pixmap = QPixmap.fromImage(self.image_qt)
            # self.image_label.setPixmap(self.image_pixmap)

    def perspective_transform(self):
        h, w = self.image.shape[:-1]
        size = (w, h)
        scr_point = []  # 存储鼠标点击的四个点，分别为，左上，右上，左下，右下
        img_2 = self.image.copy()

        # 鼠标点击透视前四个点的坐标
        def mouse_points(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img_2, (x, y), 10, (0, 0, 225), 0)
                cv2.imshow('origin', img_2)
                scr_point.append([x, y])
                if len(scr_point) == 4:
                    scr_points = np.array(scr_point, dtype='float32')
                    # 设置透视变换后四个角点坐标
                    dst_points = np.array([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]], dtype='float32')
                    # 计算透视变换矩阵
                    rotation = cv2.getPerspectiveTransform(scr_points, dst_points)
                    # 透视变换投影
                    img_warp = cv2.warpPerspective(img_2, rotation, size)
                    cv2.imshow('img_warp', img_warp)  # 透视后的展示图

        cv2.namedWindow('origin')
        cv2.setMouseCallback('origin', mouse_points)

        cv2.namedWindow('origin')
        cv2.setMouseCallback('origin', mouse_points)

        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        cv2.imshow('origin', img_2)
        key = cv2.waitKey()

        cv2.destroyAllWindows()

        cv2.destroyAllWindows()


    #(3)图像去噪
    def process_image3(self):
        noise_image=np.array(self.image)
        processing = self.processing_combo3.currentText()
        if processing == "加高斯噪声":
            noise_image = self.add_gauss_noise()
            self.image_qt = ImageProcessor.PIL2QImage(noise_image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        if processing == "加椒盐噪声":
            noise_image = self.add_salt_pepper_noise()
            self.image_qt = ImageProcessor.PIL2QImage(noise_image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "均值滤波去噪":
            self.image = np.array(self.image)
            wipe= cv2.blur(noise_image, (3, 3))
            self.image_qt = ImageProcessor.PIL2QImage(wipe)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "中值滤波去噪":
            self.image = np.array(self.image)
            wipe= cv2.medianBlur(noise_image,3)
            self.image_qt = ImageProcessor.PIL2QImage(wipe)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)



    def add_gauss_noise(self, mean=0, var=1000):
        # 将PIL图像转换为numpy数组
        image_array = np.array(self.image)
        # 添加高斯噪声
        noise = np.random.normal(mean, var ** 0.5, image_array.shape)
        out_array = image_array + noise
        out_array = np.clip(out_array, 0, 255).astype('uint8')
        # 将噪声图像转换为PIL对象
        out_image = Image.fromarray(out_array)
        return out_image

    def add_salt_pepper_noise(self, density=0.05):
        # 将PIL图像转换为numpy数组
        image_array = np.array(self.image)
        # 获取图像的宽高和通道数
        height, width, channel = image_array.shape
        # 计算需要添加椒盐噪声的像素数量
        num_saltpepper = int(height * width * density)
        # 在随机位置生成椒噪声
        coords = [np.random.randint(0, i - 1, int(num_saltpepper / 2)) for i in image_array.shape]
        image_array[coords[0], coords[1], :] = 255
        # 在随机位置生成盐噪声
        coords = [np.random.randint(0, i - 1, int(num_saltpepper / 2)) for i in image_array.shape]
        image_array[coords[0], coords[1], :] = 0
        # 将噪声图像转换为PIL对象
        out_image = Image.fromarray(image_array.astype('uint8'))
        return out_image
    #(4)
    def process_image4(self):
        processing = self.processing_combo4.currentText()
        if processing == "Sobel":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            img_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            img_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
            self.image_qt = ImageProcessor.PIL2QImage(img_sobel)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Prewitt":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            img_prewitt_x = cv2.filter2D(gray, -1, kernel_x)
            img_prewitt_y = cv2.filter2D(gray, -1, kernel_y)
            img_prewitt = cv2.addWeighted(img_prewitt_x, 0.5, img_prewitt_y, 0.5, 0)
            self.image_qt = ImageProcessor.PIL2QImage(img_prewitt)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Laplacian":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            img_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            self.image_qt = ImageProcessor.PIL2QImage(img_laplacian)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Scharr":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            img_scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            img_scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            img_scharr = cv2.addWeighted(img_scharr_x, 0.5, img_scharr_y, 0.5, 0)
            self.image_qt = ImageProcessor.PIL2QImage(img_scharr)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Canny":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            img_canny = cv2.Canny(gray, 50, 150)
            self.image_qt = ImageProcessor.PIL2QImage(img_canny)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Roberts":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])
            img_roberts_x = cv2.filter2D(gray, -1, kernel_x)
            img_roberts_y = cv2.filter2D(gray, -1, kernel_y)
            img_roberts = cv2.addWeighted(img_roberts_x, 0.5, img_roberts_y, 0.5, 0)
            self.image_qt = ImageProcessor.PIL2QImage(img_roberts)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "Log":
            self.image = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # 使用高斯滤波器对图像进行平滑处理
            img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
            # 使用LoG算子进行边缘检测
            img_log = cv2.Laplacian(img_gaussian, cv2.CV_64F)
            # 阈值化处理
            img_log = np.uint8(np.absolute(img_log))
            self.image_qt = ImageProcessor.PIL2QImage(img_log)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)


    #(5)
    def process_image5(self):
        processing = self.processing_combo5.currentText()
        if processing == "分割前景":
            self.image = np.array(self.image).astype(np.uint8)
            # 使用GrabCut算法进行图像分割
            mask = np.zeros(self.image.shape[:2], np.uint8)#创建一个与原始图像大小相同的mask矩阵，并初始化为0。
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)#创建两个bgdModel和fgdModel矩阵用于存储背景和前景模型的参数，都初始化为0
            rect = (50, 50, self.image.shape[1] - 50, self.image.shape[0] - 50)  # 分割矩形区域

            cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)#grabCut函数进行图像分割，将原始图像、mask、rect、bgdModel、fgdModel作为输入参数，并指定分割的迭代次数为5，初始化方式为基于矩形区域的初始化

            # 提取分割结果
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented_image = self.image * mask[:, :, np.newaxis]
            self.image_qt = ImageProcessor.PIL2QImage(segmented_image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "分割背景":
            self.image = np.array(self.image).astype(np.uint8)
            mask = np.zeros(self.image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (50, 50, self.image.shape[1] - 50, self.image.shape[0] - 50)  # 分割矩形区域

            cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            # 提取分割结果
            mask = np.where((mask == 0) | (mask == 2), 1, 0).astype('uint8')
            segmented_image = self.image * mask[:, :, np.newaxis]
            self.image_qt = ImageProcessor.PIL2QImage(segmented_image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "提取人眼区域":
            self.image = np.array(self.image)
            # 加载分类器
            eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
            # 转换为灰度图像
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # 检测眼睛
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            # 绘制矩形框
            for (x, y, w, h) in eyes:
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.image_qt = ImageProcessor.PIL2QImage(self.image)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)

    #(6)
    def process_image6(self):
        processing = self.processing_combo6.currentText()
        if processing == "SIFT特征检测与描述":
            self.image = np.array(self.image).astype(np.uint8)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            # 绘制关键点
            img_with_keypoints = cv2.drawKeypoints(self.image, keypoints, None)
            self.image_qt = ImageProcessor.PIL2QImage(img_with_keypoints)
            self.image_pixmap = QPixmap.fromImage(self.image_qt)
            self.image_label.setPixmap(self.image_pixmap)
        elif processing == "2幅图像匹配":
            # 运行另一个 Python 文件
            subprocess.run(['python', 'suit.py'])

    # 将 PIL.Image 转换为 QImage
    @staticmethod
    def PIL2QImage(image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)
        return qimage


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
