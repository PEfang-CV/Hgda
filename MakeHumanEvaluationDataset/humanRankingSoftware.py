# !/usr/bin/env python3
"""
Function: hunman ranking for the generated defect images
Author: TyFang
Date: 2023/6/29
"""

import os
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMessageBox,QComboBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import shutil
import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


draw_flag = False
ix, iy = -1, -1


class ImageViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reward Dataset Maker")

        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        self.setFont(font)
        self.showMaximized()
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.h_layout = QHBoxLayout(self.main_widget)
        self.image_labels = []
        self.combo_labels = []
        self.oriLabel=None   
        self.salLabel=None   


        

        #＝＝＝＝＝＝＝＝＝＝＝Change to working directory＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        self.basePath="Path"      



        self.ImgPath=[self.basePath+"/SourceData/GeneratedImg1/",
                      self.basePath+"/SourceData/GeneratedImg2/",
                      self.basePath+"/SourceData/GeneratedImg3/",
                      self.basePath+"/SourceData/GeneratedImg4/",
                      ]
        self.labelPath=self.basePath+"/SourceData/ClasssLabel/"


        self.savePath=self.basePath+"/RankingResults/"
        self.saveSalPath=self.basePath+"/RankingResults/SemanticLabel/"
        self.saveLabelPath=self.basePath+"/RankingResults/ClasssLabel/"

        label = QLabel(self)
        label.setFixedSize(256, 256)
        self.image_labels.append(label)
        self.v_layout1 = QVBoxLayout()
        self.v_layout1.addWidget(label)

        self.edit_button2 = QPushButton("Flip", self)
        self.edit_button2.setFixedSize(QSize(256, 40))
        self.v_layout1.addWidget(self.edit_button2)


        self.h_layout.addLayout(self.v_layout1)

        for i in range(4):
            label = QLabel(self)
            label.setFixedSize(256, 256)
            self.image_labels.append(label)
            self.v_layout1 = QVBoxLayout()
            self.v_layout1.addWidget(label)
            combo = QComboBox()
            combo.addItem("1")
            combo.addItem("2")
            combo.addItem("3")
            combo.addItem("4")
            # combo.setStyleSheet("QComboBox QAbstractItemView::item { align: center; }")
            combo.setItemData(0, Qt.AlignCenter, Qt.TextAlignmentRole)
            combo.setItemData(1, Qt.AlignCenter, Qt.TextAlignmentRole)
            combo.setItemData(2, Qt.AlignCenter, Qt.TextAlignmentRole)
            combo.setItemData(3, Qt.AlignCenter, Qt.TextAlignmentRole)
            combo.setFixedSize(QSize(256, 40))
            self.combo_labels.append(combo)
            self.v_layout1.addWidget(combo)
            self.h_layout.addLayout(self.v_layout1)


        self.v_layout = QVBoxLayout()


        self.open_button = QPushButton("Open Label", self)
        self.previous_button = QPushButton("Previous", self)
        self.next_button = QPushButton("Next", self)
        self.edit_button = QPushButton("Edit", self)
        self.save_button = QPushButton("Save File", self)
        self.close_button = QPushButton("Close", self)

        self.open_button.setFixedSize(QSize(100, 40))
        self.previous_button.setFixedSize(QSize(100, 40))
        self.next_button.setFixedSize(QSize(100, 40))
        self.edit_button.setFixedSize(QSize(100, 40))
        self.save_button.setFixedSize(QSize(100, 40))
        self.close_button.setFixedSize(QSize(100, 40))


        self.v_layout.addWidget(self.open_button)
        self.v_layout.addWidget(self.previous_button)
        self.v_layout.addWidget(self.next_button)
        self.v_layout.addWidget(self.save_button)
        self.v_layout.addWidget(self.edit_button)
        self.v_layout.addWidget(self.close_button)


        self.open_button.clicked.connect(self.open_file)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.save_button.clicked.connect(self.save_file)
        self.close_button.clicked.connect(self.close)
        self.edit_button2.clicked.connect(self.updown)
        self.edit_button.clicked.connect(self.editLabel)
    
        self.h_layout.addLayout(self.v_layout)

        self.image_files = []
        self.image_name = []
        self.current_image_index = -1

    def drawing(self,event, x, y, flags, param):
        # change the color according to the class
        r = 255
        g = 255
        b = 255
        color = (b, g, r)
        thickness = 5
        global ix, iy, draw_flag
        
        if event == cv2.EVENT_LBUTTONDOWN:
            draw_flag = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if draw_flag:

                cv2.circle(self.salLabel, (x, y), thickness, color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            draw_flag == False

    def open_file(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Open Label", ".", QFileDialog.DontUseNativeDialog)
        if dir_path:
            self.image_files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith('.png')]
            self.image_name = [file_name for file_name in os.listdir(dir_path)]
            self.current_image_index = -1
            self.next_image()

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.show_images()

    def next_image(self):
        if self.current_image_index + 1 < len(self.image_files):
            self.current_image_index += 1
            self.show_images()

    def show_images(self):
        file_path = self.image_files[self.current_image_index]
        orilabel_path = self.image_name[self.current_image_index]
        self.oriLabel=cv2.imread(self.labelPath+orilabel_path)
        self.salLabel = cv2.imread(file_path)
        self.salLabel = cv2.resize(self.salLabel, (256, 256))

        height, width,channel = self.salLabel.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.salLabel.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(256, 256)
        self.image_labels[0].setPixmap(pixmap)

        self.image_name
        file_path = self.image_name[self.current_image_index]
        file_path=self.ImgPath[0]+file_path[:-4]+'.jpg'
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(256, 256)
        self.image_labels[1].setPixmap(pixmap)

        file_path = self.image_name[self.current_image_index]
        file_path=self.ImgPath[1]+file_path[:-4]+'.jpg'
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(256, 256)
        self.image_labels[2].setPixmap(pixmap)

        file_path = self.image_name[self.current_image_index]
        file_path=self.ImgPath[2]+file_path[:-4]+'.jpg'
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(256, 256)
        self.image_labels[3].setPixmap(pixmap)

        file_path = self.image_name[self.current_image_index]
        file_path=self.ImgPath[3]+file_path[:-4]+'.jpg'
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(256, 256)
        self.image_labels[4].setPixmap(pixmap)


    def save_file(self):
        if self.current_image_index >= 0 and self.current_image_index < len(self.image_files):
            file_path = self.image_name[self.current_image_index]
            nums = ['1', '2', '3', '4'] 
            lst=[]

            index1=self.combo_labels[0].currentText()
            index2=self.combo_labels[1].currentText()
            index3=self.combo_labels[2].currentText()
            index4=self.combo_labels[3].currentText()

            lst.append(index1)
            lst.append(index2)
            lst.append(index3)
            lst.append(index4)

            with open("./RankingResults/HunmanRanking.txt",'a') as file:
                file.write(file_path+' ')
                file.write(' '.join(lst))
                file.write('\n')

            file_path1=self.ImgPath[0]+file_path[:-4]+'.jpg'
            file_path2=self.ImgPath[1]+file_path[:-4]+'.jpg'
            file_path3=self.ImgPath[2]+file_path[:-4]+'.jpg'
            file_path4=self.ImgPath[3]+file_path[:-4]+'.jpg'

            if set(lst) == set(nums) and len(lst) == len(set(lst)):
                shutil.copy(file_path1, self.savePath+str(index1))
                shutil.copy(file_path2, self.savePath+str(index2))
                shutil.copy(file_path3, self.savePath+str(index3))
                shutil.copy(file_path4, self.savePath+str(index4))

                cv2.imwrite(self.saveSalPath+file_path[:-4]+'.png',self.salLabel)


                classes = np.unique(self.oriLabel)
                self.salLabel = cv2.cvtColor(self.salLabel, cv2.COLOR_BGR2GRAY)
                self.salLabel[np.where(self.salLabel>0)]=classes[1]
                cv2.imwrite(self.saveLabelPath+file_path[:-4]+'.png',self.salLabel)
            else:
                reply = QMessageBox.question(self, 'Notice', 'Error', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            self.next_image()



    def updown(self):
        self.salLabel = cv2.flip(self.salLabel, 0)
        self.oriLabel = cv2.flip(self.oriLabel, 0)
        height, width,channel = self.salLabel.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.salLabel.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        self.image_labels[0].setPixmap(pixmap)


    def leftright(self):
        self.salLabel = cv2.flip(self.salLabel, 1)
        self.oriLabel = cv2.flip(self.oriLabel, 1)
        height, width,channel = self.salLabel.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.salLabel.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        self.image_labels[0].setPixmap(pixmap)



    def editLabel(self):
        cv2.namedWindow("EditImage")
        cv2.imshow("EditImage",self.salLabel)
        cv2.setMouseCallback("EditImage", self.drawing)
        classes = np.unique(self.oriLabel)
        while (1):
            cv2.imshow('EditImage', self.salLabel)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                height, width,channel = self.salLabel.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.salLabel.data, width, height, bytesPerLine, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qImg)
                self.image_labels[0].setPixmap(pixmap)
                self.salLabel[np.where(self.salLabel==0)]=classes[1]
                break
        cv2.destroyWindow("EditImage")


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Notice', 'Save current image?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            pass
            # self.save_file()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication([])
    window = ImageViewer()
    window.show()
    app.exec_()