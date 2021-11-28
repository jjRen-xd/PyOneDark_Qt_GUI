# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA and Junjie Ren
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# QT CORE
# Change for PySide Or PyQt
# ///////////////////////// WARNING: ////////////////////////////
# Remember that changing to PyQt too many modules will have 
# problems because some classes have different names like: 
# Property (pyqtProperty), Slot (pyqtSlot), Signal (pyqtSignal)
# among others.
# ///////////////////////////////////////////////////////////////
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtSvgWidgets import *


import cv2
def cv2QPix(img, resize = 1):
    # cv 图片转换成 qt图片

    # img = cv2.resize(img, (1000,1000))
    # print(img.shape)
    if resize:
        img = img[200:3200, 0:3200]
    # cv2.imshow("d", img)
    # cv2.waitKey(3000)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qt_img = QImage(img.data,  # 数据源
                            img.shape[1],  # 宽度
                            img.shape[0],  # 高度
                            img.shape[1] * 3,  # 行字节数
                            QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img)