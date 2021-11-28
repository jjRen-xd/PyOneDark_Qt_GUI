# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/utils/signal_vis.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/15
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 可视化信号输入
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/config.py
                    <1> PATH_TOOT/dataset/RML2016.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> drawAllOriSignal():
                        绘制所有信号输入样本的图像，并保存至相应标签的文件夹下
                    <1> showOriSignal():
                        绘制并展示一个样本信号的图像
                    <2> showImgSignal():
                        绘制并展示一个信号样本的二维可视化图像
                    <3> showCamSignal():
                        叠加信号与CAM图，可视化CAM解释结果，并按类型保存
                    <4> mask_image():
                        软阈值擦除CAM对应的判别性特征区域 
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/15 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/09 | 优化无name的数据集调用问题
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <2> | JunJie Ren |   v1.2    | 2020/07/13 |     增加CAM阈值擦除函数
--------------------------------------------------------------------------
'''

import sys
import os

import cv2
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from sklearn.metrics import confusion_matrix

sys.path.append("../")
from app.configs import cfgs
from app.dataset.RML2016 import loadNpy
# from app.dataset.RML2016_04c.classes import modName

def t2n(t):
    return t.detach().cpu().numpy().astype(np.int)


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def showOriSignal(sample, mod_name, idx):
    ''' 绘制并展示一个样本信号的图像 '''
    signal_data = sample[0]

    figure = plt.figure(figsize=(9, 6))
    plt.title(str(idx)+" "+str(mod_name), fontsize=30)
    plt.xlabel('N', fontsize=20)
    plt.ylabel("Value", fontsize=20)

    plt.plot(signal_data[:, 0], label = 'I', linewidth=2.0)
    plt.plot(signal_data[:, 1], color = 'red', label = 'Q', linewidth=2.0)
    plt.legend(loc="upper right", fontsize=30)
    plt.close()
    image = fig2data(figure)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def showCamSignal(signal, CAM, mod):
    """
    Args:
        signal: numpy.ndarray(size=(1, 128, 2), dtype=np.float)
        CAM: numpy.ndarray(size=(128, 2), dtype=np.float)
    Funcs: 
        叠加信号与CAM图，可视化CAM解释结果，并按类型保存
    """
    # 绘制信号

    signal_data = signal[0]
    sig_len, channel = signal_data.shape

    figure = plt.figure(figsize=(18, 12))
    plt.title(mod, fontsize=26)
    plt.xlabel('N', fontsize=20)
    plt.ylabel("Value", fontsize=20)

    plt.plot(signal_data[:, 0]*(sig_len//10), label = 'I' ,linewidth=4.0)
    plt.plot(signal_data[:, 1]*(sig_len//10), color = 'red', label = 'Q', linewidth=4.0)
    plt.legend(loc="upper right", fontsize=26)
    
    # 绘制CAM
    sig_min, sig_max = np.min(signal_data), np.max(signal_data)
    
    CAM = CAM.T                      # (2, 128)
    CAM = CAM - np.min(CAM)
    CAM = CAM / np.max(CAM)          # CAM取值归一化

    plt.imshow(CAM, cmap='jet', extent=[0., sig_len, (sig_min-0.5)*(sig_len//10), (sig_max+0.5)*(sig_len//10)])     # jet, rainbow
    # plt.colorbar()
    '''
    save_path = "figs_CAM_ACARS/{}".format(mod_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + str(idx+1)+"_CAM")
    plt.close()
    '''
    # plt.savefig("figs/CAM_cur")
    # plt.show()
    image = fig2data(figure)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.close()
    return image


def plot_confusion_matrix(y_true, y_pred, labels, title='Normalized confusion matrix', intFlag = 0):
    ''' 绘制混淆矩阵 '''
    cmap = plt.cm.Blues
    ''' 颜色参考http://blog.csdn.net/haoji007/article/details/52063168'''
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    if cm.sum(axis=1)[:, np.newaxis].all() != 0:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        intFlag = 1
    figure = plt.figure(figsize=(10, 9), dpi=360)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    # intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=12, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.0001):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c*100,) + "%", color='red', fontsize=10, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('Confusion Matrix', fontsize=18)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('./app/figs/confusion_matrix.jpg', dpi=300)

    image = fig2data(figure)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
    # plt.title(title)
    # plt.show()


def drawAllOriSignal(X, Y):
    """
    Args:
        X: numpy.ndarray(size = (bz, 1, 128, 2)), 可视化信号原始数据
        Y: numpy.ndarray(size = (bz,)), 可视化信号标签
    Returns:
        None
    Funcs:
        绘制所有信号输入样本的图像，并保存至相应标签的文件夹下
    """
    for idx in range(len(X)):
        if (idx+1)%50 == 0:
            print("{} complete!".format(idx+1))
        signal_data = X[idx][0]
        mod_name = str(modName[Y[idx]], "utf-8") \
            if cfgs.dataset_name == "RML2016.04c" else "label-"+str(t2n(Y[idx]))

        plt.figure(figsize=(6, 4))
        plt.title(mod_name)
        plt.xlabel('N')
        plt.ylabel("Value")

        plt.plot(signal_data[:, 0], label = 'I')
        plt.plot(signal_data[:, 1], color = 'red', label = 'Q')
        plt.legend(loc="upper right")

        save_path = "../figs/original_signal/{}".format(mod_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(idx+1))
        plt.close()
        
    print(X.shape)
    print(Y.shape)
    print("Complete the drawing of all original signals !!!")


def showImgSignal(sample, label):
    ''' 绘制并展示一个信号样本的二维可视化图像 '''
    data = sample[0].T                      # 2*128
    data = data - np.min(data)
    data = data / np.max(data)
    mod_name = str(modName[label], "utf-8")\
            if cfgs.dataset_name == "RML2016.04c" else "label-"+str(t2n(label))
    # print(data.shape)
    h, sig_len = data.shape

    # 叠加信号，以便显示
    img_sig = np.empty([sig_len, sig_len], dtype = float)
    # for row in range(int(sig_len/h)):
    #     img_sig[row*h:row*h+h, :] = data
    for row in range(sig_len):
        if row<sig_len/2:
            img_sig[row:row+1, :] = data[0]
        else:
            img_sig[row:row+1, :] = data[1]
    img_sig = cv2.resize(img_sig, (sig_len*2,sig_len*2))
    cv2.imshow(mod_name, img_sig)
    cv2.waitKey(0)
    return img_sig




def mask_image(cam, image, reserveORerase):
    """
    Args:
        cam: numpy.ndarray(size=(4096, 2), dtype=np.float), 0-1
        image: torch.Tensor, torch.Size([1, 4096, 2])
        reserveORerase: bool 保留(0)或擦除(1)判别性区域
    Funcs: 
        软阈值擦除/保留CAM对应的判别性特征区域
    """
    cam = torch.from_numpy(cam).cuda()
    mask = torch.sigmoid(cfgs.CAM_omega * (cam - cfgs.Erase_thr)).squeeze()
    masked_image = image - (image * mask)  if reserveORerase else image * mask 

    return masked_image.float()


def mask_image_hard(cam, image, reserveORerase, thr):
    ''' 阈值硬擦除 '''
    mask = np.zeros_like(cam)
    mask[cam >= thr] = 1
    mask[cam < thr] = 0
    mask = torch.from_numpy(mask).cuda()
    # print(mask.shape, image.shape)
    masked_image = image - (image * mask)  if reserveORerase else image * mask
    return masked_image.float()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = loadNpy(cfgs.train_path, cfgs.test_path)
    print(x_train.shape, y_train.shape)
    # drawAllOriSignal(X=x_train, Y=y_train)
    for idx in range(len(x_train)):
        showImgSignal(x_train[idx], y_train[idx])
        showOriSignal(x_train[idx], y_train[idx])

