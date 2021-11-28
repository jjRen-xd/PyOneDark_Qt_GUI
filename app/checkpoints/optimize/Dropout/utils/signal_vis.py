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

sys.path.append("../")
from configs import cfgs
from dataset.RML2016 import loadNpy
from dataset.RML2016_10a.classes import modName

def t2n(t):
    return t.detach().cpu().numpy().astype(np.int)

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


def showOriSignal(sample, label, idx):
    ''' 绘制并展示一个样本信号的图像 '''
    mod_name = str(modName[label], "utf-8")\
            if cfgs.dataset_name == "RML2016.04c" else "label-"+str(t2n(label))
    signal_data = sample[0]

    plt.figure(figsize=(18, 12))
    plt.title(mod_name)
    plt.xlabel('N')
    plt.ylabel("Value")

    plt.plot(signal_data[:, 0], label = 'I')
    plt.plot(signal_data[:, 1], color = 'red', label = 'Q')
    plt.legend(loc="upper right")

    # save_path = "figs_CAM_ACARS/{}".format(mod_name)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # plt.savefig(save_path + '/' + str(idx+1))
    # plt.close()

    plt.savefig("ori")
    plt.show()
    plt.close()


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

def showCamSignal(signal, CAM, label, idx):
    """
    Args:
        signal: numpy.ndarray(size=(1, 128, 2), dtype=np.float)
        CAM: numpy.ndarray(size=(128, 2), dtype=np.float)
    Funcs: 
        叠加信号与CAM图，可视化CAM解释结果，并按类型保存
    """
    # 绘制信号
    mod_name = str(modName[label], "utf-8")\
            if cfgs.dataset_name == "RML2016.04c" else "label-"+str(t2n(label))
    signal_data = signal[0]
    sig_len, channel = signal_data.shape

    plt.figure(figsize=(18, 12))
    plt.title(mod_name)
    plt.xlabel('N')
    plt.ylabel("Value")

    plt.plot(signal_data[:, 0]*(sig_len//10), label = 'I')
    plt.plot(signal_data[:, 1]*(sig_len//10), color = 'red', label = 'Q')
    plt.legend(loc="upper right")
    
    # 绘制CAM
    sig_min, sig_max = np.min(signal_data), np.max(signal_data)
    
    CAM = CAM.T                      # (2, 128)
    # print(np.min(CAM), np.max(CAM))
    CAM = CAM - np.min(CAM)
    CAM = CAM / np.max(CAM)          # CAM取值归一化

    plt.imshow(CAM, cmap='jet', extent=[0., sig_len, (sig_min-1)*(sig_len//10), (sig_max+1)*(sig_len//10)])     # jet, rainbow
    # plt.colorbar()
    '''
    save_path = "figs_CAM_ACARS/{}".format(mod_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + str(idx+1)+"_CAM")
    plt.close()
    '''
    # plt.savefig("cam")
    plt.show()
    plt.close()


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

