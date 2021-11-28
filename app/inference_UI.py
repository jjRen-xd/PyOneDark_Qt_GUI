# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/inference_UI.py
# Author:           JunJie Ren
# Version:          v1.2
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 测试模型分类准确率，并绘制混淆矩阵, 使用pyqt绘制界面
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/configs.py
                    <1> PATH_ROOT/dataset/RML2016.py
                    <3> PATH_ROOT/utils/strategy.py;plot.py
                    <4> PATH_ROOT/dataset/ACARS.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> inference():
                        -- 使用训练好的模型对测试集进行推理，测试分类准确率，
                        绘制混淆矩阵
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2021/06/14 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <1> | JunJie Ren |   v1.1    | 2021/07/09 |    新增ACARS测试程序选项
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <1> | JunJie Ren |   v1.2    | 2021/11/24 |         QT界面显示
--------------------------------------------------------------------------
'''

import os
import sys
from numpy.lib.function_base import insert
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data


from app.dataset.RML2016 import RMLDataset, loadNpy
from app.dataset.ACARS import ACARSDataset, loadNpy_acars
from app.utils.strategy import accuracy
from app.utils.CAM import compute_gradcampp, compute_Sigcam, t2n
from app.utils.signal_vis import showOriSignal,showCamSignal,plot_confusion_matrix
# 强化学习
from app.checkpoints.reinforce import conv_net
from app.checkpoints.reinforce.dataplot import dataharr,RL_dataloader

import cv2
import numpy as np
import random


class DefaultConfigs_RML(object):
    '''
    Func:
        默认配置参数
    # TODO
    '''
    def __init__(self) -> None:
        super().__init__()
        ''' 默认参数配置 '''
        # Dataset
        self.dataset_name = "RML2016.04c" 
        self.num_classes = 0                            # 分类类别数
        self.signal_len = "128,2"
        
        self.train_path = 'None'     # 原始训练数据目录
        self.test_path = 'None'      # 原始测试数据目录
        
        self.process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理
        self.batch_size = 4                            # DataLoader中batch大小，550/110=5 Iter
        self.num_workers = 4                             # DataLoader中的多线程数量

        # model
        self.model = "MsmcNet_RML2016"                   # 指定模型，目前就一个
        self.checkpoint_name = 'None' # 训练完成的模型名

        self.CHECKPOINT_PATH_DIR = "./app/checkpoints"
        self.DATASET_PATH_DIR = "./app/dataset/RML2016_04c"

        self.MOD_NAME = {
            "transfer": [b'QPSK', b'AM-DSB', b'AM-SSB', b'QAM64', b'WBFM'],
            "reinforce": [b'GFSK',b'WBFM',b'AM-SSB',b'AM-DSB',b'QPSK',b'QAM16',b'CPFSK',b'BPSK',b'PAM4',b'QAM64',b'8PSK'],
            "optimize": [b'GFSK',b'WBFM',b'AM-SSB',b'AM-DSB',b'QPSK',b'QAM16',b'CPFSK',b'BPSK',b'PAM4',b'QAM64',b'8PSK'],
            "interpretability": [b'GFSK',b'WBFM',b'AM-SSB',b'AM-DSB',b'QPSK',b'QAM16',b'CPFSK',b'BPSK',b'PAM4',b'QAM64',b'8PSK'],
        }


    def get_modName(self, menu):
        return self.MOD_NAME[menu]

    def get_path(self, path_type, menu, method = "None", shot_num = -1):
        '''
            path_type = "checkpoint" or "dataset";
            menu = "optimize" or "transfer" or "reinforce" or "interpretability"
        '''
        try:
            if path_type == "checkpoint":
                # 优化
                if menu == "optimize":
                    pth_path = f"{self.CHECKPOINT_PATH_DIR}/{menu}/{method}/MsmcNet_RML2016_04c_6dB_{shot_num}.pth"
                    insert_path = f"{self.CHECKPOINT_PATH_DIR[1:]}/{menu}/{method}/"
                    return pth_path, insert_path
                elif menu  == "transfer":
                    pth_path = f"{self.CHECKPOINT_PATH_DIR}/{menu}/{method}/best_{shot_num}.pth"
                    insert_path = f"{self.CHECKPOINT_PATH_DIR[1:]}/{menu}/"
                    return pth_path, insert_path
                elif menu  == "reinforce":
                    pth_path = f"{self.CHECKPOINT_PATH_DIR}/{menu}/model/{method}.pth"
                    insert_path = f"{self.CHECKPOINT_PATH_DIR[1:]}/{menu}/"
                    return pth_path, insert_path
                elif menu  == "interpretability":
                    pth_path = f"{self.CHECKPOINT_PATH_DIR}/MsmcNet_{method}.pth"
                    insert_path = "/app/"
                    return pth_path, insert_path

            elif path_type == "dataset":
                if menu == "optimize":
                    train_path = f"{self.DATASET_PATH_DIR}/{menu}/6dB-SNR_{shot_num}-samples.npy"
                    test_path = f"{self.DATASET_PATH_DIR}/{menu}/6dB-SNR_815-test.npy"
                    return train_path, test_path
                elif menu  == "transfer":
                    train_path = f"{self.DATASET_PATH_DIR}/{menu}/6dB-SNR_tran_{shot_num}-samples.npy"
                    test_path = f"{self.DATASET_PATH_DIR}/{menu}/6dB-SNR_tran-test.npy"
                    return train_path, test_path
                elif menu  == "reinforce":
                    test_path = f"{self.DATASET_PATH_DIR}/{menu}/stars_haar_level2.h5"
                    return test_path
                elif menu  == "interpretability":
                    train_path = f"{self.DATASET_PATH_DIR}/6dB-SNR_50-samples.npy"
                    test_path = f"{self.DATASET_PATH_DIR}/6dB-SNR_4055-test.npy"
                    return train_path, test_path
        except ValueError:
            print("Error: Path error! <path_type> is 'checkpoint' or 'dataset'.")

cfgs = DefaultConfigs_RML()


def showDataset(menu, idx_range, img_num):
    '''
    Func:
        返回对相应方法的数据集中的信号图像
    Args：
        menu (str): 方法名, e.g.: "optimize","transfer","reinforce","interpretability"
        idx_range (str list): 要绘制的信号索引范围, e.g.: ["1", "500"]
        img_num (int): 要绘制的信号数量
    Returns：
        raw_images (cv2::Mat list): 绘制的信号图像,长度为img_num, e.g.: [Mat, Mat, ...]
    '''
    # 数据集路径及调制类型获取
    train_path, test_path = cfgs.get_path("dataset", menu, shot_num=10)
    modName = cfgs.get_modName(menu)
    # 测试区间获取
    start_idx = int(idx_range[0])-1
    end_idx = int(idx_range[1])-1
    # 载入数据
    _, _, x_test, y_test = loadNpy(
        train_path,
        test_path,
        modName,
        cfgs.process_IQ
    )
    # 随机取N个样本绘制并返回
    raw_images = []
    show_idx = random.sample(range(start_idx, end_idx), img_num)
    for idx in show_idx:
        raw_image = showOriSignal(x_test[idx], modName[y_test[idx]], idx)
        raw_images.append(raw_image)

    # cv2::Mat
    return raw_images


def inference_OP_TL(menu, method, dataset, idx_range, shot_num, infer_mode = 0, mat_norm = 1):
    # 数据集路径及调制类型获取
    train_path, test_path = cfgs.get_path("dataset", menu, shot_num=10)
    modName = cfgs.get_modName(menu)
    model_path, insert_path = cfgs.get_path("checkpoint", menu, method, shot_num)

    # model
    # 扯淡的路径问题！！！
    sys.path.insert(0, os.getcwd()+insert_path)  # 保证Pytorch能够找到模型所在的原始目录
    model = torch.load(model_path)               # load checkpoint
    sys.path.remove(os.getcwd()+insert_path)
    print(model)


    model.cuda()

    # Dataset
    if dataset == "RML2016.04c":
        _, _, x_test, y_test = loadNpy(
            train_path,
            test_path,
            modName,
            cfgs.process_IQ
        )
        Dataset = RMLDataset
    else :
        print('ERROR: No Dataset {}!!!'.format(cfgs.model))

    # 测试区间获取
    start_idx = int(idx_range[0])-1
    end_idx = int(idx_range[1])-1
    x_test = x_test[start_idx:end_idx+1, :, :, :]
    y_test = y_test[start_idx:end_idx+1]

    # Valid data
    # BUG,BUG,BUG,FIXME
    transform = transforms.Compose([])

    valid_dataset = Dataset(x_test, y_test, transform=transform)
    dataloader_valid = DataLoader(valid_dataset, \
                                batch_size=cfgs.batch_size, \
                                num_workers=cfgs.num_workers, \
                                shuffle=True, \
                                drop_last=False)
    sum = 0
    val_top1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_valid:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda().float()
        target = Variable(label).cuda()
        if infer_mode == 0:
            output = model(input)
        elif infer_mode == 1:
            output,_ = model(input)
        elif infer_mode == 2:
            _,output = model(input)

        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
    avg_top1 = (val_top1_sum / sum).cpu().numpy()[0]
    print('acc: {}'.format(avg_top1))

    confusion_matrix = plot_confusion_matrix(labels, preds, modName, intFlag = not mat_norm)
    return avg_top1, confusion_matrix


def inference_RL(method, idx_range, mat_norm = 1):
    if method == "stars":
        obs = np.array([1, 0, 0, 0, 0, 0, 0])
    elif method == "stars_haar":
        obs = np.array([1, 1, 1, 1, 1, 1, 1])
    elif method == "stars_haar_reinforcement":
        obs = np.array([1, 0, 0, 0, 1, 1, 0])
    checkpoint_path, insert_path = cfgs.get_path("checkpoint", "reinforce", method)
    dataset_path = cfgs.get_path("dataset", "reinforce")
    modName = cfgs.get_modName("reinforce")
    
    _, _, x_test, y_test = RL_dataloader(obs, dataset_path)


    # 测试区间获取
    start_idx = int(idx_range[0])-1
    end_idx = int(idx_range[1])-1
    x_test = x_test[start_idx:end_idx+1, :, :]
    y_test = y_test[start_idx:end_idx+1, :]
    print(x_test.shape, y_test.shape)

    ''' 强化学习推理 '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = conv_net.conv_net_harr(11,int(sum(obs))).to(device)

    # 扯淡的路径问题！！！
    sys.path.insert(0, os.getcwd()+insert_path)  # 保证Pytorch能够找到模型所在的原始目录
    model = torch.load(checkpoint_path)
    sys.path.remove(os.getcwd()+insert_path)
    print(model)


    model.eval()
    test_label,test_data = torch.from_numpy(y_test).long(), torch.from_numpy(x_test).float()
    test_label = torch.argmax(test_label, -1)

    torch_test_dataset = Data.TensorDataset(test_data,test_label)

    test_loader = Data.DataLoader(
        dataset = torch_test_dataset,
        batch_size = 256,
        shuffle = True,
    )

    testing_correct = 0
    labels_CM = []
    preds_CM = []
    for _,(images,labels) in enumerate(test_loader):
        labels_CM += labels.numpy().tolist()
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == labels)
        preds_CM += pred.t().cpu().numpy().tolist()

    testing_correct = testing_correct.cpu().numpy()
    test_label = test_label.cpu().numpy()
    acc = testing_correct/len(test_label)

    confusion_matrix = plot_confusion_matrix(labels_CM, preds_CM, modName, intFlag = not mat_norm)
    print(acc)
    return acc*100, confusion_matrix


def nextRLdata(method, dataChoice):
    if method == "stars":
        obs = np.array([1, 0, 0, 0, 0, 0, 0])
    elif method == "stars_haar":
        obs = np.array([1, 1, 1, 1, 1, 1, 1])
    elif method == "stars_haar_reinforcement":
        obs = np.array([1, 0, 0, 0, 1, 1, 0])
    dataset_path = cfgs.get_path("dataset", "reinforce")
    _, root_path = cfgs.get_path("checkpoint", "reinforce")

    label1,label2 = dataharr(root_path, dataset_path, dataChoice, obs)


    starImg = cv2.imread('.'+root_path+'img/star.png')
    realImg = cv2.imread('.'+root_path+'img/real.png')
    virtualImg = cv2.imread('.'+root_path+'img/virtual.png')

    return (label1, label2),(starImg, realImg, virtualImg)

    

def showCAM(dataset, layer = 4, imgLen = 4):
    train_path,test_path = cfgs.get_path("dataset","interpretability")
    checkpoint_name, insert_path = cfgs.get_path("checkpoint", "interpretability", dataset)
    modName = cfgs.get_modName("interpretability")
    # model
    # 扯淡的路径问题！！！
    sys.path.insert(0, os.getcwd()+insert_path)  # 保证Pytorch能够找到模型所在的原始目录
    model = torch.load(checkpoint_name)    # load checkpoint
    sys.path.remove(os.getcwd()+insert_path)

    print(model)
    model.cuda()
    
   # Dataset
    if dataset == "RML2016.04c":
        _, _, x_test, y_test = loadNpy(
            train_path,
            test_path,
            modName,
            cfgs.process_IQ
        )
        Dataset = RMLDataset
    elif dataset == "ACARS":
        x_train, y_train, x_test, y_test = loadNpy_acars(
            cfgs.train_path_x,
            cfgs.train_path_y,
            cfgs.test_path_x,
            cfgs.test_path_y,
            cfgs.process_IQ
        )
        Dataset = ACARSDataset
    else :
        print('ERROR: No Dataset {}!!!'.format(dataset))
        
    # Valid data
    # BUG,BUG,BUG,FIXME
    transform = transforms.Compose([ 
                                        # transforms.ToTensor()
                                        # waiting add
                                    ])
    valid_dataset = Dataset(x_test, y_test, transform=transform)
    dataloader_valid = DataLoader(valid_dataset, \
                                batch_size=cfgs.batch_size, \
                                num_workers=cfgs.num_workers, \
                                shuffle=True, \
                                drop_last=False)
    idx = 0

    return_image = [ [] for i in range(len(modName))]
    for images, labels in dataloader_valid:
        images = Variable(images).cuda().float()
        labels = Variable(labels).cuda()

        # Grad-CAM++
        # cams, _, pred_labels = compute_gradcampp(images, labels, model, gt_known = True)     # (bz, 128, 2)

        # 不同层的Grad-CAM++
        vis_layers = [model.Block_1[2], model.Block_2[2], model.Block_3[2], model.Block_4[2]]
        _, cams, _, _, pred_labels = compute_Sigcam(images, labels, model, vis_layers[layer-1], gt_known = True)     # (bz, 128, 2)
        
        for image, cam, label, pred_label in zip(images, cams, labels, pred_labels):
            if cam.max() == 0:
                print("CAM is 0, break", label, pred_label)
                continue
            idx += 1
            # showOriSignal(t2n(image), label, idx)             # 绘制原始信号图
            cur_CAM = showCamSignal(t2n(image), cam, modName[label])          # 绘制原始信号与CAM叠加图
            
            if len(return_image[label]) < imgLen:
                print(idx)
                return_image[label].append(cur_CAM)

        return_flag = 1
        for i in range(len(modName)):
            if len(return_image[i]) != imgLen:
                return_flag = 0
        if return_flag:
            break
    return return_image

