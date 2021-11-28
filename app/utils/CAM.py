# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/utils/signal_vis.py
# Author:           JunJie Ren
# Version:          v1.4
# Created:          2021/06/15
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 利用GradCAM++可视化技术，解释网络隐层特征
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/15 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/01 | 更新CAM计算逻辑，支持batch
                                             | 新增gt-known与top1计算选项
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <2> | JunJie Ren |   v1.2    | 2020/07/14 | 优化pred_scores计算方式
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <3> | JunJie Ren |   v1.3    | 2020/07/16 |   修复计算CAM的致命bug
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <4> | JunJie Ren |   v1.4    | 2020/07/24 | 新增SigCAM的计算, 失败
--------------------------------------------------------------------------
'''


import os

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from app.configs import cfgs

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

class SaveValues():
    """
        后期新增，记录中间反传梯度
    """
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

import time
def compute_gradcampp(images, labels, model, top1 = False, gt_known = True):
    """
    Args:
        images: torch.Size([bz, 1, 128, 2]), 将要送入模型的一个batch数据
        model: PyTorch标准模型
        top1: 计算预测得分的最高类的CAM; gt-konwn: 计算实际标签类别的CAM
    Returns:
        CAMs: list(bz, 128, 2), 输入batch数据所对应的CAMs解释
        pred_scores: torch.Size([20])该batch下所输入labels的得分
        pred_labes: torch.Size([20])该batch下所真实预测的lables
    Funcs:
        根据一个batch的输入，计算该batch图像对应的CAM
    """
    # FIXME Done! 对一个batch的得分进行反向传播，并hoock倒数, 计算一个batch的gradcampp 
    # FIXME BUG BUG BUG CAM存在全为0的情况
    ''' 修复不能计算batch的问题 '''
    _, _, h, w = images.shape

    images = images.cuda()
    # 指定需要可视化的一层，并hook参数及倒数
    if cfgs.model == "MsmcNet_RML2016":
        target_layer = model.Block_4[3]
    elif cfgs.model == "MsmcNet_ACARS":
        target_layer = model.Block_7[3]
    # target_layer = model.layer4[2].bn3
    print("target_layer:", target_layer)
    hook_values = SaveValues(target_layer)

    # 前向传播计算每个类别的score,并hook特征图,并计算真实预测的labels
    logits = model(images)      # torch.Size([bz, 11])
    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, labels]
    else: 
        print("Error in indicator designation!!!")
        exit()
    pred_labels = logits.argmax(dim = 1)

    ''' 修复不能计算batch的问题 '''
    # 1. 反向传播计算并hook梯度
    model.zero_grad()                          
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)
    activations = hook_values.activations           # ([bz, 15, 5, 1])
    gradients = hook_values.gradients               # ([bz, 15, 5, 1])
    bz, nc, _, _ = activations.shape                # (batch_size, num_channel, height, width)

    # 2. 计算梯度图中每个梯度的权重alpha
    numerator = gradients.pow(2)
    denominator = 2 * gradients.pow(2)
    ag = activations * gradients.pow(3)
    denominator += ag.view(bz, nc, -1).sum(-1, keepdim=True).view(bz, nc, 1, 1)
    denominator = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator)
    )
    alpha = numerator / (denominator + 1e-7)        # ([bz, 15, 5, 1])

    # 3. 计算梯度图权重weights
    Y = pred_scores.exp().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # ([bz, 15, 5, 1])
    Y_grad = F.relu(Y * gradients)
    weights = (alpha * Y_grad).view(bz, nc, -1).sum(-1).view(bz, nc, 1, 1)   # ([bz, 1024, 1, 1])

    # 4. 计算一组batch的CAMs
    cams = (weights * activations).sum(1)            
    cams = t2n(F.relu(cams))                        # ([bz, 5, 1])
    # cams = t2n(cams)                              # ([bz, 5, 1])
    CAMs = []
    for idx, (cam, image) in enumerate(zip(cams, images)):
        '''
        # BUG 一些CAM不能计算的问题，1）CAM结果都为负，被ReLU抹除，2）都为0
        if cam.max() == 0:
        # if 1:
            print("cam:", cam)
            print("act:",activations.min(), activations.max())
            print("grad:", gradients.min(), activations.max())
            print("alphaJ:", alpha.min(), alpha.max())
            print("weight:", weights.min(), weights.max())
            print("numerator", numerator.min(), numerator.max())
            print("denominator", denominator.min(), denominator.max())
            print("weight*activate", (weights * activations).sum(1)[idx], cams[idx])
            exit()
        '''
        cam = cv2.resize(cam, (w, h),               # 上采样，基于4x4像素邻域的3次插值法
                                interpolation=cv2.INTER_CUBIC)
        cam_normalized = normalize_scoremap(cam)    # (224, 224)
        CAMs.append(cam_normalized)

    return CAMs, pred_scores, pred_labels


def compute_Sigcam(images, labels, model, vis_layer, top1 = False, gt_known = True):
    """
    Args:
        images: torch.Size([bz, 1, sigLen, 2]), 将要送入模型的一个batch数据
        model: PyTorch标准模型
        top1: 计算预测得分的最高类的CAM; gt-konwn: 计算实际标签类别的CAM
    Returns:
        CAMs: np.array(bz, 128, 2), 输入batch数据所对应的CAMs解释
        pred_scores: torch.Size([20])该batch下所输入labels的得分
        pred_labes: torch.Size([20])该batch下所真实预测的lables
    Funcs:
        根据一个batch的输入，计算该batch图像对应的CAM
    """
    # FIXME Done! 对一个batch的得分进行反向传播，并hock倒数, 计算一个batch的gradcampp 
    _, _, h, w = images.shape
    images = images.cuda()

    # 指定需要可视化的一层，并hook参数及倒数
    if cfgs.model == "MsmcNet_RML2016":
        target_layer = model.Block_4[3]
    elif cfgs.model == "MsmcNet_ACARS":
        target_layer = model.Block_7[3]
        # IQ_layer = model.Block_1[2]
    IQ_layer = vis_layer
    # print("target_layer:", target_layer, IQ_layer)
    hook_values = SaveValues(target_layer)
    hook_values_IQ = SaveValues(IQ_layer)

    # 前向传播计算每个类别的score,并hook特征图,并计算真实预测的labels
    logits = model(images)      # torch.Size([bz, 11])
    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, labels]
    else: 
        print("Error in indicator designation!!!")
        exit()
    pred_labels = logits.argmax(dim = 1)

    ''' 修复不能计算batch的问题 '''
    # 反向传播计算并hook梯度
    model.zero_grad()                          
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)
    
    CAMs_target = gradcampp(pred_scores, hook_values, (w, h))
    CAMs_IQ = gradcampp(pred_scores, hook_values_IQ, (w, h))

    # 保留IQ全部激活部分
    # CAMs_IQ = t2n(hook_values_IQ.gradients.sum(1))

    # 将两路的CAM与最终的CAM决策点乘，获得具有高细粒度的解释
    CAMs = CAMs_IQ * CAMs_target
    CAMs -= CAMs.min()
    CAMs /= CAMs.max()

    return CAMs_target, CAMs_IQ, CAMs, pred_scores, pred_labels


def gradcampp(scores, hook_values, cam_shape):
    # 1.获取指定的hook梯度与激活
    activations = hook_values.activations           # ([bz, 15, 5, 1])
    gradients = hook_values.gradients               # ([bz, 15, 5, 1])
    bz, nc, _, _ = activations.shape                # (batch_size, num_channel, height, width)

    # 2. 计算梯度图中每个梯度的权重alpha
    numerator = gradients.pow(2)
    denominator = 2 * gradients.pow(2)
    ag = activations * gradients.pow(3)
    denominator += ag.view(bz, nc, -1).sum(-1, keepdim=True).view(bz, nc, 1, 1)
    denominator = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator)
    )
    alpha = numerator / (denominator + 1e-7)        # ([bz, 15, 5, 1])

    # 3. 计算梯度图权重weights
    Y = scores.exp().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # ([bz, 15, 5, 1])
    Y_grad = F.relu(Y * gradients)
    weights = (alpha * Y_grad).view(bz, nc, -1).sum(-1).view(bz, nc, 1, 1)   # ([bz, 1024, 1, 1])

    # 4. 计算一组batch的CAMs
    cams = (weights * activations).sum(1)            
    cams = t2n(F.relu(cams))                        # ([bz, 5, 1])
    CAMs = []
    for idx, cam in enumerate(cams):
        cam = cv2.resize(cam, cam_shape,               # 上采样，基于4x4像素邻域的3次插值法
                                interpolation=cv2.INTER_CUBIC)
        cam_normalized = normalize_scoremap(cam)    # (224, 224)
        CAMs.append(cam_normalized)

    return np.array(CAMs)

if __name__ == "__main__":
    pass