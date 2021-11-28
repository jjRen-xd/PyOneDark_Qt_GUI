# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        networks/MsmcNet.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- MsmcNet网络结构定义
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> MsmcNet_RML2016():
                        -- MsmcNet网络结构定义，所有参数均按照MsmcNet的keras
                        版本设置
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/14 | 实现MsmcNet网络的定义
# ------------------------------------------------------------------------
'''

import sys
import torch
import torch.nn as nn
from torchsummary import summary
import copy

class MsmcNet_RML2016(nn.Module):
    ''' 
    Funcs:
        RML2016数据集下的MsmcNet网络结构定义，所有参数均按照MsmcNet的keras版本设置
    Network input size:
        convention: (N, 1, 128, 2)
    Notes：
        <1> PyTorch中conv2d无padding='same'，经手动调整padding大小保证一致
        <2> FIXME PyTorch在网络定义中无法加入正则项：kernel_regularizer=l2(0.001)
        <3> FIXME PyTorch中的BN的参数量与kears不一致，仅为其一半
    '''
    #def __init__(self,num_classes):
    def __init__(self):
        super(MsmcNet_RML2016, self).__init__()
        #self._out_features = self.fc1.in_features #in_features是指提取某一层的固定参数

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.Block_2 = nn.Sequential(
            nn.Conv2d(30, 25, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_3 = nn.Sequential(
            nn.Conv2d(25, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_4 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(75, 6)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(100, 7)

    def forward(self, x):
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)

        #x = self.flatten(x)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc2(x)
        return x
    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        #return self._out_features
        self._out_features = self.fc.in_features
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


if __name__ == "__main__":
    ''' 测试MsmcNet.py，测试网络结构构建是否构建正确，并打印每层参数 '''
    model = MsmcNet_RML2016(num_classes=11)
    model.cuda()
    print(model)
    # 统计网络参数及输出大小
    summary(model, (1, 128, 2), batch_size=1, device="cuda")
