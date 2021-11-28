# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        config.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/13
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 参数配置文件
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/13 |          creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/09 |     增加ACARS配置部分
# ------------------------------------------------------------------------
'''

class DefaultConfigs_RML(object):
    ''' 默认参数配置 '''
    # Dataset
    dataset_name = "ACARS"                # RML2016.04c(6db,50): 89.0689
    num_classes = 20                            # 分类类别数
    signal_len = "4096, 2"
    train_path_x = 'D:\\WIN_10_DEDKTOP\\onWorking\\SAR&DD\\PyTorch_Codes\\Signal_master\\dataset\\ACARS\\x_train_100persample.npy'   # 原始训练数据目录
    train_path_y = 'D:\\WIN_10_DEDKTOP\\onWorking\\SAR&DD\\PyTorch_Codes\\Signal_master\\dataset\\ACARS\\y_train_100persample.npy'   # 原始训练数据目录
    test_path_x = 'D:\\WIN_10_DEDKTOP\\onWorking\\SAR&DD\\PyTorch_Codes\\Signal_master\\dataset\\ACARS\\x_test_8000samples.npy'    # 原始测试数据目录
    test_path_y = 'D:\\WIN_10_DEDKTOP\\onWorking\\SAR&DD\\PyTorch_Codes\\Signal_master\\dataset\\ACARS\\y_test_8000samples.npy'    # 原始测试数据目录
    process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理

    batch_size = 64                            # DataLoader中batch大小，550/110=5 Iter
    num_workers = 4                             # DataLoader中的多线程数量

    # model
    model = "MsmcNet_ACARS"                   # 指定模型，MsmcNet_ACARS or MsmcNet_RML2016
    resume = True                              # 是否加载训练好的模型
    checkpoint_name = 'MsmcNet_ACARS.pth' # 训练完成的模型名

    # train
    num_epochs = 5                          # 训练轮数
    lr = 0.005                                   # 初始lr
    valid_freq = 1                              # 每几个epoch验证一次
    
    # log
    iter_smooth = 10                             # 打印&记录log的频率

    # seed = 1000                               # 固定随机种子

    # CAM
    Erase_thr = 0.3                             # CAM擦除软阈值，越小MASK保留越多，擦除越多
    CAM_omega = 20                              # 该参数从GAIN论文中获得，理论上将CAM的尺度拓展到0-omega，值越小越软



class DefaultConfigs_RML(object):
    ''' 默认参数配置 '''
    # Dataset
    dataset_name = "RML2016.04c"                # RML2016.04c(6db,50): 89.0689
    num_classes = 11                            # 分类类别数
    signal_len = "128,2"
    train_path = '/home/hp-video/Documents/LiuRT/Signal_master-master/dataset/RML2016_04c/6dB_SNR/6dB-SNR_100-samples.npy'  # 原始训练数据目录
    test_path = '/home/hp-video/Documents/LiuRT/Signal_master-master/dataset/RML2016_04c/6dB_SNR/6dB-SNR_815-test.npy'    # 原始测试数据目录
    process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理

    batch_size = 110                            # DataLoader中batch大小，550/110=5 Iter
    num_workers = 4                             # DataLoader中的多线程数量

    # model
    model = "MsmcNet_RML2016"                   # 指定模型，目前就一个
    resume = False                              # 是否加载训练好的模型
    checkpoint_name = 'MsmcNet_RML2016_04c_6dB_100-base-1.pth' # 训练完成的模型名

    # train
    num_epochs = 10000                          # 训练轮数
    lr = 0.01                                   # 初始lr
    valid_freq = 1                              # 每几个epoch验证一次
    
    # log
    iter_smooth = 10                             # 打印&记录log的频率

    # seed = 1000                               # 固定随机种子


cfgs = DefaultConfigs_RML()
#cfgs = DefaultConfigs_ACARS()
