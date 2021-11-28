# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/utils/Guided_BP.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/07/24
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 实现指导性反向传播（Guided BackPropagation）
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> TODO
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2020/07/24 |           creat
--------------------------------------------------------------------------
'''

import torch
from torch.nn import ReLU


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, input_image):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_input(input_image)
    
    def hook_input(self, input_image):
        def hook_image(grad):
            self.gradients = grad
            print("iamge_grad:", grad.shape)
        input_image.requires_grad=True
        input_image.register_hook(hook_image)

    ''' FIXME 由于PyTorch官方register_backward_hook()函数无法处理多操作module的bug原因，弃用
        self.hook_layers()
    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]     # torch.Size([1, 150, 4096, 2])
            print(len(grad_in), len(grad_out))
            print(grad_in[0].shape, grad_in[1].shape)
            print(grad_out[0].shape)
        # Register hook to the first layer
        first_layer = list(self.model.Block_3._modules.items())[0][1]
        print(first_layer)
        first_layer.register_backward_hook(hook_function)
    '''
    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for block_num, block in self.model._modules.items():
            for pos, module in block._modules.items():
                if isinstance(module, ReLU):
                    module.register_backward_hook(relu_backward_hook_function)
                    module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        gradient = one_hot_output       # torch.Size([1, 20])
        model_output.backward(gradient.cuda())
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        
        gradients_as_arr = self.gradients.detach().cpu().numpy()    # FIXED (1, 1, 4096, 2)
        
        # 归一化
        gradients_as_arr -= gradients_as_arr.min()
        gradients_as_arr /= gradients_as_arr.max()
        
        return gradients_as_arr

