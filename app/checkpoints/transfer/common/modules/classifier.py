from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch

__all__ = ['Classifier']


class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True):
        super(Classifier, self).__init__()
        self.backbone = backbone #
        self.num_classes = num_classes #分类的类别
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
                #nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x) #backbone网络
        f = self.bottleneck(f) #这个bottleneck是要修改的，这个应该是最后一个全连接前的操作
        predictions = self.head(f) #这里就是最后一个全连接
        return predictions, f

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

