import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class resnet10(ResNet):
    def __init__(self, num_classes=10):
        super(resnet10, self).__init__(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)



