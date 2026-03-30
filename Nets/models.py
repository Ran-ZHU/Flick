# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from .nets import LeNet5
from .ResNet import resnet10
from efficientnet_pytorch import EfficientNet

def get_model(conf, pretrained=True):
	data_name = conf["data"]
	if data_name == "mnist":
		model = LeNet5()
	else:
		if data_name == "office-caltech":
			model = resnet10(num_classes=conf['num_classes'])
		else:
			# model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
			model = models.resnet18()
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, conf['num_classes'])

	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model