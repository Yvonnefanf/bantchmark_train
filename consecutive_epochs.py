# to compare
# 200-205 training data
# 200-205 training data with wrong testing data
"""
For motivating examples in DVI
We want to compare the difference between
200-205 epochs with pure training data
200-205 training data with wrong testing data,
We will observe the boundaries and positions of data afterward
"""
# import modules
import torch
from cifar10_models.resnet import resnet18

torch.manual_seed(1331)
# train original model
