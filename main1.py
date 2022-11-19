# -*- CODING: UTF-8 -*-
# @time 2022/11/17 13:17
# @Author tyqqj
# @File main1.py


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader



training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()#调用父类的构造函数
        self.flatten = nn.Flatten()#将输入的图像展平
        # nn.Linear()是一个线性层，它的作用是将输入的数据进行线性变换，即y = Wx + b
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),#输入的图像是28*28的，所以输入的数据是28*28=784
            nn.ReLU(),#激活函数
            nn.Linear(512, 512),#输入的数据是512，输出的数据是512
            nn.ReLU(),#激活函数
            nn.Linear(512, 10),#输入的数据是512，输出的数据是10
            nn.ReLU()#激活函数

        )

    def forward(self, x):
        x = self.flatten(x)#将输入的图像展平
        logits = self.linear_relu_stack(x)#将展平的图像输入到神经网络中
        return logits#返回输出
model = NeuralNetwork().to(device)
print(model)