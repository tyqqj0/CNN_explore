# -*- CODING: UTF-8 -*-
# @time 2022/11/17 13:17
# @Author tyqqj
# @File main.py


import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets



# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
