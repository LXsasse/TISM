
import torch
import torch.nn as nn

##### Copied from https://github.com/kundajelab/yuzu
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# to test yuzu

'''
# extracted from https://github.com/kundajelab/yuzu/blob/main/tutorials/3.%20Using%20Yuzu%20with%20Your%20Model.ipynb
# Something must have changed and it doesn't work anymore to assign terminal_layers

class Crop(torch.nn.Module):
    def __init__(self, trimming):
        super(Crop, self).__init__()
        self.trimming = trimming
    
    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming
        return X[:, :, start:end]

class BPNet2(torch.nn.Module):
    def __init__(self, n_filters=64, n_layers=3):
        super(BPNet2, self).__init__()
        self.trimming = 2 ** n_layers + 3
        self.n_filters = n_filters
        self.n_layers = n_layers

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=7, padding=3)
        self.relu1 = torch.nn.ReLU()
        
        self.rconv1 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2, dilation=2)
        self.relu2 = torch.nn.ReLU()
        
        self.rconv2 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=4, dilation=4)
        self.relu3 = torch.nn.ReLU()
        
        self.rconv3 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=8, dilation=8)
        self.relu4 = torch.nn.ReLU()
        
        self.fconv = torch.nn.Conv1d(n_filters, 1, kernel_size=7, padding=3)
        self.crop = Crop(self.trimming)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.relu1(self.iconv(X))
        X = self.relu2(self.rconv1(X))
        X = self.relu3(self.rconv2(X))
        X = self.relu4(self.rconv3(X))
        X = self.fconv(X)
        X = self.crop(X)
        X = self.logsoftmax(X)
        return X
    
model = BPNet2()
seq_len = 500

precomputation = precompute(model, seq_len, alpha=1.05, terminal_layers=(Crop,))
idxs = numpy.random.RandomState(0).randn(4, seq_len).argmax(axis=0)
X = numpy.zeros((1, 4, seq_len), dtype='float32')
X[0, idxs, numpy.arange(seq_len)] = 1

y_ism = yuzu_ism(model, X, precomputation, terminal_layers=(Crop,))

#And we can compare the time to the naive ISM method.

from yuzu.naive_ism import naive_ism

n_ism = naive_ism(model, X)
'''


# Extracted from
# https://github.com/kundajelab/yuzu/yuzu_ism/models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch
import random

'''
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)
'''
# Need to import these layers from models because yuzu_ism uses predefined terminal_layers that cannot be given to the model anymore
from yuzu.models import Flatten, Unsqueeze


class OneLayer(torch.nn.Module):
    def __init__(self, n_inputs, n_filters=64, kernel_size=7, seq_len=None, random_state=0):
        super(OneLayer, self).__init__()
        torch.manual_seed(random_state)
        self.conv = torch.nn.Conv1d(n_inputs, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, X):
        with torch.no_grad():
            return self.conv(X)

class DeepSEA(torch.nn.Module):
    def __init__(self, n_inputs, seq_len=None, random_state=0):
        super(DeepSEA, self).__init__()
        torch.manual_seed(random_state)

        k = 4

        self.conv1 = torch.nn.Conv1d(4, 320, kernel_size=2*k+1, padding=k)
        self.relu1 = torch.nn.ReLU()
        self.mp1 = torch.nn.MaxPool1d(k)

        self.conv2 = torch.nn.Conv1d(320, 480, kernel_size=2*k+1, padding=k)
        self.relu2 = torch.nn.ReLU()
        self.mp2 = torch.nn.MaxPool1d(k)

        self.conv3 = torch.nn.Conv1d(480, 960, kernel_size=2*k+1, padding=k)
        self.relu3 = torch.nn.ReLU()

        self.reshape = Flatten()
        self.fc = torch.nn.Linear((seq_len // k // k) * 960, 925)
        self.sigmoid = torch.nn.Sigmoid()
        self.unsqueeze = Unsqueeze(1)

    def forward(self, X):
        with torch.no_grad():
            X = self.mp1(self.relu1(self.conv1(X)))
            X = self.mp2(self.relu2(self.conv2(X)))
            X = self.relu3(self.conv3(X))

            X = self.reshape(X)
            X = self.sigmoid(self.fc(X))
            X = self.unsqueeze(X)
            return X


class Basset(torch.nn.Module):
    def __init__(self, n_inputs, seq_len=None, random_state=0):
        super(Basset, self).__init__()
        torch.manual_seed(random_state)

        self.conv1 = torch.nn.Conv1d(4, 300, kernel_size=19, padding=9)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(300)
        self.maxpool1 = torch.nn.MaxPool1d(3)

        self.conv2 = torch.nn.Conv1d(300, 200, kernel_size=11, padding=5)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.maxpool2 = torch.nn.MaxPool1d(4)

        self.conv3 = torch.nn.Conv1d(200, 200, kernel_size=7, padding=3)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(200)
        self.maxpool3 = torch.nn.MaxPool1d(4)

        self.reshape = Flatten()

        self.fc1 = torch.nn.Linear((seq_len // 3 // 4 // 4) * 200, 1000)
        self.relu4 = torch.nn.ReLU()
        #self.bn4 = torch.nn.BatchNorm1d(1000)

        self.fc2 = torch.nn.Linear(1000, 1000)
        self.relu5 = torch.nn.ReLU()
        #self.bn5 = torch.nn.BatchNorm1d(1000)


        self.fc3 = torch.nn.Linear(1000, 164)
        self.unsqueeze = Unsqueeze(1)

    def forward(self, X):
        with torch.no_grad():
            X = self.maxpool1(self.bn1(self.relu1(self.conv1(X))))
            X = self.maxpool2(self.bn2(self.relu2(self.conv2(X))))
            X = self.maxpool3(self.bn3(self.relu3(self.conv3(X))))

            X = self.reshape(X)

            X = self.relu4(self.fc1(X))
            X = self.relu5(self.fc2(X))
            
            X = self.fc3(X)
            X = self.unsqueeze(X)
            return X





### ATTENTION YUZU automatically sums over isms from all output tracks
# if you have different cell types as output it will automicacally apply the sum over all of these

from yuzu import yuzu_ism, precompute
import numpy as np
import time
#from yuzu.models import DeepSEA, Basset
from yuzu.naive_ism import naive_ism


if __name__ == '__main__':
    seq_len, n_choices = 200, 4

    idxs = np.random.RandomState(0).randn(n_choices, seq_len).argmax(axis=0)
    X = np.zeros((1, n_choices, seq_len), dtype='float32')
    X[0, idxs, np.arange(seq_len)] = 1
    print(X.shape)

    model = DeepSEA(seq_len=seq_len, n_inputs=n_choices)
    model1 = Basset(seq_len=seq_len, n_inputs=n_choices)

    y = model.forward(torch.Tensor(X))
    y1 = model1.forward(torch.Tensor(X))

    print(y.shape, y1.shape)

    precomputation = precompute(model, seq_len, n_choices, device='cpu')

    tic = time.time()
    yuzu_result = yuzu_ism(model, X, precomputation, device='cpu')
    yuzu_time = time.time() - tic

    tic = time.time()
    naive_result = naive_ism(model, X, device='cpu')
    naive_time = time.time() - tic

    print(naive_time, yuzu_time, naive_result.shape, yuzu_result.shape)



