import numpy as np
import torch
import torch.nn as nn

class seqtofunc_cnn(nn.Module):
    def __init__(self, n_features, l_seqs, n_kernels = 200, l_kernels = 15, l_conv=7, padding = 2/3, N_convs = 3, pooling_size = 3, n_tracks = 1, **kwargs):
        super(seqtofunc_cnn, self).__init__()

        kpadding = int(l_kernels*padding)
        self.conv1 = nn.Sequential(nn.Conv1d(n_features, n_kernels, kernel_size = l_kernels, bias = False, stride = 1, padding = kpadding),nn.GELU())
        currlen = l_seqs+2*kpadding-l_kernels+1

        convpadding = int(l_conv*padding)
        convs_ = []
        for i in range(N_convs):
            convs_.append(nn.Conv1d(n_kernels, n_kernels, kernel_size = l_conv, bias = False, stride = 1, padding = convpadding))
            convs_.append(nn.GELU())
            convs_.append(nn.AvgPool1d(pooling_size))
            currlen = (currlen+2*convpadding - l_conv + 1)//pooling_size
        
        self.convpools = nn.Sequential(*convs_)
        
        
        currdim = currlen * n_kernels
        self.fully_connects = nn.Sequential(nn.Linear(currdim, currdim), nn.GELU(), nn.Linear(currdim, currdim), nn.GELU(),nn.Linear(currdim, currdim), nn.GELU())
        self.head = nn.Linear(currdim, n_tracks)
        
    def forward(self, x, **kwargs):
        pred = self.conv1(x)
        pred = self.convpools(pred)
        pred = pred.flatten(start_dim = 1)
        pred = self.fully_connects(pred)
        pred = self.head(pred)
        return pred




class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 320, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(67840, 2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
            ),
        )

    def forward(self, x):
        return self.model(x.unsqueeze(2))
