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

#### Need to import these layers from models because yuzu_ism uses predefined
# terminal_layers that cannot be given to the model for some reason ###
from yuzu.models import Flatten, Unsqueeze

class YuzuAItac(torch.nn.Module):
    def __init__(self, seq_len, tracks = 1, n_kernels = 300, embedding_size = 200, kernel_size = 19, pooling1 = 3, pooling_size = 4, random_state=0):
        super(YuzuAItac, self).__init__()
        torch.manual_seed(random_state)

        self.conv1 = torch.nn.Conv1d(4, n_kernels, kernel_size=kernel_size, padding=kernel_size//2 )
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(n_kernels)
        self.maxpool1 = torch.nn.MaxPool1d(pooling1)

        self.conv2 = torch.nn.Conv1d(n_kernels, embedding_size, kernel_size=11, padding=5)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool2 = torch.nn.MaxPool1d(pooling_size)

        self.conv3 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool3 = torch.nn.MaxPool1d(pooling_size)

        self.reshape = Flatten()

        fcdimension = (seq_len // pooling1 // pooling_size // pooling_size) * embedding_size
        self.fc1 = torch.nn.Linear(fcdimension, 1000)
        self.relu4 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(1000, 1000)
        self.relu5 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(1000, tracks)

    def forward(self, X):
        X = self.maxpool1(self.bn1(self.relu1(self.conv1(X))))
        X = self.maxpool2(self.bn2(self.relu2(self.conv2(X))))
        X = self.maxpool3(self.bn3(self.relu3(self.conv3(X))))

        X = self.reshape(X)

        X = self.relu4(self.fc1(X))
        X = self.relu5(self.fc2(X))
        
        X = self.fc3(X)
        
        return X

class YuzuAItacDeep(torch.nn.Module):
    def __init__(self,  seq_len, tracks = 1, n_kernels = 300, embedding_size = 200, kernel_size = 19, pooling1 = 2, pooling_size = 2, random_state=0):
        super(YuzuAItacDeep, self).__init__()
        torch.manual_seed(random_state)

        self.conv1 = torch.nn.Conv1d(4, n_kernels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(n_kernels)

        self.conv2a = torch.nn.Conv1d(n_kernels, embedding_size, kernel_size=11, padding=5)
        self.relu2a = torch.nn.ReLU()
        self.bn2a = torch.nn.BatchNorm1d(embedding_size)
        
        self.conv2b = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=11, padding=5)
        self.relu2b = torch.nn.ReLU()
        self.bn2b = torch.nn.BatchNorm1d(embedding_size)
        
        self.conv2c = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=11, padding=5)
        self.relu2c = torch.nn.ReLU()
        self.bn2c = torch.nn.BatchNorm1d(embedding_size)
        
        self.maxpool2 = torch.nn.MaxPool1d(pooling1)

        self.conv3 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool3 = torch.nn.MaxPool1d(pooling_size)

        self.conv4 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3)
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool4 = torch.nn.MaxPool1d(pooling_size)
        
        self.conv5 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3)
        self.relu5 = torch.nn.ReLU()
        self.bn5 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool5 = torch.nn.MaxPool1d(pooling_size)
        
        self.conv6 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=7, padding=3)
        self.relu6 = torch.nn.ReLU()
        self.bn6 = torch.nn.BatchNorm1d(embedding_size)
        self.maxpool6 = torch.nn.MaxPool1d(pooling_size)

        self.reshape = Flatten()

        fcdimension = (seq_len // pooling1 // pooling_size // pooling_size // pooling_size // pooling_size) * embedding_size
        self.fc1 = torch.nn.Linear(fcdimension, 1000)
        self.relu7 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(1000, 1000)
        self.relu8 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(1000, 1000)
        self.relu9 = torch.nn.ReLU()

        self.fc4 = torch.nn.Linear(1000, tracks)
        
    def forward(self, X):
        X = self.bn1(self.relu1(self.conv1(X)))
        
        X = self.bn2a(self.relu2a(self.conv2a(X)))
        X = self.bn2b(self.relu2b(self.conv2b(X)))
        X = self.bn2c(self.relu2c(self.conv2c(X)))
        
        X = self.maxpool2(X)
        
        X = self.maxpool3(self.bn3(self.relu3(self.conv3(X))))
        X = self.maxpool4(self.bn4(self.relu4(self.conv4(X))))
        X = self.maxpool5(self.bn5(self.relu5(self.conv5(X))))
        X = self.maxpool6(self.bn6(self.relu6(self.conv6(X))))

        X = self.reshape(X)

        X = self.relu7(self.fc1(X))
        X = self.relu8(self.fc2(X))
        X = self.relu9(self.fc3(X))
        
        X = self.fc4(X)
            
        return X




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
