import torch
import torch.nn as nn
import numpy as np
from utils import util

class backbone(nn.Module):
    def __init__(self,input_size=3):
        super(backbone,self).__init__()
        self.layers = nn.LSTM(input_size=input_size,hidden_size=128,num_layers=2,batch_first=True)

    def forward(self,x):
        out,(h,c) = self.layers(x)
        return out[:,-1,:]

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
    def forward(self,x,weight=None):
        if weight is not None:
            x = torch.mul(x, weight.detach())
        out = self.layers(x)
        return out

class model(nn.Module):
    def __init__(self,init_weights=True):
        super(model,self).__init__()
        self.features = backbone()
        self.predictor = Predictor()
        if init_weights:
            self._initialize_weights()
    def forward(self,x,weight=None):
        x = x.transpose(1, 2)
        embed = self.features(x)
        pred = self.predictor(embed,weight)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    batch_size = 32
    seq_len = 128
    input_size = 3
    x = torch.randn(batch_size,input_size,seq_len)
    net = model()
    y = net(x)
    print(f'input:{x.shape}, output:{y.shape}')

