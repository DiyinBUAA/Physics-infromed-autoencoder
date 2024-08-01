import torch
import torch.nn as nn
import numpy as np
from utils import util
import os

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        # 使用tanh激活函数并将输出缩放和平移以获得[1,2]范围内的值
        out = 0.5*(torch.tanh(x) + 1)
        return out

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
            # nn.Dropout(),
            nn.ReLU(True),
            #nn.LayerNorm(64),
            nn.Linear(64, 1),
            #nn.Sigmoid(),#CustomActivation(),
        )
    def forward(self,x,weight=None):
        if weight is not None:
            x = torch.mul(x, weight.detach())
        out = self.layers(x)
        return out

class Encoder(nn.Module):
    def __init__(self,input_size=3):
        super(Encoder,self).__init__()
        self.features = backbone()
        self.predictor = Predictor()
        # self.layers = nn.LSTM(input_size=input_size,hidden_size=128,num_layers=2,batch_first=True)
        # self.hiddenstate = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     # nn.Dropout(),
        #     nn.Linear(64, 1),
        #     CustomActivation(),
        # )

    def forward(self,x,weight=None):
        x = x.transpose(1, 2)
        embed = self.features(x)
        z_state = self.predictor(embed, weight)
        # out,(h,c) = self.layers(x)
        # z_state=self.hiddenstate(out[:,-1,:])
        return z_state

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.Linear(32, 100),
            # nn.Sigmoid()，
        )
    def forward(self,x,weight=None):
        if weight is not None:
            x = torch.mul(x, weight.detach())
        out = self.layers(x)
        return out

class Decoder_phy(nn.Module):
    def __init__(self):
        super(Decoder_phy,self).__init__()
        self.layer2 = nn.Linear(10, 1)
        self.layer2.bias.data.fill_(0.)
        self.layer2.bias.requires_grad = False

    def phy_input(self,x,params):
        middle_linear2 = torch.zeros((x.shape[0], 100,10), dtype=x.dtype, device=x.device)
        for i in range(0,10):
            middle_linear2[:,:,i] = torch.pow((1-params/x[:,0].reshape(-1,1)),i+1)
        return middle_linear2

    def forward(self,x,params,weight=None):
        if weight is not None:
            x = torch.mul(x, weight.detach())
        middle_linear1 = x#self.layer1(x)
        middle_q=self.phy_input(middle_linear1.reshape(-1,1),params[:,0,:])
        ocv=self.layer2(middle_q).squeeze()
        out=ocv+params[:,1,:]
        return out

class modelPAE(nn.Module):
    def __init__(self,init_weights=True):
        super(modelPAE,self).__init__()
        self.encoder=Encoder()
        if init_weights:
            self._initialize_weights()
        self.coeff=nn.Linear(1,1)
        self.coeff.weight.data.fill_(1.2)
        self.coeff.weight.requires_grad=True
        self.coeff.bias.data.fill_(0.)
        self.coeff.bias.requires_grad = False
        self.decoder=Decoder_phy()
        self.hiddenstate =None
        # pretrained-result based on data fusion method
        state_dict = torch.load(os.path.join(
            '.\experiments\PAE\CALCE\Pretrain\pth',
            'Epoch381.pth'))
        self.encoder.load_state_dict(state_dict)

    def forward(self,x,params,weight=None):
        out = self.encoder(x)
        embed = out
        self.hiddenstate=embed
        pred = self.decoder(self.coeff(embed),params,weight)
        return pred,out

    def hiddenState(self):
        return self.hiddenstate

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
                nn.init.normal_(m.weight, 1, 0.01)
                nn.init.constant_(m.bias, 0)

class modelAE(nn.Module):
    def __init__(self,init_weights=True):
        super(modelAE,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
        if init_weights:
            self._initialize_weights()
        # pretrained-result based on data fusion method
        state_dict = torch.load(os.path.join(
            '.\experiments\PAE\CALCE\Pretrain\pth',
            'Epoch381.pth'))
        self.encoder.load_state_dict(state_dict)

    def forward(self,x,params,weight=None):
        out = self.encoder(x)
        embed = out
        self.hiddenstate=embed
        pred = self.decoder(embed,weight)
        return pred,out

    def hiddenState(self):
        return self.hiddenstate

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
                nn.init.normal_(m.weight, 1, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    from CALCE_Phy_loader import load_single_domain_data
    from utils.config import get_args
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    args = get_args()
    parent_dir = os.path.dirname(os.getcwd())
    os.chdir(parent_dir)
    setattr(args, 'source_dir', '.\data\CALCE_Anysis\AE_input\CS2')
    setattr(args, 'test_id', 1)
    data_set = args.source_dir.split('/')[-1]
    train_loader, valid_loader,test_loader=load_single_domain_data(args)
    net = modelPAE().to('cuda')
    state_dict = torch.load(os.path.join(
        'E:\PINN\PINN_sample\example_codes\Explainability-driven_SOH-master\experiments_CALCE_AE\CALCE-phyAE\LRP guide False\CS2\\test battery 1\experiment 0\pth',
        'Epoch4842.pth'))
    net.load_state_dict(state_dict)
    criterion=nn.MSELoss()
    loss=[]
    for bias in np.arange(0,0.5,0.1):
        for data,label in test_loader:
            x1=data[:,0,0:3,:].to('cuda')
            params1=data[:,0,3:5,:].to('cuda')
            y1,Q1 = net(x1,params1)



