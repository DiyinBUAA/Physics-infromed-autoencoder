#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2023-08-03
# @file resultDisplay.py
import pandas as pd

import utils.function as func
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from utils.AE_model import modelPAE,Encoder
from utils.config import get_args
from utils.trainer_function import get_optimizer,get_scheduler,load_data,set_random_seed
import torch
import os

############################sin Curve#############################
x=2*np.pi/100*np.arange(101)
y=np.sin(x)
fig,ax=plt.subplots()
sns.lineplot(x=x,y=y,ax=ax)
func.figset(fig,ax,'sin','t','Position')
plt.show()

args = get_args()
set_random_seed(args.seed)
setattr(args, 'source_dir', f'data\CALCE_Anysis\AE_input\CS2')
setattr(args, 'test_id', 1)
setattr(args, 'method_type', 'AE')
_, train_loader, valid_loader, test_loader = load_data(args)
############################Q-V Curve#############################
with open('data\CALCE_Anysis\AE_input\CS2\CS2_33\\'+ 'CS2_33_inputdata_phy' + '.pkl',
          'rb') as file:
    data = pickle.load(file)
fig,ax=plt.subplots()
palette = sns.color_palette("Blues_r", int(1.5*len(data)))
for i in np.arange(len(data)):
    v=data['CCDV'].values[i]
    q=data['Q'].values[i]
    if i==len(data)-1:
        sns.lineplot(x=q,y=v, linestyle='-', color='red',ax=ax,linewidth=2)
    else:
        sns.lineplot(x=q, y=v, linestyle='-', color=palette[i], ax=ax)
    plt.ylim((3.4,4.2))
func.figset(fig,ax,'OCV-Q','Discharge amount (Ah)','OCV(V)')
plt.show()

###########################SOH-comparison-battery1##########################
root_path = 'data\CALCE_Anysis\AE_input\CS2'
file_name='CS2_33'
with open(os.path.join(root_path, file_name) + '\\' + file_name + '.pkl', 'rb') as file:
    data = pickle.load(file)
soh_label=data['record_capacity']/max(data['record_capacity'])
m = modelPAE()
encoder= Encoder()
state_dict = torch.load(os.path.join(
            'experiments\PAE\CALCE\Train\\test battery 1\experiment 0\pth',
            'Epoch4842.pth'))
state_dict_pretrain = torch.load(os.path.join(
            'experiments\PAE\CALCE\Pretrain\pth',
            'Epoch381.pth'))
m.load_state_dict(state_dict)
encoder.load_state_dict(state_dict_pretrain)
SOH = list()
pre_SOH = list()
# # 1.pretrain/physic-informed/label
fig2,ax2=plt.subplots()
color_palette=sns.color_palette(['#ac9f8a','#547689','#2e59a7'])
for data, label in test_loader:
    data = data.float()
    label = label.float()
    m.eval()
    with torch.no_grad():
        input_data = data[:, 0, :3, :]
        params = data[:, 0, 3:5, :]
        V, _ = m(input_data, params)  # m(input_data,params)
        pre_soh=encoder(input_data)
        soh = m.hiddenState()
        SOH.append(list(soh.numpy()))
        pre_SOH.append(list(pre_soh.numpy()))
soh_result=np.array(sum(SOH,[])).reshape(-1,1)
pre_soh_result=np.array(sum(pre_SOH,[])).reshape(-1,1)
sns.lineplot(soh_result.reshape(-1),label='Physics-informed',ax=ax2,color=color_palette[0],linewidth=3)
sns.lineplot(soh_label[2:],label='True',ax=ax2,color=color_palette[1],linewidth=3)
sns.lineplot(pre_soh_result.reshape(-1),label='Initial',ax=ax2,color=color_palette[2],linewidth=3)
func.figset(fig2,ax2,'SOH Curve','cycle','SOH')

plt.show()
setattr(args, 'test_id', 2)
file_name='CS2_34'
with open(os.path.join(root_path, file_name) + '\\' + file_name + '.pkl', 'rb') as file:
    data = pickle.load(file)
soh_label=data['record_capacity']/max(data['record_capacity'])
soh_label[275] = (soh_label[274] + soh_label[276]) / 2
soh_label[439] = (soh_label[438] + soh_label[440]) / 2
_, train_loader, valid_loader, test_loader = load_data(args)
state_dict = torch.load(os.path.join(
            'experiments\PAE\CALCE\Train\\test battery 2\experiment 0\pth',
            'Epoch93.pth'))
m.load_state_dict(state_dict)
SOH = list()
pre_SOH = list()
fig8,ax8=plt.subplots()
color_palette=sns.color_palette(['#ac9f8a','#547689','#2e59a7'])
for data, label in test_loader:
    data = data.float()
    label = label.float()
    m.eval()
    with torch.no_grad():
        input_data = data[:, 0, :3, :]
        params = data[:, 0, 3:5, :]
        V, _ = m(input_data, params)  # m(input_data,params)
        pre_soh=encoder(input_data)
        soh = m.hiddenState()
        SOH.append(list(soh.numpy()))
        pre_SOH.append(list(pre_soh.numpy()))
soh_result=np.array(sum(SOH,[])).reshape(-1,1)
pre_soh_result=np.array(sum(pre_SOH,[])).reshape(-1,1)
sns.lineplot(soh_result.reshape(-1),ax=ax8,color=color_palette[0],linewidth=3)
sns.lineplot(soh_label[2:],ax=ax8,color=color_palette[1],linewidth=3)
sns.lineplot(pre_soh_result.reshape(-1),ax=ax8,color=color_palette[2],linewidth=3)
func.figset(fig8,ax8,'SOH Curve','cycle','SOH')
plt.show()






