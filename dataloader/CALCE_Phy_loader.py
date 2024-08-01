import copy
import sys
sys.path.append('E:\\PINN\\PINN_sample\\example_codes\\Explainability-driven_SOH-master')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from utils.config import get_args
import os
from utils.util import Scaler
from sklearn.model_selection import train_test_split
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt
import utils.function as func

def  data_reconstruct_AE(args, dir_path,sample_num=128):
    X_list = []
    Y_list = []
    with open(args.source_dir + '\\' + dir_path + '\\' + dir_path+'_inputdata_d'+ '.pkl',
              'rb') as file:  # (args.source_dir+'\\'+dir_path[0]+'\\'+dir_path[0]+'.pkl', 'rb')  (args.source_dir+'\\'+dir_path[0], 'rb')c#'inputdata'
        data = pickle.load(file)
    with open(args.source_dir + '\\' + dir_path + '\\' + dir_path + '_label.pkl',
              'rb') as file:  # (args.source_dir+'\\'+dir_path[0]+'\\'+dir_path[0]+'.pkl', 'rb')  (args.source_dir+'\\'+dir_path[0], 'rb')
        data_label = pickle.load(file)
    battery_data = np.empty((len(data), 100, 3))
    battery_label = np.empty((len(data), 10))
    battery_t = np.empty((len(data), 100, 1))
    battery_voltage = np.empty((len(data), 100,1))
    for i in np.arange(len(data)):
        sample_dis=math.ceil(len(data['CCDT'].values[i])/sample_num)
        x_columns = ['CCDT', 'CCDC', 'CCDV']
        y_columns = ['CCDV']
        d_t = data['CCDT'].values[i]    #
        d_c = data['CCDC'].values[i]    #
        d_v = data['CCDV'].values[i]    #
        y_cycle_label=data_label['V'].values[i]

        battery_i_data = np.stack((d_t, d_c, d_v), axis=1)#np.stack((d_t, d_c, d_v, d_cycle,d_R), axis=1)
        battery_i_voltage=np.array(data[data.cycle == i + 1]['CCDV'][i])
        battery_i_data=battery_i_data.reshape((-1, 3))
        battery_i_voltage=battery_i_voltage.reshape((-1,1))
        battery_data[i, :, :] = battery_i_data
        battery_voltage[i, :, :] =battery_i_voltage
        battery_label[i, :] =y_cycle_label
    battery_data_out=battery_data[1:-1,:,:]
    battery_voltage_out=battery_voltage[1:-1,-10:,:].reshape(-1,10) #当前时刻1:-1 /下一时刻 2:
    battery_label_out=battery_label[1:-1,:]
    data_x_temp = Scaler(battery_data_out).minmax()
    data_x = data_x_temp
    data_y = battery_label_out
    X_list.append(data_x)
    Y_list.append(data_y)
    return X_list,Y_list

def  data_reconstruct_AE_phy(args, dir_path,sample_num=128):
    X_list = []
    Y_list = []
    with open(args.source_dir + '\\' + dir_path + '\\' + dir_path+'_inputdata_phy'+ '.pkl',
              'rb') as file:  # (args.source_dir+'\\'+dir_path[0]+'\\'+dir_path[0]+'.pkl', 'rb')  (args.source_dir+'\\'+dir_path[0], 'rb')c#'inputdata'
        data = pickle.load(file)
    with open(args.source_dir + '\\' + dir_path + '\\' + dir_path + '_label.pkl',
              'rb') as file:  # (args.source_dir+'\\'+dir_path[0]+'\\'+dir_path[0]+'.pkl', 'rb')  (args.source_dir+'\\'+dir_path[0], 'rb')
        data_label = pickle.load(file)
    battery_label = np.empty((len(data), 10))
    battery_t = np.empty((len(data), 100, 1))
    battery_data=np.empty((len(data), 100, 5))
    battery_voltage = np.empty((len(data), 100,1))
    for i in np.arange(len(data)):
        sample_dis=math.ceil(len(data['CCDT'].values[i])/sample_num)
        x_columns = ['CCDT', 'CCDC', 'CCDV','Q','IR']
        y_columns = ['CCDV']
        d_t = data['CCDT'].values[i]    #
        d_c = data['CCDC'].values[i]    #
        d_v = data['CCDV'].values[i]    #
        Q=data['Q'][i]
        IR=data['d_IR'][i]
        y_cycle_label=data_label['V'][i]
        battery_i_data = np.stack((d_t, d_c, d_v,Q,IR), axis=1)
        battery_i_voltage=np.array(data[data.cycle == i + 1]['CCDV'][i])
        battery_i_data=battery_i_data.reshape((1,-1, 5))
        battery_i_voltage=battery_i_voltage.reshape((-1,1))
        battery_data[i, :, :]=copy.deepcopy(battery_i_data)
        battery_voltage[i, :, :] =battery_i_voltage
        battery_label[i, :] =y_cycle_label
    battery_data_out=battery_data[1:-1,:,:]
    battery_voltage_out=battery_voltage[1:-1,-10:,:].reshape(-1,10) #当前时刻1:-1 /下一时刻 2:
    battery_label_out=battery_label[1:-1,:]
    data_x_temp = Scaler(battery_data_out[:, :, :3]).minmax()
    data_x = np.concatenate((data_x_temp, battery_data_out[:, :, 3:5].reshape(-1, 100, 2)), axis=2)
    data_y = battery_label_out
    X_list.append(data_x)
    Y_list.append(data_y)
    return X_list,Y_list

def load_single_domain_data(args):
    dir_paths = os.listdir(args.source_dir)

    ############################
    ####### load data
    ############################
    count = 0
    X_list=[]
    Y_list=[]
    X_test = []
    Y_test = []
    dir_paths=['CS2_33','CS2_34','CS2_35','CS2_36']
    for dir_path in dir_paths:
        count += 1
        if count == args.test_id:
            test_x,test_y = data_reconstruct_AE_phy( args, dir_path)  #data_reconstruct_AE & data_reconstruct_AE_phy
            print(f'target test battery (id={args.test_id}): {dir_path}')
            X_test.append(np.concatenate(test_x, axis=0).astype(np.float32))
            Y_test.append(np.concatenate(test_y, axis=0).astype(np.float32))
            continue
        x_list,y_list=data_reconstruct_AE_phy( args, dir_path)  #data_reconstruct_AE
        X_list.append(np.concatenate(x_list, axis=0).astype(np.float32))
        Y_list.append(np.concatenate(y_list, axis=0).astype(np.float32))
    train_X = np.concatenate(X_list, axis=0).astype(np.float32)
    train_Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    test_X = np.concatenate(X_test, axis=0).astype(np.float32)
    test_Y = np.concatenate(Y_test, axis=0).astype(np.float32)
    print('=' * 50)
    print('CALCE data:')
    print(f'train(valid): {train_X.shape}, {train_Y.shape}')
    print(f'test:  {test_X.shape}, {test_Y.shape}')
    print('-' * 50)

    train_x_temp = torch.from_numpy(np.transpose(train_X, (0, 2, 1)))
    train_y_temp = torch.from_numpy(train_Y)
    test_x_temp = torch.from_numpy(np.transpose(test_X, (0, 2, 1)))
    test_y_temp = torch.from_numpy(test_Y)
    #########################################################################
                                #考虑前后顺序#
    #########################################################################
    step_interval=20#20
    len_train=train_x_temp.shape[0]
    len_test=test_x_temp.shape[0]
    train_x= torch.zeros((len_train-step_interval,2,5,100))#torch.zeros((len_train-1,2,6,100))
    train_y=torch.zeros((len_train-step_interval,2,10))#torch.zeros((len_train-1,2,10))
    test_x=torch.zeros((len_test-step_interval,2,5,100))#torch.zeros((len_test-1,2,6,100))
    test_y=torch.zeros((len_test-step_interval,2,10))#torch.zeros((len_test-1,2,10))
    train_x[:,0,:,:]=train_x_temp[0:len_train-step_interval,:,:]
    train_x[:,1,:,:]=train_x_temp[step_interval:,:,:]
    test_x[:,0,:,:] = test_x_temp[0:len_test-step_interval, :, :]
    test_x[:,1,:,:]= test_x_temp[step_interval:, :, :]
    train_y[:,0,:] = train_y_temp[0:len_train-step_interval, :]
    train_y[:,1,:] = train_y_temp[step_interval:, :]
    test_y[:,0,:] = test_y_temp[0:len_test-step_interval, :]
    test_y[:,1,:] = test_y_temp[step_interval:, :]

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=args.seed)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False,
                              drop_last=False)
    return train_loader, valid_loader,test_loader


if __name__ == '__main__':
    pass


