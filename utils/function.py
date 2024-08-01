#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2023/5/31
# @file function.py
import os
import numpy as np
import pandas as pd
def creatpath(save_path):
    # 创建新文件夹路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def soc_ocv_model(parameters,soc):
    '''
    Calculate the params of model between SOC-OCV:
    OCV = phi + theta * exp(-beta * SOC)
    :param parameters:
    :param soc:
    :return:
    '''
    phi = parameters[0]
    theta = parameters[1]
    beta = parameters[2]
    ocv = float(phi) + float(theta) * np.exp(-float(beta) * soc)
    return ocv

def soc_ocv_model_multiphases(parameters,soc):
    '''
    Calculate the params of model between SOC-OCV:
    OCV = phi + theta * exp(-beta * SOC)
    :param parameters:
    :param soc:
    :return:
    '''
    phi = parameters[0]
    theta = parameters[1]
    beta = parameters[2]
    gama = parameters[3]
    ocv = float(phi) - float(theta) * np.power((-np.log(soc)),2.1)+float(beta)*soc+float(gama)*np.exp(30*(soc-1))
    return ocv

def residuals(parameters, data, y_observed, func):
    '''
    Compute residuals of y_predicted - y_observed
    where:
    y_predicted = func(parameters,x_data)
    '''
    return func(parameters, data) - y_observed

def figset(fig,ax,title,ylabel,xlabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    ax.tick_params(direction='in', length=5, width=1, grid_alpha=0.5, top=True, right=True, left=True, bottom='True')

def Mon(data):
    front_part = data[1:]
    follow_part = data[:-1]
    temp=front_part-follow_part
    num_pos = temp[temp > 0].shape[0]
    num_neg = temp[temp < 0].shape[0]
    mon_value=np.abs(num_pos - num_neg)/len(front_part)
    return mon_value


def smoothFilter(input_data, tao):
    def SES(y, n, a):
        y_S = []
        S_0 = 0
        for i in range(n):
            S_0 = y[i] + S_0
        S_0 = S_0 / n
        for i in range(0, y.size):
            S_i = a * y[i] + (1 - a) * S_0
            y_S.append(S_i)
            S_0 = S_i
        return pd.Series(y_S)

    return SES(SES(input_data, 1, tao), 1, tao)

def Rob(data):
    temp = data
    trend_data = smoothFilter(temp.HI, 0.4)
    observed_data = data
    rob_value=np.mean(np.exp(-abs((observed_data - trend_data) / observed_data)))
    return rob_value

def Tre(data):
    data=data[::-1]
    m=len(data)
    observed_data = data
    time_cycle = np.arange(len(data))+1
    sum_hxt = sum(observed_data * time_cycle)
    sum_hxsum_t = sum(observed_data) * sum(time_cycle)
    E_observed =  m*sum(observed_data ** 2) - sum(observed_data) ** 2
    E_time =  m*sum(time_cycle ** 2) - sum(time_cycle) ** 2
    tre_value=abs( m*sum_hxt - sum_hxsum_t) / np.sqrt(E_observed * E_time)
    return tre_value

