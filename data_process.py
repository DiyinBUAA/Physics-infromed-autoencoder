import copy
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
import utils.function as func
import pickle
from scipy import optimize
import seaborn as sns
from datetime import date


def load_data(name, dir_path):
    '''
    :return: xlsx中分段提取充放点信息，保存为pkl格式
    '''
    Battery = {}
    print('Load Dataset ' + name + ' ...')
    path = glob.glob(dir_path +'\\' +name + '/*.xlsx')
    dates = []
    #文件的顺序在0-9满足顺序，大于不满足-->基于时间顺序，进行数据名称的重排列
    for p in path:
        if '~$' in p:
            continue
        df = pd.read_excel(p, sheet_name=1)
        print('Load ' + str(p) + ' ...')
        dates.append(df['Date_Time'][0])
    idx = np.argsort(dates)
    path_sorted = np.array(path)[idx]
    # 变量定义
    count = 0
    discharge_capacities = []
    health_indicator = []
    internal_resistance = []
    internal_resistance_charge = []
    CCCT = []
    CVCT = []
    CCDC=[]
    CCDV=[]
    CCDT=[]
    OCV=[]
    SOC=[]
    record_capacity=[]
    last_c=[]
    CCCC=[]
    CCCV=[]
    CCCQ=[]
    CCDQ=[]
    file_capacity = []

    for p in path_sorted:
        df = pd.read_excel(p,sheet_name=1)
        groups = df.groupby('Cycle_Index')
        cycle_d = groups.apply(lambda x: x.iloc[-1]['Discharge_Capacity(Ah)'])
        cycle_c = groups.apply(lambda x: x.iloc[-1]['Charge_Capacity(Ah)'])
        print('Load ' + str(p) + ' ...')
        cycles = list(set(df['Cycle_Index']))
        for c in cycles:
            df_lim = df[df['Cycle_Index'] == c]
            if (len(df_lim[df_lim['Step_Index'] == 7])==0):
                if (len(df_lim[df_lim['Step_Index'] == 4])==0):
                    continue
                else:
                    continue
            #Charging
            df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
            c_v = df_c['Voltage(V)']
            c_c = df_c['Current(A)']
            c_t = df_c['Test_Time(s)']
            #CC or CV
            df_cc = df_lim[df_lim['Step_Index'] == 2]
            df_cv = df_lim[df_lim['Step_Index'] == 4]
            c_im = df_lim['Internal_Resistance(Ohm)']
            CCCT.append(df_cc['Test_Time(s)'])
            CVCT.append(df_cv['Test_Time(s)'])
            if (len(list(c_c)) != 0):
                time_diff = np.diff(list(c_t))
                c_c = np.array(list(c_c))[1:]
                charge_capacity = time_diff * c_c / 3600  # Q = A*h
                internal_resistance_charge.append(np.mean(np.array(c_im)))
                all_capacity=np.sum(charge_capacity )
                charge_capacity = [np.sum(charge_capacity[:n])
                                      for n in range(charge_capacity.shape[0])]
                charge_capacity.append(all_capacity)
                count += 1
                # print(f'充电容量为:{charge_capacity[-1]}')
            CCCC.append(c_c)
            CCCV.append(c_v)
            CCCQ.append(charge_capacity)

            #Discharging
            df_d = df_lim[df_lim['Step_Index'] == 7]
            d_v = df_d['Voltage(V)']
            d_c = df_d['Current(A)']
            d_t = df_d['Test_Time(s)']
            d_im = df_d['Internal_Resistance(Ohm)']
            d_c_record=df_d['Discharge_Capacity(Ah)']
            cap=d_c_record.iloc[-1]-d_c_record.iloc[0]
            if(len(list(d_c))>1):
                time_diff = np.diff(list(d_t))
                ocv = np.array(d_v + d_im * d_c)
                d_c1=np.array(list(d_c))
                d_c = np.array(list(d_c))[1:]
                discharge_capacity = time_diff*d_c/3600 # Q = A*h
                all_capacity = np.sum(discharge_capacity[:])
                discharge_capacity = [np.sum(discharge_capacity[:n])
                                      for n in range(discharge_capacity.shape[0])]
                discharge_capacity.append(all_capacity)
                discharge_capacities.append(-1*discharge_capacity[-1])
                health_indicator.append(-1 * discharge_capacity[-1]/1.1)
                internal_resistance.append(np.mean(np.array(d_im)))
                soc=1+discharge_capacity/cap
                #重采样过程
                D_t,D_c1=resampleData(np.array(d_t), np.array(d_c1))
                D_t,D_v=resampleData(np.array(d_t), np.array(d_v))
                D_t,Ocv=resampleData(np.array(d_t), np.array(ocv))
                D_t,Soc=resampleData(np.array(d_t), np.array(soc))
                CCDC.append(D_c1) #d_c1
                CCDV.append(D_v)  #d_v
                CCDT.append(D_t)  #d_t
                OCV.append(Ocv)   #ocv
                SOC.append(Soc)   #soc
                CCDQ.append(discharge_capacity)
                count += 1
                # print(f'放电容量为:{discharge_capacity[-1]}')
            record_capacity.append(cap)
            last_c.append(df_lim['Discharge_Capacity(Ah)'].iloc[-1])

    #为了维度变换，(86,128)->(86,),引入维度不一致部分
    CCDC.append(np.ones(10))
    CCDV.append(np.ones(10))
    CCDT.append(np.ones(10))
    OCV.append(np.ones(10))
    SOC.append(np.ones(10))
    CCDQ.append(np.ones(10))
    #列表转为array，用于dataframe存储
    discharge_capacities = np.array(discharge_capacities)
    health_indicator = np.array(health_indicator)
    internal_resistance = np.array(internal_resistance)
    internal_resistance_charge= np.array(internal_resistance_charge)
    CCCT = np.array(CCCT)
    CVCT = np.array(CVCT)
    CCDC = np.array(CCDC)
    CCDV = np.array(CCDV)
    CCDT = np.array(CCDT)
    OCV = np.array(OCV)
    SOC =np.array(SOC)
    CCCC = np.array(CCCC)
    CCCV = np.array(CCCV)
    CCCQ = np.array(CCCQ)
    CCDQ = np.array(CCDQ)
    #去除维度变换引入的新列
    CCDC = CCDC[:-1]
    CCDV = CCDV[:-1]
    CCDT = CCDT[:-1]
    OCV = OCV[:-1]
    SOC = SOC[:-1]
    CCDQ = CCDQ[:-1]
    idx = drop_outlier(discharge_capacities, count, 40)
    df_result = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                              'capacity':CCDQ[idx],
                              'SoH':health_indicator[idx],
                              'record_capacity':np.array(record_capacity)[idx],
                              'resistance':internal_resistance[idx],
                              'internal_resistance_charge':internal_resistance_charge[idx],
                              'CCCT':CCCT[idx],
                              'CVCT':CVCT[idx],
                              'CCDC':CCDC[idx],
                              'CCDV':CCDV[idx],
                              'CCDT':CCDT[idx],
                              'OCV':OCV[idx],
                              'SOC':SOC[idx],
                              'CCCC':CCCC[idx],
                              'CCCV':CCCV[idx],
                              'CCCQ':CCCQ[idx]})
    return df_result

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

def resampleData(x,y):
    t=x-x[0]
    # 计算样条表示的参数
    tck = interpolate.splrep(t, y, s=0)
    # 需要预测的数据点
    data_len=t[-1]
    x_resample=np.arange(data_len,0,-data_len/128)[::-1]
    # 使用样条来预测数据点的值
    y_resample=interpolate.splev(x_resample,tck,der=0)
    return x_resample,y_resample

def AEInputGenerate(data):
    def resample(x, y):
        t = x - x[0]
        tck = interpolate.splrep(t, y, s=0)
        data_len = t[-1]
        x_resample = np.arange(data_len, 0, -data_len / 100)[::-1]
        y_resample = interpolate.splev(x_resample, tck, der=0)
        return x_resample[:100], y_resample[:100]

    # 1.obtain voltage range info during charge/discharge stage
    data['cycle'] = np.arange(1, len(data) + 1)
    C_v_upper = []
    C_v_lower = []
    D_v_upper = []
    D_v_lower = []
    palette = sns.color_palette('Blues', len(data))
    for cycle in data.cycle.unique():
        data_cycle = data[data.cycle == cycle]
        # data alignment--charge phase
        data_charge_ct = np.array(data_cycle['CCCT'][cycle - 1])[1:]
        len_cc = len(data_charge_ct)
        data_charge_cc = np.array(data_cycle['CCCC'][cycle - 1])[:len_cc]     #charge current in constant charge phase
        data_charge_cv = np.array(data_cycle['CCCV'][cycle - 1])[1:len_cc + 1] #charge voltage in constant charge phase

        # discharge phase
        data_discharge_cc = data_cycle['CCDC']           #charge current in constant charge phase
        data_discharge_cv = np.array(data_cycle['CCDV']) #charge current in constant charge phase
        data_discharge_ct = data_cycle['CCDT']           #charge current in constant charge phase

        C_v_upper.append(np.array(data_charge_cv)[-1])
        C_v_lower.append(np.array(data_charge_cv)[0])
        D_v_upper.append(data_discharge_cv[0][0])  # 放电数据
        D_v_lower.append(data_discharge_cv[0][-1])
    c_v_upper = 4.2#min(C_v_upper)
    c_v_lower = 3.5#max(C_v_lower)
    d_v_upper = 4.2#min(D_v_upper)  # 放电数据
    d_v_lower = 3.3#max(D_v_lower)
    # 2.截断电流范围
    data_recons = copy.deepcopy(data)
    C_c = []
    C_v = []
    C_t = []
    Cv_c = []
    Cv_v = []
    Cv_t = []
    D_c = []  # 放电数据
    D_v = []
    D_t = []
    D_IR=[]
    y_label = []
    palette = sns.color_palette("Blues", len(data))  ###test
    for cycle in data.cycle.unique():
        data_cycle = data[data.cycle == cycle]
        #恒压段提取
        data_charge_ct = np.array(data_cycle['CCCT'][cycle - 1])[1:] - np.array(data_cycle['CCCT'][cycle - 1])[1]
        len_cc = len(data_charge_ct)
        data_charge_cdct = np.array(data_cycle['CVCT'][cycle - 1])-np.array(data_cycle['CVCT'][cycle - 1])[0]
        print(cycle)
        data_charge_cdcc = np.array(data_cycle['CCCC'][cycle - 1])[len_cc:]  # 恒压
        data_charge_cdcv = np.array(data_cycle['CCCV'][cycle - 1])[len_cc + 1:]
        cv_resample_t, cv_resample_c = resample(data_charge_cdct, data_charge_cdcc)
        cv_resample_t, cv_resample_v = resample(data_charge_cdct, data_charge_cdcv)

        # 3. 数据对齐
        data_charge_ct = np.array(data_cycle['CCCT'][cycle - 1])[1:] - np.array(data_cycle['CCCT'][cycle - 1])[1]
        len_cc = len(data_charge_ct)
        data_charge_cc = np.array(data_cycle['CCCC'][cycle - 1])[:len_cc]
        data_charge_cv = np.array(data_cycle['CCCV'][cycle - 1])[1:len_cc + 1]
        data_discharge_cc = data_cycle['CCDC']  # 放电数据
        data_discharge_cv = data_cycle['CCDV']
        data_discharge_ct = data_cycle['CCDT']
        dc_t = np.diff(list(data['CCDT'][cycle - 1]))  # 相邻放电时间间隔
        delat_q = dc_t * -np.array(data_discharge_cc[cycle-1])[1:] / 3600  # 安时积分法算容量
        discharge_capacity = [np.sum(delat_q[:n + 1])
                           for n in range(delat_q.shape[0])]
        discharge_capacity.insert(0, 0.0)
        # 充电过程始末点
        start_point = np.where(np.array(data_charge_cv - c_v_lower) >= 0)[0][0]
        if len(np.where(np.array(data_charge_cv - c_v_upper) >= 0)) > 0:
            end_point = np.where(np.array(data_charge_cv - c_v_upper) >= 0)[0][0]
        else:
            end_point = len(data_charge_cv[0]) - 1
        # 放电过程始末点
        dis_start_point = np.where(np.array(data_discharge_cv - d_v_upper)[0] <= 0)[0][0]
        if len(np.where(np.array(data_discharge_cv - d_v_lower)[0] <= 0)) > 0:
            dis_end_point = np.where(np.array(data_discharge_cv - d_v_lower)[0] <= 0)[0][0]
        else:
            dis_end_point = len(data_discharge_cv[0]) - 1
        c_truncated_c = np.array(data_charge_cc[start_point:end_point + 1])#[start_point:end_point + 1]
        c_truncated_t = np.array(data_charge_ct[start_point:end_point + 1])
        c_truncated_v = np.array(data_charge_cv[start_point:end_point + 1])
        c_resample_t, c_resample_c = resample(c_truncated_t, c_truncated_c)
        c_resample_t, c_resample_v = resample(c_truncated_t, c_truncated_v)
        d_truncated_c = np.array(data_discharge_cc[cycle - 1][dis_start_point:dis_end_point])#[dis_start_point:dis_end_point]
        d_truncated_t = (data_discharge_ct[cycle - 1][dis_start_point:dis_end_point])
        d_truncated_v = (data_discharge_cv[cycle - 1][dis_start_point:dis_end_point])
        d_truncated_q=np.array(discharge_capacity[dis_start_point:dis_end_point])
        d_resample_q, d_resample_t = resample(d_truncated_q, d_truncated_t)
        d_resample_q, d_resample_c = resample(d_truncated_q, d_truncated_c)
        d_resample_q, d_resample_v = resample(d_truncated_q, d_truncated_v)

        C_c.append(pd.Series([c_resample_c], index=[cycle - 1]))
        C_v.append(pd.Series([c_resample_v], index=[cycle - 1]))
        C_t.append(pd.Series([c_resample_t], index=[cycle - 1]))
        Cv_c.append(pd.Series([cv_resample_c], index=[cycle - 1]))
        Cv_v.append(pd.Series([cv_resample_v], index=[cycle - 1]))
        Cv_t.append(pd.Series([cv_resample_t], index=[cycle - 1]))
        D_c.append(pd.Series([d_resample_c], index=[cycle - 1]))
        D_v.append(pd.Series([d_resample_v], index=[cycle - 1]))
        D_t.append(pd.Series([d_resample_t], index=[cycle - 1]))
        sample_index = np.arange(100)[::10]
        # sample_index = np.append(sample_index, 99)
        sample_V = d_resample_v[sample_index]
        y_label.append(pd.Series([sample_V], index=[cycle]))
        #discharge current*resistance
        D_IR.append(pd.Series([data_cycle['resistance'][cycle - 1]*d_resample_c],index=[cycle - 1]))
    ChargeData = pd.DataFrame(
        {'cycle': np.arange(1, len(C_c) + 1), 'CCDC': np.array(C_c).reshape(-1), 'CCDV': np.array(C_v).reshape(-1),
         'CCDT': np.array(C_t).reshape(-1)})
    ChargeData_cv = pd.DataFrame(
        {'cycle': np.arange(1, len(Cv_c) + 1), 'CCDC': np.array(Cv_c).reshape(-1), 'CCDV': np.array(Cv_v).reshape(-1),
         'CCDT': np.array(Cv_t).reshape(-1)})
    DischargeData = pd.DataFrame(
        {'cycle': np.arange(1, len(D_c) + 1), 'CCDC': np.array(D_c).reshape(-1), 'CCDV': np.array(D_v).reshape(-1),
         'CCDT': np.array(D_t).reshape(-1)})
    sample_D_V = pd.DataFrame({'V': np.array(y_label).reshape(-1)})
    inputdata_d_IR=pd.DataFrame({'d_IR': np.array(D_IR).reshape(-1)})
    return ChargeData, ChargeData_cv,DischargeData, sample_D_V,inputdata_d_IR

def Q_estimate():
    def resample(x, y):
        t = x - x[0]
        tck = interpolate.splrep(t, y, s=0)
        data_len = t[-1]
        x_resample = np.arange(data_len, 0, -data_len / 100)[::-1]
        y_resample = interpolate.splev(x_resample, tck, der=0)
        return x_resample[:100], y_resample[:100]

    path='.\data\CALCE_Anysis\Preprocess\CS2'
    Battary_list = os.listdir(path)
    for file_name in Battary_list:
        #1. load params
        with open(path + '\\' +file_name+'\\'+file_name+'_clean.pkl', 'rb') as file:
            data2 = pickle.load(file)
        print(len(data2.cycle.unique()))
        Q = pd.DataFrame(columns=['Q'])
        for i in data2.cycle.unique():
            data = data2[data2.cycle == i]
            CDT = np.array(data['CCDT'][i - 1])[:]  # time in discharging process
            CDC = np.array(data['CCDC'][i - 1])[:]  # current in discharging process
            CCV = np.array(data['CCDV'][i - 1])[:]  # voltage in discharging process
            c_t = np.diff(list(data['CCDT'][i - 1]))  #  time difference between two time points
            R=data['resistance'][i - 1]
            delat_q = c_t * -CDC[1:] / 3600  # calculate the delta capacity
            charge_capacity = [np.sum(delat_q[:n + 1])
                               for n in range(delat_q.shape[0])]
            charge_capacity.insert(0,0.0)
        # 2 conform used partial region of discharging process
            start_point = np.where(np.array(CCV - 4.2) <= 0)[0][0]
            end_point = np.where(np.array(CCV- 3.3) <= 0)[0][0]
            input_q = charge_capacity[start_point:end_point]
            d_resample_t, d_resample_q = resample(np.array(input_q), np.array(input_q))
            Q=pd.concat((Q,pd.DataFrame({'Q':pd.Series([np.array(d_resample_q)])})),ignore_index=True)
        Q.to_csv('.\data\CALCE_Anysis\\AE_Input\\CS2\\'+file_name+'\Q_'+file_name+'.csv')

def strToarray(data):
    data1=copy.deepcopy(data)
    for i in np.arange(len(data)):
        # clean string and spilt with ' '
        cleaned_string = data1[i].replace('[', '').replace(']', '')
        split_string = cleaned_string.split()

        # change format: list->float array
        data[i] = np.array(split_string, dtype=float)
    return data



if __name__ == '__main__':
    ############################画图初始化################################
    sns.set_theme(style="white", font='Times New Roman', font_scale=1.4)
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    today = date.today()
    file_name = str(today)
    debug_path = func.creatpath(os.path.join('./', 'debug//'+file_name))
    #######################多文件xlsx->单文件pkl#######################
    data_types = ['CS2']
    for data_type in data_types:
        dir_path = '.\data\CALCE_Anysis\\' + data_type
        Battary_list = os.listdir(dir_path)
        for name in Battary_list:
            save_dir = func.creatpath('.\data\CALCE_Anysis\Preprocess\\'+data_type+'\\'+name)
            data = load_data(name, dir_path)
            with open(save_dir + '\\' + name + '.pkl', 'wb') as file:
                pickle.dump(data, file)
    # ##########################剔除异常数据 ###########################
    root_path = '.\data\CALCE_Anysis\\Preprocess\CS2'
    save_dir = '.\data\CALCE_Anysis\\AE_input\CS2'
    files = os.listdir(root_path)
    for file_name in files:
        with open(os.path.join(root_path, file_name) + '\\' + file_name + '.pkl', 'rb') as file:
            data = pickle.load(file)
        # 1.Defined failre thresold: SOH=0.5
        SOH_END=0.5
        if len(np.where((np.array(data.SoH) - SOH_END) <= 0)[0])>0:
            cycyle_endPoint = np.where((np.array(data.SoH) - SOH_END) <= 0)[0][0]
            data = data.iloc[:cycyle_endPoint]
        new_dir=func.creatpath(os.path.join(save_dir, file_name))
        # 2.Drop abnormal data without correct charging process
        abnormal_idnex=[]
        data.cycle=np.arange(1, len(data) + 1)
        for cycle in data.cycle.unique():
            data_cycle = data[data.cycle == cycle]
            data_charge_ct = np.array(data_cycle['CCCT'][cycle - 1])[1:]
            len_cc = len(data_charge_ct)
            data_charge_cc = np.array(data_cycle['CCCC'][cycle - 1])[:len_cc]
            data_charge_cv = np.array(data_cycle['CCCV'][cycle - 1])[1:len_cc + 1]
            data_charge_cvcv = np.array(data_cycle['CCCV'][cycle - 1])[len_cc + 1:]
            if len(data_charge_cv) == 0:
                abnormal_idnex.append(cycle-1)
                continue
            if np.array(data_charge_cv)[0]>4:
                abnormal_idnex.append(cycle-1)
                continue
            if len(data_charge_cvcv) <= 1:
                abnormal_idnex.append(cycle - 1)
                continue
        dataout=data.drop(abnormal_idnex)
        # 3.data reindex
        dataout.index=range(len(dataout))
        dataout.cycle=np.arange(1, len(dataout) + 1)
        if file_name=='CS2_34':
            dataout.loc[dataout.cycle == 276, 'record_capacity'] = (dataout[dataout.cycle == 275]['record_capacity'].iloc[0] +
                                                              dataout[dataout.cycle == 277]['record_capacity'].iloc[0]) / 2
            dataout.loc[dataout.cycle == 440, 'record_capacity'] = (dataout[dataout.cycle == 439]['record_capacity'].iloc[0] +
                                                              dataout[dataout.cycle == 441]['record_capacity'].iloc[0]) / 2
        with open(new_dir + '\\' + file_name + '.pkl', 'wb') as newfile:
            pickle.dump(dataout, newfile)
    # ##########################生成作为输入的不同特征块###########################
    # root_path = '.\data\CALCE_Anysis\\AE_input\CS2'
    # files = os.listdir(root_path)
    # for file_name in files:
    #     with open(os.path.join(root_path, file_name) + '\\' + file_name + '.pkl', 'rb') as file:
    #         data = pickle.load(file)
    #     inputdata_c, inputdata_cv, inputdata_d, outlabel,inputdata_d_IR = AEInputGenerate(data)
    #     save_dir = func.creatpath('.\data\CALCE_Anysis\\AE_input\CS2' + '\\' + file_name)
    #     # 充电恒流阶段数据
    #     with open(save_dir + '\\' + file_name + '_inputdata.pkl', 'wb') as file:
    #         pickle.dump(inputdata_c, file)
    #     # 放电阶段端口电压，作为AE重构数据的标签
    #     with open(save_dir + '\\' + file_name + '_label.pkl', 'wb') as file:
    #         pickle.dump(outlabel, file)
    #     # 放电阶段数据
    #     with open(save_dir + '\\' + file_name + '_inputdata_d.pkl', 'wb') as file:
    #         pickle.dump(inputdata_d, file)
    #     # 充电恒压阶段数据
    #     with open(save_dir + '\\' + file_name + '_inputdata_cv.pkl', 'wb') as file:
    #         pickle.dump(inputdata_cv, file)
    #     # 放电阶段 电流*内阻
    #     with open(save_dir + '\\' + file_name + '_inputdata_d_IR.pkl', 'wb') as file:
    #         pickle.dump(inputdata_d_IR, file)
    #     # 每一个cycle的容量
    # Q_estimate()
    # ##########################以放电阶段的特征块组成网络输入###########################
    path_data='.\data\CALCE_Anysis\AE_input\CS2'
    counter=0
    for file_name in ['CS2_33','CS2_34','CS2_35','CS2_36','CS2_37','CS2_38']:
        with open(path_data + '\\' + file_name + '\\' + file_name + '_inputdata_d' + '.pkl',
                  'rb') as file:
            data = pickle.load(file)
        with open(path_data + '\\' + file_name + '\\' + file_name + '_inputdata_d_IR' + '.pkl',
                  'rb') as file1:
            data_d_IR = pickle.load(file1)
        data_q=pd.read_csv(path_data+'\\' + file_name + '\\' + 'Q_'+file_name+'.csv')
        data['d_IR'] = data_d_IR.d_IR
        data['Q']=strToarray(data_q.Q)
        counter += 1
        with open(path_data + '\\' + file_name + '\\'+file_name+'_inputdata_phy.pkl', 'wb') as file:
            pickle.dump(data, file)
