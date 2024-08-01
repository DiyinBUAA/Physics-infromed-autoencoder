import torch
import torch.nn as nn
import os
import numpy as np
from utils.util import AverageMeter,save_to_txt,mkdir,create_logger,eval_metrix
from utils.LSTM_model import model
import matplotlib.pyplot as plt
import time
from utils.trainer_function import get_optimizer,get_scheduler,load_data,set_random_seed
import seaborn as sns
import pickle
from utils.util import Scaler
from utils.util import My_loss
import utils.function as func
import math

def final_test(test_loader,model,args):
    ground_true = []
    predict_label = []

    for data,label in test_loader:
        model.eval()
        data1 = data[:,0,:3,:].to(args.device).float()
        params = data[:,0,3:5,:].to(args.device).float()
        pred,_ = model(data1,params)#model(data1,params)[:,::10]#pred,_ = model(data1,params)
        pred=pred[:,::10]
        #pred,_,_ = model(data1, params)
        ground_true.append(label[:, 0, :])
        predict_label.append(pred.cpu().detach().numpy())

    return np.concatenate(ground_true), np.concatenate(predict_label)

def final_test_lstm(test_loader,model,args):
    ground_true = []
    predict_label = []

    for data,label in test_loader:
        model.eval()
        data1 = data[:,0,:3,:].to(args.device).float()
        pred = model(data1)
        ground_true.append(label[:, 0, :])
        predict_label.append(pred.cpu().detach().numpy())

    return np.concatenate(ground_true), np.concatenate(predict_label)


def train(train_loader, valid_loader, test_loader,model,optimizer,lr_scheduler,args):
    if args.is_save_logging:
        mkdir(args.save_root)
        log_name = args.save_root + '/train info.log'
        log, consoleHander, fileHander= create_logger(filename=log_name)
        log.critical(args)
    else:
        log, consoleHander = create_logger()

    try:
        stop = 0
        min_test_loss = 10
        last_best_model = None
        criterion = nn.MSELoss()
        for e in range(1,args.n_epoch+1):
            model.train()
            pred_loss = AverageMeter()
            lrp_loss = AverageMeter()

            for data,label in train_loader:
                model.train()
                data = data[:,0,0:3,:].to(args.device).float()#data[:,1:3,:]
                label =label[:,0,:].to(args.device)#label
                pred = model(data)
                pred_l = criterion(pred, label)

                #################################################
                loss = pred_l
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_loss.update(loss.item())
            if lr_scheduler:
                lr_scheduler.step()

            train_info = f'training......Epoch:[{e}/{args.n_epoch}], ' \
                         f'pred_loss:{100*pred_loss.avg:.4f},'
            log.info(train_info)

            ##################### test #######################
            if valid_loader is None:
                valid_loader = test_loader
            stop += 1
            #### valid
            true_label,pred_label = final_test_lstm(model=model, test_loader=valid_loader, args=args)
            raw_matrix = eval_metrix(true_label=true_label,pred_label=pred_label)
            valid_info = f"validing......valid_MAE:[{raw_matrix[0]:.4f}]  lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
            log.warning(valid_info)
            min_loss = min(raw_matrix[0],raw_matrix[0])

            if min_test_loss > min_loss:
                min_test_loss = min_loss

                true_label,pred_label = final_test_lstm(model=model, test_loader=test_loader, args=args)
                metrix = eval_metrix(true_label=true_label,pred_label=pred_label)

                test_loss = metrix[2]*100
                test_info = f"testing......Epoch:[{e}/{args.n_epoch}],  test_loss(100x):{test_loss:.4f}," \
                            f"matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},RMSE={metrix[3]:.6f}."
                log.error(test_info)

                stop = 0
                #######plot test results#########
                if args.is_plot_test_results:
                    plt.plot(true_label, label='true')
                    plt.plot(pred_label, label='pred')
                    plt.title(f"Epoch:{e}, MAE:{metrix[0]:.4f}")
                    plt.legend()
                    plt.show()
                ####### save model ########
                if args.is_save_best_model:
                    if last_best_model is not None and e<50:
                        os.remove(last_best_model)  # delete last best model

                    save_folder = args.save_root + '/pth'
                    mkdir(save_folder)
                    best_model = os.path.join(save_folder, f'Epoch{e}.pth')
                    torch.save(model.state_dict(), best_model)
                    last_best_model = best_model
                #########save test results (test info) to txt #####
                if args.is_save_to_txt:
                    txt_path = args.save_root + '/test_info.txt'
                    time_now = time.strftime("%Y-%m-%d", time.localtime())
                    if e == 1:
                        save_to_txt(txt_path, ' ')
                        #save_to_txt(txt_path,f'########## experiment {args.experiment_time} ##########')
                        for k,v in vars(args).items():
                            save_to_txt(txt_path,f'{k}:{v}')
                    info = time_now + f' epoch = {e}, test_loss(100x):{test_loss:.6f}， ' \
                                      f'matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},RMSE={metrix[3]:.6f}.'

                    save_to_txt(txt_path,info)
                #########save test results (predict value) to np ######
                if args.is_save_test_results:
                    np.save(args.save_root+'/pred_label',pred_label)
                    np.save(args.save_root+'/true_label',true_label)
            if args.early_stop > 0 and stop > args.early_stop:
                print(' Early Stop !')
                if args.is_save_logging:
                    log.removeHandler(consoleHander)
                    log.removeHandler(fileHander)
                else:
                    log.removeHandler(consoleHander)
                break
    except:
        txt_path = args.save_root + '/test_info.txt'
        save_to_txt(txt_path, 'Error !')
    if args.is_save_logging:
        log.removeHandler(consoleHander)
        log.removeHandler(fileHander)
    else:
        log.removeHandler(consoleHander)

def train_AE(train_loader, valid_loader, test_loader,model,optimizer,lr_scheduler,args,criterion,log_sigma_u,log_sigma_f,log_sigma_f_t):
    if args.is_save_logging:
        mkdir(args.save_root)
        log_name = args.save_root + '/train info.log'
        log, consoleHander, fileHander= create_logger(filename=log_name)
        log.critical(args)
    else:
        log, consoleHander = create_logger()
    try:
        stop = 0
        min_test_loss = 10
        last_best_model = None

        for e in range(1,args.n_epoch+1):
            # if e==2000:
            #     criterion = My_loss(mode='Sum')
            model.train()
            pred_loss = AverageMeter()   #L(y-y_~)
            for data,label in train_loader:  #单次训练
                model.train()
                input1=data[:,0,:3,:]
                params1=data[:,0,3:5,:]
                input1 = input1.to(args.device).float()
                params1= params1.to(args.device).float()
                label1 = label[:,0,:].to(args.device).float()
                pred1,ze = model(input1,params1)#pred1,ze = model(input1,params1)
                pred1=pred1[:,::10]
                # pred1, mu, sigma = model(input1,params1)
                h1=model.hiddenState()
                input2 = data[:,1,:3,:]
                params2 = data[:,1,3:5,:]
                input2 = input2.to(args.device).float()
                params2 = params2.to(args.device).float()
                label2 = label[:,1, :].to(args.device).float()
                pred2,ze2 = model(input2, params2)#pred2,ze2 = model(input2, params2)
                pred2 = pred2[:, ::10]
                h2 = model.hiddenState()
                # step=128//args.num_samples
                # label_sample=label[:, 18::step]#label[:, ::step]
                # pred_l = criterion(pred, label_sample)#label_sample[:,:10]
                pred_l = criterion(
                    outputs1=pred1,
                    targets1=label1,
                    outputs2=ze,#h1
                    targets2=ze2,#params1[:,0,-1].reshape(-1,1)
                    hiddenstate1=h1,
                    hiddenstate2=h1,
                    log_sigma_u=log_sigma_u,
                    log_sigma_f=log_sigma_f,
                    log_sigma_f_t=log_sigma_f_t
                )
                #################################################
                loss = pred_l
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_loss.update(loss.item())
            if lr_scheduler:
                lr_scheduler.step()
            train_info = f'training......Epoch:[{e}/{args.n_epoch}], ' \
                         f'pred_loss:{100*pred_loss.avg:.4f}, ' \
                         f'regulation1:{torch.exp(-log_sigma_u).detach():.4f},regulation2:{torch.exp(-log_sigma_f).detach():.4f}'
            log.info(train_info)
            ##################### test #######################

            stop += 1
            #### valid
            if valid_loader is None:
                valid_loader = test_loader
            true_label,pred_label = final_test(model=model, test_loader=valid_loader, args=args)
            raw_matrix = eval_metrix(true_label=true_label,pred_label=pred_label)
            valid_info = f"validing......valid_MAE:[{raw_matrix[0]:.4f}]  lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
            log.warning(valid_info)
            min_loss = raw_matrix[0]

            if min_test_loss > min_loss:
                min_test_loss = min_loss

                true_label,pred_label = final_test(model=model, test_loader=test_loader, args=args)
                metrix = eval_metrix(true_label=true_label,pred_label=pred_label)

                test_loss = metrix[2]*100
                test_info = f"testing......Epoch:[{e}/{args.n_epoch}],  test_loss(100x):{test_loss:.4f}," \
                            f"matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},MSE={metrix[2]:.6f},RMSE={metrix[3]:.6f}."
                log.error(test_info)

                stop = 0
                #######plot test results#########
                if args.is_plot_test_results:
                    plt.plot(true_label, label='true')
                    plt.plot(pred_label, label='pred')
                    plt.title(f"Epoch:{e}, MAE:{metrix[0]:.4f}")
                    plt.legend()
                    plt.show()
                ####### save model ########
                if args.is_save_best_model:
                    if last_best_model is not None and e<50:
                        os.remove(last_best_model)  # delete last best model

                    save_folder = args.save_root + '/pth'
                    mkdir(save_folder)
                    best_model = os.path.join(save_folder, f'Epoch{e}.pth')
                    torch.save(model.state_dict(), best_model)
                    last_best_model = best_model
                #########save test results (test info) to txt #####
                if args.is_save_to_txt:
                    txt_path = args.save_root + '/test_info.txt'
                    time_now = time.strftime("%Y-%m-%d", time.localtime())
                    if e == 1:
                        save_to_txt(txt_path, ' ')
                        #save_to_txt(txt_path,f'########## experiment {args.experiment_time} ##########')
                        for k,v in vars(args).items():
                            save_to_txt(txt_path,f'{k}:{v}')
                    info = time_now + f' epoch = {e}, test_loss(100x):{test_loss:.6f}， ' \
                                      f'matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},MSE={metrix[2]:.6f},RMSE={metrix[3]:.6f}.'
                    save_to_txt(txt_path,info)
                #########save test results (predict value) to np ######
                if args.is_save_test_results:
                    np.save(args.save_root+'/pred_label',pred_label)
                    np.save(args.save_root+'/true_label',true_label)
            if args.early_stop > 0 and stop > args.early_stop:
                print(' Early Stop !')
                if args.is_save_logging:
                    log.removeHandler(consoleHander)
                    log.removeHandler(fileHander)
                else:
                    log.removeHandler(consoleHander)
                break

    except:
        txt_path = args.save_root + '/test_info.txt'
        save_to_txt(txt_path, 'Error !')
    if args.is_save_logging:
        log.removeHandler(consoleHander)
        log.removeHandler(fileHander)
    else:
        log.removeHandler(consoleHander)

def main(args):
    if args.method_type == 'PAE':
        from utils.AE_model import modelPAE as model
    elif args.method_type == 'AE':
        from utils.AE_model import modelAE as model
    else:
        from utils.LSTM_model import model
    from utils import util
    set_random_seed(args.seed)
    _, train_loader, valid_loader, test_loader = load_data(args)
    m = model().to(args.device)
    if args.isTest == False:
        if (args.method_type == 'AE')|(args.method_type == 'PAE'):
            #####################正则项##########################
            # log_sigma_u = torch.zeros(())
            # log_sigma_f = torch.zeros(())
            # log_sigma_f_t = torch.zeros(())
            #####################自适应##########################
            log_sigma_u = torch.randn((), requires_grad=True)
            log_sigma_f = torch.randn((), requires_grad=True)
            log_sigma_f_t = torch.randn((), requires_grad=True)
            if args.method_type == 'PAE':
                criterion = My_loss(mode='Sum')
            else:
                criterion = My_loss(mode='Baseline')
            params = ([p for p in m.parameters()] + [log_sigma_u] + [log_sigma_f] + [log_sigma_f_t])
        else:
            log_sigma_u = torch.zeros(())
            log_sigma_f = torch.zeros(())
            log_sigma_f_t = torch.zeros(())
            criterion = My_loss(mode='Baseline')
            params = m.parameters()
        optimizer = get_optimizer(m, args,params)
        if args.lr_scheduler:
            scheduler = get_scheduler(optimizer, args)
        else:
            scheduler = None
        if (args.method_type=='AE')|(args.method_type=='PAE'):
            train_AE(train_loader, valid_loader, test_loader, m, optimizer, scheduler, args,criterion,log_sigma_u,log_sigma_f,log_sigma_f_t)
        else:
            train(train_loader, valid_loader, test_loader, m,optimizer, scheduler, args)
        #torch.cuda.empty_cache()
        del train_loader
        del valid_loader
        del test_loader
    else:
        m = model()
        path = args.save_root + '\\pth'
        state_dict = torch.load(os.path.join(
            'C:\\Users\huangxuc\Desktop\PAE_github\experiments\PAE\CALCE\Train\\test battery 1\experiment 0\pth','Epoch4842.pth'))
        root_path =args.source_dir
        file_name=util.get_batterynumber(args.test_id)
        with open(os.path.join(root_path, file_name) + '\\' + file_name + '.pkl', 'rb') as file:
            data = pickle.load(file)
        soh_label=data['record_capacity']/max(data['record_capacity'])
        m.load_state_dict(state_dict)
        fig, ax = plt.subplots()
        SOH = list()
        color_palette = sns.color_palette(['#ac9f8a', '#547689', '#2e59a7'])
        for data, label in test_loader:
            data = data.float()
            label = label.float()
            m.eval()
            with torch.no_grad():
                input_data = data[:,0, :3, :]
                params=data[:,0, 3:5, :]
                V,_ = m(input_data,params)
                soh=m.hiddenState()
                # soh=m(input_data)
                SOH.append(list(soh.numpy()))
        soh_result = np.array(sum(SOH, [])).reshape(-1, 1)
        plt.ylim(0,1.2)
        sns.lineplot(soh_result.reshape(-1), label='Physics-informed', ax=ax, color=color_palette[0], linewidth=3)
        sns.lineplot(soh_label[2:], label='True', ax=ax, color=color_palette[1], linewidth=3)
        func.figset(fig, ax, 'SOH Curve', 'cycle', 'SOH')
        fig.show()

def run_on_CALCE_data(isTest):
    from utils.config import get_args
    args = get_args()
    for i in range(5):  # 5 experiments
        for test_id in [1]:
            for data in ['CS2']:
                try:
                    setattr(args, 'is_save_best_model', True)
                    setattr(args, 'source_dir', f'data/CALCE_Anysis/AE_input/{data}')#'data/CALCE/{data}'
                    setattr(args, 'test_id', test_id)
                    setattr(args, 'save_root',
                            f'experiments/PAE/CALCE/Pretrain1')
                    if isTest == True:
                        setattr(args, 'isTest',
                                True)
                    main(args)
                except:
                    continue

def run_on_CALCE_data_AE(isTest):
    from utils.config import get_args
    args = get_args()
    for i in range(5):  # 5 experiments
        for test_id in [1]:
            for data in ['CS2']:
                try:
                    setattr(args, 'is_save_best_model', True)
                    setattr(args, 'source_dir', f'data/CALCE_Anysis/AE_input/{data}')
                    setattr(args, 'method_type', 'AE')
                    setattr(args, 'test_id', test_id)
                    setattr(args, 'save_root',
                            f'experiments/AE/CALCE/test battery {test_id}/experiment {i}')
                    if isTest == True:
                        setattr(args, 'isTest',
                                True)
                    main(args)
                except:
                    continue

def run_on_CALCE_data_AE_phy(isTest):
    from utils.config import get_args
    args = get_args()
    for i in range(1):  # repeat 1 experiments
        for test_id in [1]:
            for lrp in [False]:
                for data in ['CS2']:
                    try:
                        setattr(args, 'is_save_best_model', True)
                        setattr(args, 'source_dir', f'./data/CALCE_Anysis/AE_input/{data}')
                        setattr(args, 'method_type', 'PAE')
                        setattr(args, 'test_id', test_id)
                        setattr(args, 'save_root',
                                f'experiments/PAE/CALCE/Train1/test battery {test_id}/experiment {i}')
                        if isTest == True:
                            setattr(args, 'isTest',
                                    True)
                        main(args)
                    except:
                        continue

if __name__ == '__main__':
    run_on_CALCE_data_AE_phy(isTest=False)
    # run_on_CALCE_data_AE(isTest=True)
    # run_on_CALCE_data(isTest=True)




