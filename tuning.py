import torch
import itertools
import os
import numpy as np
import pandas as pd
from time import time

from online import online
from model import SimpleNet2
from stop_algo import fix_train_model


def tuning():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.6
    Model = SimpleNet2().to(device)
    dataset_path = "data/dataset/fuzzy_data_u1_b10/"
    data_para = [1000, 1]
    out_path = "data/result/tuning/data_aug/test2/"

    batch_size_list = [1000]
    train_epoch_list = [10]
    online_epoch_list = [50]
    sigma_list = [10**(-4)]

    # 0ならFalse, 1ならTrue
    reset_list = [0]
    # 1: loss1, 2: 1-p, 3: p
    loss_list = [1, 2, 3]

    tune_list = np.array(list(itertools.product(batch_size_list, train_epoch_list, online_epoch_list, sigma_list, reset_list, loss_list)))
    df = pd.DataFrame(data=tune_list, columns=['batch_size', 'train_epoch', 'online_epoch', 'sigma', 'reset', 'loss'])
    df.to_csv(out_path+"para.txt")

    for i, (batch_size, train_epoch, online_epoch, sigma, reset, loss) in enumerate(tune_list, 1):

        print("="*50)
        print("epoch: {}".format(i))
        print("batch_size: {}".format(batch_size))
        print("train_epoch: {}".format(train_epoch))
        print("online_epoch: {}".format(online_epoch))
        print("sigma: {}".format(sigma))

        if reset == 0:
            reset_flag = False
        else:
            reset_flag = True
        
        if loss == 1:
            loss_type = "loss1"
        elif loss == 2:
            loss_type = "1-p"
        elif loss == 3:
            loss_type = "p"

        print("reset_flag: {}".format(reset_flag))
        print("loss_type: {}".format(loss_type))

        os.makedirs(out_path + str(i) + "/p", exist_ok=True)
        os.makedirs(out_path + str(i) + "/d", exist_ok=True)
        os.makedirs(out_path + str(i) + "/l", exist_ok=True)

        online(acc, Model, dataset_path, data_para, out_path + str(i) + "/", i, 
                batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
                online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
                reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                )
    
def tuning2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.8
    Model = SimpleNet2().to(device)
    dataset_path = "data/dataset/fuzzy_data_u1_b10/"
    # data_para = [1000, 1]
    data_para = [100, 1]
    out_path = "data/result/tuning/stop_algo/1/"

    batch_size_list = [100]
    train_epoch_list = [50]
    limit_phase_list = [2]
    

    tune_list = np.array(list(itertools.product(batch_size_list, train_epoch_list, limit_phase_list)))
    df = pd.DataFrame(data=tune_list, columns=['batch_size', 'train_epoch', 'limit_phase'])
    df.to_csv(out_path+"para.txt")

    for i, (batch_size, train_epoch, limit_phase) in enumerate(tune_list, 1):

        print("="*50)
        print("epoch: {}".format(i))
        print("batch_size: {}".format(batch_size))
        print("train_epoch: {}".format(train_epoch))

        if limit_phase == 1:
            limit_phase = "train"
        elif limit_phase == 2:
            limit_phase = "val"

        os.makedirs(out_path + str(i) + "/p", exist_ok=True)
        os.makedirs(out_path + str(i) + "/d", exist_ok=True)
        os.makedirs(out_path + str(i) + "/l", exist_ok=True)

        fix_train_model(acc, Model, dataset_path, data_para, out_path + str(i) + "/", i,
            num_epochs=train_epoch, batch_size=batch_size, limit_phase=limit_phase)

# tuning()
tuning2()