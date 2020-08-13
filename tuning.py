import torch
import itertools
import os
import numpy as np
import pandas as pd

from online import online
from model import SimpleNet2


def tuning():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.80
    Model = SimpleNet2().to(device)
    # dataset_path = "data/artifact/"
    dataset_path = "data/dataset/fuzzy_data_u1_b10/"
    out_path = "data/result/tuning/loss_test/p5_2/"

    batch_size_list = [200]
    train_epoch_list = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    online_epoch_list = [50]
    sigma_list = [10**(-4)]

    # 0ならFalse, 1ならTrue
    reset_list = [0]
    # 1: loss1, 2: 1-p, 3: p
    loss_list = [3]

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

        online(acc, Model, dataset_path, out_path + str(i) + "/", i, 
                batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
                online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
                reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                )
    
tuning()