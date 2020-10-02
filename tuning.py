import torch
import itertools
import os
import numpy as np
import pandas as pd
from time import time

from online import online, online_MNIST, online_multi, online_check
from model import SimpleNet2, MNISTNet, SimpleNet3
from stop_algo import fix_train_model
import dataset


def tuning():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.80
    Model = SimpleNet3().to(device)
    # Model = MNISTNet().to(device)
    dataset_path = "data/dataset/multi_fuzzy_1000_2/"
    # dataset_path = "data/dataset/fuzzy_1000_1/"
    # data_num_list = [200, 1000]
    # mu_list = [1, 3/4]
    out_path = "data/result/tuning/multi/8/"

    batch_size_list = [200]
    train_epoch_list = [10]
    online_epoch_list = [50]
    sigma_list = [10**(-4)]

    # 0ならFalse, 1ならTrue
    reset_list = [0]
    # 1: loss1, 2: 1-p, 3: p
    loss_list = [1, 1, 2, 2, 3, 3]

    # tune_list = np.array(list(itertools.product(data_num_list, mu_list, batch_size_list, train_epoch_list, online_epoch_list, sigma_list, reset_list, loss_list)))
    tune_list = np.array(list(itertools.product(batch_size_list, train_epoch_list, online_epoch_list, sigma_list, reset_list, loss_list)))
    # df = pd.DataFrame(data=tune_list, columns=['data_num', 'mu', 'batch_size', 'train_epoch', 'online_epoch', 'sigma', 'reset', 'loss'])
    df = pd.DataFrame(data=tune_list, columns=['batch_size', 'train_epoch', 'online_epoch', 'sigma', 'reset', 'loss'])
    df.to_csv(out_path+"para.txt")

    for i, (batch_size, train_epoch, online_epoch, sigma, reset, loss) in enumerate(tune_list, 1):

        print("="*50)
        print("epoch: {}".format(i))
        # print("data_num: {}".format(data_num))
        # print("mu: {}".format(mu))
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

        # online(acc, Model, dataset_path, out_path + str(i) + "/", i, 
        #         batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
        #         online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
        #         reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
        #         )

        # online_check(acc, Model, dataset_path, out_path + str(i) + "/", i, 
        #         batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
        #         online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
        #         reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
        #         )

        online_multi(acc, Model, dataset_path, out_path + str(i) + "/", i, 
                batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
                online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
                reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                )
    
def tuning2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.8
    Model = SimpleNet2().to(device)
    dataset_path = "data/dataset/fuzzy_1000_1/"
    data_num_list = [1000]
    mu_list = [1]
    out_path = "data/result/tuning/data_aug/0.8/"

    batch_size_list = [200]
    train_epoch_list = [50]
    # 1: train, 2: val
    limit_phase_list = [1, 2]
    # 1: loss1, 2: 1-p, 3: p
    # loss_list = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    loss_list = [1, 1, 1, 2, 2, 2, 3, 3, 3]

    tune_list = np.array(list(itertools.product(data_num_list, mu_list, batch_size_list, train_epoch_list, limit_phase_list, loss_list)))
    df = pd.DataFrame(data=tune_list, columns=['data_num', 'mu', 'batch_size', 'train_epoch', 'limit_phase', 'loss'])
    df.to_csv(out_path+"para.txt")

    for i, (data_num, mu, batch_size, train_epoch, limit_phase, loss) in enumerate(tune_list, 1):

        print("="*50)
        print("epoch: {}".format(i))
        print("data_num: {}".format(data_num))
        print("mu: {}".format(mu))
        print("batch_size: {}".format(batch_size))
        print("train_epoch: {}".format(train_epoch))

        if limit_phase == 1:
            limit_phase = "train"
        elif limit_phase == 2:
            limit_phase = "val"

        if loss == 1:
            loss_type = "loss1"
        elif loss == 2:
            loss_type = "1-p"
        elif loss == 3:
            loss_type = "p"


        os.makedirs(out_path + str(i) + "/p", exist_ok=True)
        os.makedirs(out_path + str(i) + "/d", exist_ok=True)
        os.makedirs(out_path + str(i) + "/l", exist_ok=True)

        fix_train_model(acc, Model, dataset_path, int(data_num), mu, out_path + str(i) + "/", i,
            num_epochs=int(train_epoch), batch_size=int(batch_size), limit_phase=limit_phase, loss_type=loss_type)

def tuningMNIST():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.6
    # Model = SimpleNet2().to(device)
    Model = MNISTNet().to(device)
    out_path = "data/result/tuning/mnist/play/1_7_3/"

    dataset_set = dataset.MNIST_load()

    batch_size_list = [200]
    train_epoch_list = [10]
    online_epoch_list = [50]
    sigma_list = [10**(-4)]

    # 0ならFalse, 1ならTrue
    reset_list = [0]
    # 1: loss1, 2: 1-p, 3: p
    loss_list = [1, 1, 2, 2, 3, 3]

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

        os.makedirs(out_path + str(i), exist_ok=True)

        online_MNIST(acc, Model, dataset_set, out_path, i,
                batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
                online_epoch=int(online_epoch), sigma=sigma,  # オンライン予測のパラメータ
                reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                )
tuning()
# tuning2()
# tuningMNIST()