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
    dataset_path = "data/fuzzy_data_u1_b10/"
    out_path = "data/result/tuning/loss_test/p/"

    batch_size_list = [10, 25, 50]
    train_epoch_list = [5, 10, 20, 30]
    online_epoch_list = [50]
    sigma_list = [10**(-5)]
    # batch_size_list = [10]
    # train_epoch_list = [10]
    # online_epoch_list = [50]
    # sigma_list = [10**(-5)]

    tune_list = np.array(list(itertools.product(batch_size_list, train_epoch_list, online_epoch_list, sigma_list)))
    print(tune_list)

    df = pd.DataFrame(data=tune_list, columns=['batch_size', 'train_epoch', 'online_epoch', 'sigma'])
    df.to_csv(out_path+"para.txt")
    # np.savetxt(out_path+"para.txt", tune_list, fmt="%0.5f", delimiter=",")

    for i, (batch_size, train_epoch, online_epoch, sigma) in enumerate(tune_list, 1):

        print("="*50)
        print("epoch: {}".format(i))
        print("batch_size: {}".format(batch_size))
        print("train_epoch: {}".format(train_epoch))
        print("online_epoch: {}".format(online_epoch))
        print("sigma: {}".format(sigma))

        os.makedirs(out_path + str(i) + "/p", exist_ok=True)
        os.makedirs(out_path + str(i) + "/d", exist_ok=True)
        os.makedirs(out_path + str(i) + "/l", exist_ok=True)

        online(acc, Model, dataset_path, out_path + str(i) + "/", i, 
                batch_size=int(batch_size), train_epoch=int(train_epoch),  # 分類器のパラメータ
                online_epoch=int(online_epoch), sigma=sigma # オンライン予測のパラメータ
                )
    
tuning()