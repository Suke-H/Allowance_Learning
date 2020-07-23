import torch
import itertools
import os
import numpy as np

from online import online
from model import SimpleNet2


def tuning():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc = 0.75
    Model = SimpleNet2().to(device)
    dataset_path = "data/artifact/"
    out_path = "data/result/tuning/test/"

    batch_size_list = [10]
    train_epoch_list = [10]
    online_epoch_list = [50]
    sigma_list = [10**(-5)]

    tune_list = list(itertools.product(batch_size_list, train_epoch_list, online_epoch_list, sigma_list))
    print(tune_list)

    np.savetxt(out_path+"para.txt", tune_list, fmt="%0.5f", delimiter=",")

    for i, (batch_size, train_epoch, online_epoch, sigma) in enumerate(tune_list, 1):

        print("epoch {}".format(i))

        os.makedirs(out_path + str(i) + "/p", exist_ok=True)
        os.makedirs(out_path + str(i) + "/d", exist_ok=True)
        os.makedirs(out_path + str(i) + "/l", exist_ok=True)

        online(acc, Model, dataset_path, out_path + str(i) + "/", i, 
                batch_size=batch_size, train_epoch=train_epoch,  # 分類器のパラメータ
                online_epoch=online_epoch, sigma=sigma # オンライン予測のパラメータ
                )
    
tuning()