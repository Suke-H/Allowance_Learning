import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tensorflow as tf

import dataset
import model


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    num_classes = 10
    batch_size = 100
    epochs = 50
    acc = 0.6
    train_n = 1000
    lr = 10**(-4)

    # データセット作成
    x_train, y_train = dataset.make_artificial_data(100, "train")
    # val_x, val_t = dataset.make_artificial_data(10, "val")
    # test_x, test_t = dataset.make_artificial_data(10, "test")

    n = len(x_train)
    k = int(n * (1-acc))
    eta = np.sqrt(8*np.log(n)/epochs)
    beta = np.exp(-eta)

    print(n, k)

    # Model = model.ResNet18().to(device)
    Model = model.SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Model.parameters(), lr=lr)

    train_acclist = []
    val_acclist = []
    test_acclist = []

    #===algorithm setting===

    #============================================
    # algorithm = "WAA"
    # w = np.array([1/n for i in range(n)])
    #============================================

    #============================================
    algorithm = "FPL"
    cumulative_loss = np.zeros(n)
    xt = np.array(random.sample(range(0,n),k=k))
    #============================================
    xt = np.sort(xt)

    print(xt)

    # xt(損失の小さいtop-k)をひっくり返す
    flip_y_train = np.copy(y_train)
    flip_y_train[xt] = (y_train[xt] + 1) % 2

    print(list(y_train))
    print(list(flip_y_train))
    
    # ds_selected = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(flip_y_train))
    # dataloader_selelected = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=True)
    # train(epoch)
    # loss_list, train_acc = cal_loss(epoch)
    # cumulative_loss = cumulative_loss + loss_list
    # perturbation = np.random.normal(0,0.00001,(n))
    # virtual_loss = cumulative_loss + eta*perturbation

    # # 損失の小さいtop-k個を選択
    # xt = np.argsort(virtual_loss)[:k]
    
    # val_acc = eval(epoch)
    # test_acc = test(epoch)
    
    # train_acclist.append(train_acc)
    # val_acclist.append(val_acc)
    # test_acclist.append(test_acc)