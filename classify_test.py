import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import dataset
import model
from visual import visualization_test, acc_plot_test
from main import train, eval, test, cal_loss

if __name__ == '__main__':
    
    torch.manual_seed(1)

    # パラメータ
    epochs = 50
    lr = 10**(-2)

    # Model = model.SimpleNet().to(device)
    Model = model.SimpleNet2().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(Model.parameters(), lr=lr)
    optimizer = optim.Adam(Model.parameters(), lr=lr)

    # x_train, y_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset("data/artifact/")
    dataloader_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset("data/test_arti/")

    train_acclist = []
    val_acclist = []
    test_acclist = []

    for epoch in range(1, epochs+1):

        train(Model, dataloader_train, optimizer, criterion)
        _, train_acc = cal_loss(Model, dataloader_train)
        val_acc = eval(Model, dataloader_val)
        test_acc = test(Model, dataloader_test)
        
        train_acclist.append(train_acc)
        val_acclist.append(val_acc)
        test_acclist.append(test_acc)

        # 可視化
        # visualization(Model, x_train[:100], flip_y_train[:100], virtual_loss[:100], epoch, "data/result/try6/")
        visualization_test(Model, epoch, "data/result/test_classify/1/")
        
    acc_plot_test(train_acclist, val_acclist, test_acclist, "data/result/test_classify/", 1)
