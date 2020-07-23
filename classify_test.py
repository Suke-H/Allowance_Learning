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
from visual import visualization, acc_plot_test
from online import train, eval, test, cal_loss

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

    x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset("data/fuzzy_data_u1_b10/")

    train_acclist = []
    val_acclist = []
    test_acclist = []

    for epoch in range(1, epochs+1):

        train(Model, dataloader_train, optimizer, criterion)
        loss_list, p_list, train_acc = cal_loss(Model, dataloader_train)
        val_acc = eval(Model, dataloader_val)
        test_acc = test(Model, dataloader_test)
        
        train_acclist.append(train_acc)
        val_acclist.append(val_acc)
        test_acclist.append(test_acc)

        # 可視化
        # visualization_test(Model, epoch, "data/result/fuzzy/1/")
        # visualization(Model, x_train, flip_y_train, virtual_loss, epoch, "data/result/fuzzy/1/d/")
        visualization(Model, x_train, y_train, p_list, epoch, "data/result/fuzzy/u1_b10/p/")
        visualization(Model, x_train, y_train, loss_list, epoch, "data/result/fuzzy/u1_b10/l/")
        
    acc_plot_test(train_acclist, val_acclist, test_acclist, "data/result/fuzzy/u1_b10/", 1)
