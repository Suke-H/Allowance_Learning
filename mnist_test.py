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
from visual import acc_plot
from online import train, eval, cal_loss

if __name__ == '__main__':
    
    torch.manual_seed(1)

    # パラメータ
    epochs = 2
    lr = 10**(-2)

    # Model = model.SimpleNet().to(device)
    Model = model.MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(Model.parameters(), lr=lr)
    optimizer = optim.Adam(Model.parameters(), lr=lr)

    # x_train, y_train, dataloader_train, dataloader_test = dataset.make_and_load_artifical_dataset(1000, 1)
    x_train, y_train, dataloader_train, dataloader_test = dataset.MNIST_load()

    train_acclist = []
    test_acclist = []

    for epoch in range(1, epochs+1):

        train(Model, dataloader_train, optimizer, criterion)
        loss_list, p_list, train_acc = cal_loss(Model, dataloader_train, "p")
        test_acc = eval(Model, dataloader_test)
        
        train_acclist.append(train_acc)
        test_acclist.append(test_acc)

        print(train_acc, test_acc)
        
    # 学習曲線をプロット
    acc_plot(train_acclist, test_acclist, test_acclist, 0.7, "data/result/tuning/mnist/test/2/", epochs)