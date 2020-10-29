import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
#import torchvision.model_prototypes as model_prototypes
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from time import time

import dataset
from model import SimpleNet2, SimpleNet2_2, SimpleNet2_3
from visual import visualization

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_prototype = SimpleNet2_3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_prototype.parameters(), lr=10**(-2))    

def net_test(n_epoch, dataloader):

    # ネットワークの重みを初期化
    model_prototype.apply(init_weights)

    model_prototype.train()
    #scheduler.step()
    # correct = 0
    # total = 0

    for i in range(n_epoch):

        correct = 0
        total = 0
 
        for step, (images, labels) in enumerate((dataloader), 1):

            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model_prototype(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if i == n_epoch-1:
            #     _, predicted = torch.max(outputs.data, 1)
            #     correct += (predicted == labels).sum().item()
            #     total += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # print(total, correct/total)

    return correct/total, model_prototype


# dataset_path = "data/dataset/fuzzy_1000_1/"
# x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)
# print(net_test(50, dataloader_train, Optimizer, Criterion))