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
from model import SimpleNet2

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_prototype = SimpleNet2().to(device)    

def net_test(n_epoch, dataloader, optimizer, criterion):

    # ネットワークの重みを初期化
    model_prototype.apply(init_weights)

    model_prototype.train()
    #scheduler.step()
    correct = 0
    total = 0

    for i in range(n_epoch):
 
        for step, (images, labels) in enumerate((dataloader), 1):

            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model_prototype(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i == n_epoch-1:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    return correct/total
