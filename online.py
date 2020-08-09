import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
#import torchvision.models as models
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

import dataset
import model
from visual import visualization, acc_plot, init_visual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, criterion):
    model.train()
    #scheduler.step()
    correct = 0
    total = 0
 
    for step, (images, labels) in enumerate((dataloader), 1):

        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
 
    # print("Train Acc : %.4f" % (correct/total))
            
def softmax(Llist):
    exp_x = np.exp(Llist)    
    y = exp_x / np.array([np.sum(exp_x,axis=1)]).T    
    return np.max(y,axis=1)

def cal_loss(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    label_RightorWrong = []
    with torch.no_grad():
        p_list = []
        for step, (images, labels) in enumerate(dataloader, 1):
            
            labelinf = np.zeros(len(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)
            
            p = softmax([list(map(float,i)) for i in list(outputs)])         
            p_list.extend(p)
            
            prelist = list(map(int,predicted))
            lablist = list(map(int,labels))
            
            labelinf[np.where(np.array(prelist) == np.array(lablist))[0]] = 1
            labelinf[np.where(np.array(prelist) != np.array(lablist))[0]] = -1
            
            label_RightorWrong.extend(labelinf)

        train_acc = len(np.where(np.array(label_RightorWrong)==1)[0])/len(label_RightorWrong)
        # loss_list = (1 - np.array(label_RightorWrong)*np.array(p_list))/2
        loss_list = np.array(p_list)
    
    return np.array(loss_list), np.array(p_list), train_acc

def eval(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
 
    # print("Val Acc : %.4f" % (correct/total))
    
    return correct/total

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
 
    # print("Test Acc : %.4f" % (correct/total))
    
    return correct/total

def online(acc, Model, dataset_path, out_path, tune_epoch, 
            batch_size = 10, train_epoch = 10,  # 分類器のパラメータ
            online_epoch = 50, sigma=10**(-5) # オンライン予測のパラメータ
            ):
    
    torch.manual_seed(1)

    x_train, y_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)

    print(len(x_train))

    init_visual(x_train, y_train, out_path)

    # n: データ数
    n = len(x_train)
    # k: 改変するデータ数
    k = int(n * (1-acc))
    # eta: 今回固定
    eta = np.sqrt(8*np.log(n)/online_epoch)

    # 分類器のパラメータ(固定)
    lr = 10**(-2)
    Criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(Model.parameters(), lr=lr)
    Optimizer = optim.Adam(Model.parameters(), lr=lr)

    train_acclist = []
    val_acclist = []
    test_acclist = []

    virtual_loss_list = np.empty((0, n))

    #===algorithm setting===

    #============================================
    # algorithm = "WAA"
    # w = np.array([1/n for i in range(n)])
    #============================================

    #============================================
    algorithm = "FPL"
    # 累積損失
    cumulative_loss = np.zeros(n)
    # データのインデックスから損失の小さいtop-kを順に並べたもの
    xt = np.array(random.sample(range(0,n),k=k))
    #============================================
        
    if algorithm == "FPL":
        for epoch in range(1, online_epoch+1):
            # print("\n--- Epoch : %2d ---" % epoch)
            xt = np.sort(xt)

            # 損失の小さいtop-kをひっくり返す
            flip_y_train = np.copy(y_train)
            flip_y_train[xt] = (y_train[xt] + 1) % 2
            
            # dataset, dataloader作成
            ds_selected = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(flip_y_train))
            dataloader_fliped = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=False)

            # 学習
            for i in range(train_epoch):
                train(Model, dataloader_fliped, Optimizer, Criterion)

            # 損失を返す
            loss_list, p_list, train_acc = cal_loss(Model, dataloader_fliped)

            # 累積損失
            cumulative_loss = cumulative_loss + loss_list

            # 累積損失にガウス分布により乱数を足し算したものが損失
            perturbation = np.random.normal(0, sigma, (n))
            virtual_loss = cumulative_loss + eta*perturbation

            # 損失の小さいtop-k個を選択
            xt = np.argsort(virtual_loss)[:k]
        
            val_acc = eval(Model, dataloader_val)
            test_acc = test(Model, dataloader_test)
            
            train_acclist.append(train_acc)
            val_acclist.append(val_acc)
            test_acclist.append(test_acc)

            virtual_loss_list = np.append(virtual_loss_list, virtual_loss)

            # 可視化
            # visualization(Model, x_train, flip_y_train, virtual_loss, epoch, "data/result/try1/d/")
            # visualization(Model, x_train, flip_y_train, p_list, epoch, "data/result/try1/p/")
            # visualization(Model, x_train, flip_y_train, loss_list, epoch, "data/result/try1/l/")

            visualization(Model, x_train, flip_y_train, y_train, virtual_loss, epoch, "d", out_path + "d/")
            visualization(Model, x_train, flip_y_train, y_train, p_list, epoch, "p", out_path + "p/")
            visualization(Model, x_train, flip_y_train, y_train, loss_list, epoch, "loss", out_path + "l/")
    
    acc_plot(train_acclist, val_acclist, test_acclist, acc, out_path+"../", tune_epoch)