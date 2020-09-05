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
from time import time

import dataset
import model
from visual import visualization, acc_plot, init_visual, visualize_weights

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

def cal_loss(model, dataloader, loss_type):
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

        if loss_type == "loss1":
            loss_list = (1 - np.array(label_RightorWrong)*np.array(p_list))/2

        elif loss_type == "p":
            loss_list = np.array(p_list)

        elif loss_type == "1-p":
            loss_list = 1 - np.array(p_list)
    
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

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def online(acc, Model, dataset_path, data_para, out_path, tune_epoch, 
            batch_size = 10, train_epoch = 10,  # 分類器のパラメータ
            online_epoch = 50, sigma=10**(-5), # オンライン予測のパラメータ
            reset_flag=False, loss_type = "loss1" # 学習リセットするか、損失の種類
            ):
    
    torch.manual_seed(1)

    # 入力データをロード
    # x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)
    x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset.make_and_load_artifical_dataset(data_para[0], data_para[1])

    # 入力データを可視化
    init_visual(x_train, y_train, out_path)

    # ネットワークの重みを初期化
    Model.apply(init_weights)

    # n: データ数
    n = len(x_train)
    # k: 改変するデータ数
    k = int(n * (1-acc))
    # eta: 今回固定
    eta = np.sqrt(8*np.log(n)/online_epoch)

    # パラメータ
    lr = 10**(-2)
    Criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(Model.parameters(), lr=lr)

    train_acc_ori_list = []
    train_acc_change_list = []
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

            start = time()

            # print("\n--- Epoch : %2d ---" % epoch)
            xt = np.sort(xt)

            # 損失の小さいtop-kをひっくり返す
            flip_y_train = np.copy(y_train)
            flip_y_train[xt] = (y_train[xt] + 1) % 2
            
            # dataset, dataloader作成
            ds_selected = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(flip_y_train))
            dataloader_fliped = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=False)

            # ネットワークの重みを初期化
            if reset_flag:
                Model.apply(init_weights)

            # 学習
            for i in range(train_epoch):
                train(Model, dataloader_fliped, Optimizer, Criterion)

            # 損失を返す
            loss_list, p_list, train_acc_change = cal_loss(Model, dataloader_fliped, loss_type)

            # 累積損失
            cumulative_loss = cumulative_loss + loss_list

            # 累積損失にガウス分布により乱数を足し算したものが損失
            perturbation = np.random.normal(0, sigma, (n))
            virtual_loss = cumulative_loss + eta*perturbation

            # 損失の小さいtop-k個を選択
            xt = np.argsort(virtual_loss)[:k]

            # 改変前の訓練データのacc
            train_acc_ori = eval(Model, dataloader_train)
            # テストデータのacc
            test_acc = eval(Model, dataloader_test)
            
            # 各ラウンドごとのaccを保存
            train_acc_ori_list.append(train_acc_ori)
            train_acc_change_list.append(train_acc_change)
            test_acclist.append(test_acc)

            # 各ラウンドごとの累積損失を保存
            virtual_loss_list = np.append(virtual_loss_list, virtual_loss.reshape(1, n), axis=0)

            # 可視化
            # visualization(Model, x_train, flip_y_train, virtual_loss, epoch, "data/result/try1/d/")
            # visualization(Model, x_train, flip_y_train, p_list, epoch, "data/result/try1/p/")
            # visualization(Model, x_train, flip_y_train, loss_list, epoch, "data/result/try1/l/")
            visualization(Model, x_train, flip_y_train, y_train, virtual_loss, epoch, "d", out_path + "d/")
            visualization(Model, x_train, flip_y_train, y_train, p_list, epoch, "p", out_path + "p/")
            visualization(Model, x_train, flip_y_train, y_train, loss_list, epoch, "loss", out_path + "l/")
    
    # 学習曲線をプロット
    acc_plot(train_acc_ori_list, train_acc_change_list, test_acclist, acc, out_path+"../", tune_epoch)

    # 重みの遷移を可視化
    visualize_weights(x_train, virtual_loss_list, tune_epoch, out_path+"../")