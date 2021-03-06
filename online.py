import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from time import time

import dataset
import model
from visual import visualization, acc_plot, acc_plot2, init_visual, visualize_weights, display30, tSNE, tSNE2, visualization_multi
from net_test import net_test 

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
            
# def softmax(Llist):
#     exp_x = np.exp(Llist)    
#     y = exp_x / np.array([np.sum(exp_x,axis=1)]).T    
#     return np.max(y,axis=1)

# def softmax2(Llist):
#     exp_x = np.exp(Llist)    
#     y = exp_x / np.array([np.sum(exp_x,axis=1)]).T    
#     return y

# def softmax3(Llist, labels):
#     exp_x = np.exp(Llist)    
#     y = exp_x / np.array([np.sum(exp_x,axis=1)]).T
#     arr = [i for i in range(len(labels))]
#     return y[arr, labels]

def softmax(outputs):
    """
    outputsのクラス内で最大の尤度を返す

    outputs: (batch_size, class)
    p: (batch_size, )
    """

    p, _ = torch.max(F.softmax(outputs, dim=1), 1)  
    return p.to('cpu').detach().numpy().copy()

def softmax2(outputs):
    """
    outputsの尤度を返す

    outputs: (batch_size, class)
    p: (batch_size, class)
    """
    p = F.softmax(outputs, dim=1)
    return p.to('cpu').detach().numpy().copy()

def softmax3(outputs, labels):
    """
    outputsから選択したラベルの尤度を返す

    outputs: (batch_size, class)
    p: (batch_size, )
    """
    p = F.softmax(outputs, dim=1)
    p = p.gather(1, labels.reshape(len(labels), 1)).squeeze()
    return p.to('cpu').detach().numpy().copy()

def cal_loss_binary(model, dataloader, loss_type):
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
             
            p = softmax(outputs)
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

def cal_loss_multi(model, dataloader, loss_type):
    model.eval()
    correct = 0
    total = 0
    label_RightorWrong = []
    with torch.no_grad():

        p_list = []
        new_p_list = []
        # クラス数
        p_class_list = np.empty((0, 4))
        # p_class_list = np.empty((0, 10))
        
        for step, (images, labels) in enumerate(dataloader, 1):
            
            labelinf = np.zeros(len(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)

            # クラス内で最大の尤度
            p = softmax(outputs)     
            p_list.extend(p)

            # 尤度
            p_class = softmax2(outputs)
            p_class_list = np.append(p_class_list, p_class, axis=0)

            # 元の教師ラベルの尤度
            new_p = softmax3(outputs, labels)
            new_p_list.extend(new_p)

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
            # loss_list = 1 - np.array(p_list)
            loss_list = np.array(new_p_list)
    
    return np.array(loss_list), np.array(p_class_list), train_acc

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

def online(acc, Model, dataset_set, out_path, stage_no, 
            classfy_type, data_type, # 分類タイプ, データタイプ
            batch_size = 10, n_epochs = 10,  # 分類器のパラメータ
            n_rounds = 50, sigma=10**(-5), # オンライン予測のパラメータ
            reset_flag=False, loss_type = "loss1" # 学習リセットするか、損失の種類
            ):
    
    torch.manual_seed(1)

    # 入力データをロード
    x_train, y_train, dataloader_train, dataloader_test, train_dataset = dataset_set
    x_train = torch.from_numpy(x_train)

    # ネットワークの重みを初期化
    Model.apply(init_weights)

    # n: データ数
    n = len(x_train)
    # k: 改変するデータ数
    k = round(n * (1-acc))
    print(k)
    # eta: 今回固定
    eta = np.sqrt(8*np.log(n)/n_rounds)

    # パラメータ
    lr = 10**(-2)
    Criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(Model.parameters(), lr=lr)

    train_acc_ori_list = []
    train_acc_change_list = []
    test_acclist = []
    virtual_loss_list = np.empty((0, n))

    #============================================
    # 累積損失
    cumulative_loss = np.zeros(n)

    # 二クラス分類の場合、xtを適当に初期化
    if classfy_type == "binary":
        xt = np.array(random.sample(range(0,n),k=k))
    
    # 多クラス分類の場合、最初に学習させてlossによりxtを初期化
    elif classfy_type == "multi":
        # 学習
        for i in range(n_epochs):
            train(Model, dataloader_train, Optimizer, Criterion)
        # 損失を返す
        loss_list, p_list, train_acc_change = cal_loss_multi(Model, dataloader_train, loss_type)

        # 損失の小さいtop-k個を選択
        xt = np.argsort(loss_list)[:k]
        
    ### オンライン予測 ###
    for _round in range(1, n_rounds+1):
        if data_type == "mnist" and _round % 10 == 0:
            print("--- Round : %2d ---" % _round)
        xt = np.sort(xt)
        flip_y_train = np.copy(y_train)

        ### 改変パート ###
        # 損失の小さいtop-kをひっくり返す
        if classfy_type == "binary":
            flip_y_train[xt] = (y_train[xt] + 1) % 2

        # 多クラス分類の場合、元のラベルを除いて尤度最大のクラスを改変ラベルにする
        elif classfy_type == "multi":
            p_temp_list = np.copy(p_list)
            p_temp_list[[i for i in range(n)], y_train] = 0
            flip_y_train[xt] = np.argmax(p_temp_list[xt], axis=1)

        # dataset, dataloader作成
        ds_selected = data.TensorDataset(x_train, torch.from_numpy(flip_y_train))
        dataloader_fliped = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=False)

        # ネットワークの重みを初期化
        if reset_flag:
            Model.apply(init_weights)

        # 学習
        for i in range(n_epochs):
            train(Model, dataloader_fliped, Optimizer, Criterion)

        # 損失を返す
        if classfy_type == "binary":
            loss_list, p_list, train_acc_change = cal_loss_binary(Model, dataloader_fliped, loss_type)
        
        elif classfy_type == "multi":
            loss_list, p_list, train_acc_change = cal_loss_multi(Model, dataloader_fliped, loss_type)

        # 累積損失
        cumulative_loss = cumulative_loss + loss_list

        # 累積損失にガウス分布による乱数を足し算したものが損失
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

        # 各ラウンドごとの可視化(artifactの時のみ実行)
        if data_type == "artifact":
            # ニクラス分類
            if classfy_type == "binary":
                visualization(Model, x_train, flip_y_train, y_train, virtual_loss, _round, "d", out_path + "d/")
                # visualization(Model, x_train, flip_y_train, y_train, p_list, _round, "p", out_path + "p/")
                # visualization(Model, x_train, flip_y_train, y_train, loss_list, _round, "loss", out_path + "l/")

            # 多クラス分類
            elif classfy_type == "multi":
                visualization_multi(Model, x_train, flip_y_train, y_train, virtual_loss, _round, "d", out_path + "d/")

    # 学習曲線をプロット
    acc_plot(train_acc_ori_list, train_acc_change_list, test_acclist, acc, out_path, stage_no)


    # # データセットがartifact
    # if data_type == "artifact":
    #     # 重みの遷移を可視化
    #     visualize_weights(x_train, virtual_loss_list, stage_no, out_path+"../")

    # データセットがmnist
    if data_type == "mnist":
        # 損失の低かった上位30枚の画像を表示
        display30(train_dataset, np.argsort(virtual_loss)[:30], stage_no, out_path)

        # tSNEで可視化
        tSNE(x_train, y_train, xt, stage_no, out_path)
        tSNE2(x_train, y_train, xt, stage_no, out_path)
