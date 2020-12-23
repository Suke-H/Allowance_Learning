# パッケージのimport
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import dataset
from online import cal_loss_binary, init_weights
from visual import visualization, visualize_weights

def stop_algo(limit_acc, net, dataset_tuple, root_path, file_no,
    num_epochs=50, batch_size=10, limit_phase="val", loss_type="p"):

    """
    trainかval accが目標accを超えたら学習を停止するアルゴリズム
    """

    # ネットワークの重みを初期化
    net.apply(init_weights)

    # 学習結果の保存用
    history = {
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }

    # 入力データをロード
    x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset_tuple

    dataloaders_dict = {"train": dataloader_train, 
                        "val": dataloader_val, 
                        "test": dataloader_test}

    # パラメータ
    lr = 10**(-2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    n = len(x_train)
    # eta: 今回固定
    eta = np.sqrt(8*np.log(n)/num_epochs)
    sigma=10**(-4)
    # 停止フラグ
    stop_flag = False
    # 累積損失
    cumulative_loss = np.zeros(n)

    virtual_loss_list = np.empty((0, n))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとに学習とバリデーション
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに
            # net.eval()

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for i, (inputs, labels) in enumerate(dataloaders_dict[phase]):

                inputs, labels = inputs.to(device), labels.to(device)
                _batch_size = inputs.size(0)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測
                    
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * _batch_size 
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)
                   
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            # historyに保存
            history[phase + '_acc'].append(epoch_acc.item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 上限accを越したら学習停止
            if epoch_acc >= limit_acc and phase == limit_phase:
                print("{}_acc : {} >= {}\nstop at {} epoch!".format(phase, epoch_acc, limit_acc, epoch+1))
                stop_flag = True

        # pを可視化
        loss_list, p_list, _ = cal_loss_binary(net, dataloader_train, loss_type)

        # 累積損失
        cumulative_loss = cumulative_loss + loss_list

        # 累積損失にガウス分布により乱数を足し算したものが損失
        perturbation = np.random.normal(0, sigma, (n))
        virtual_loss = cumulative_loss + eta*perturbation

        # 各ラウンドごとの累積損失を保存
        virtual_loss_list = np.append(virtual_loss_list, virtual_loss.reshape(1, n), axis=0)

        visualization(net, x_train, y_train, y_train, p_list, epoch+1, "p", root_path + "p/")
        visualization(net, x_train, y_train, y_train, loss_list, epoch+1, "l", root_path + "l/")
        visualization(net, x_train, y_train, y_train, virtual_loss, epoch+1, "d", root_path + "d/")

        # 停止フラグが立ったら終了
        if stop_flag:
            break

    end_epoch = epoch + 1

    if stop_flag == False:
        print("learning_epoch has reached the upper_limit_epoch {} before {}_acc reached the upper_limit_acc {} ...".format(end_epoch, phase, limit_acc))

    # 学習記録(epoch-acc)をプロット
    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, history['train_acc'], label='train_acc')
    plt.plot(runs, history['val_acc'], label='val_acc')
    plt.plot(runs, [limit_acc for i in range(end_epoch)], label='limit_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(runs)
    plt.legend()
    plt.savefig(root_path+"../acc_"+str(file_no)+".png")

    # 重みの遷移を可視化
    visualize_weights(x_train, virtual_loss_list, file_no, root_path+"../")
