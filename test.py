import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import dataset
from model import SimpleNet2
from online import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vis(net, aabb, grid_num=100):
    """
    識別関数を可視化

    Attribute
    grid_num: AABBの範囲でグリットを刻む回数
    aabb: AABB(Axis Aligned Bounding Box)の座標[xmin, xmax, ymin, ymax]

    """

    ### 格子点を入力する準備
    x = np.linspace(aabb[0], aabb[1], grid_num)
    y = np.linspace(aabb[2], aabb[3], grid_num)
    xx, yy = np.meshgrid(x,y)
    xy = np.stack([xx, yy])
    xy = xy.T.reshape(grid_num**2, 2)

    # 入力データ
    grid_x = np.array(xy, dtype="float32")
    grid_x = torch.from_numpy(grid_x)

    # 格子点を入力にする
    grid_output = net(grid_x)
    # ラベルを予測
    _, grid_y = torch.max(grid_output, 1)  

    ### 識別関数 可視化
    # グリッドをクラス分け
    index0 = np.where(grid_y == 0)
    index1 = np.where(grid_y == 1)
    label0 = "0"
    label1 = "1"

    grid0, grid1 = grid_x[index0], grid_x[index1]

    plt.plot(grid0[:, 0], grid0[:, 1],marker=".",linestyle="None",color="lightgray", label=label0)
    plt.plot(grid1[:, 0], grid1[:, 1],marker=".",linestyle="None",color="gray", label=label1)

    # 凡例の表示
    plt.legend()

    plt.show()

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":

    Model_origin = SimpleNet2().to(device)
    
    torch.manual_seed(1)

    dataset_path = "data/artifact/"
    x_train, y_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)

    # dataset, dataloader作成
    ds_selected = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    dataloader_train = data.DataLoader(dataset=ds_selected, batch_size=25, shuffle=False)

    Model = copy.deepcopy(Model_origin)

    # n: データ数
    n = len(x_train)

    train_epoch = 20

    # 分類器のパラメータ(固定)
    lr = 10**(-2)
    Criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(Model.parameters(), lr=lr)
    Optimizer = optim.Adam(Model.parameters(), lr=lr)

    aabb = [np.min(x_train[:, 0]), np.max(x_train[:, 0]), np.min(x_train[:, 1]), np.max(x_train[:, 1])]
    
    for i in range(train_epoch):
        # 5回ごとに学習リセット
        if i % 5 == 0:
            print("Reset")
            # Model = copy.deepcopy(Model_origin)
            # Model = SimpleNet2().to(device)
            Model.apply(init_weights)

        # 学習
        train(Model, dataloader_train, Optimizer, Criterion)
        # 可視化
        vis(Model, aabb)
