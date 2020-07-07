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
import matplotlib.pyplot as plt

def grid_plot(grid, output, fig_path, x, y):

    # gridをクラス分け
    index0 = np.where(output == 0)
    index1 = np.where(output == 1)

    label0 = "0"
    label1 = "1"

    grid0, grid1 = grid[index0], grid[index1]

    plt.plot(grid0[:, 0],grid0[:, 1],marker=".",linestyle="None",color="red", label=label0)
    plt.plot(grid1[:, 0],grid1[:, 1],marker=".",linestyle="None",color="blue", label=label1)

    # xをyでクラス分け
    ind0 = np.where(y == 0)
    ind1 = np.where(y == 1)
    x0, x1 = x[ind0], x[ind1]
 
    # plt.plot(x0[:, 0], x0[:, 1],marker=".",linestyle="None",color="white", label=label0)
    plt.plot(x1[:, 0], x1[:, 1],marker=".",linestyle="None",color="white", label=label1)

    # xy軸
    plt.plot([-1, 1], [0, 0], marker=".",color="black")
    plt.plot([0, 0], [-1, 1], marker=".",color="black")

    # 凡例の表示
    plt.legend()
    
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

def visualization(net, X, Y, path):

    grid_num = 100

    # 格子点を入力する準備
    grid_elem = np.linspace(-1, 1, grid_num)

    xx = np.array([[x for x in grid_elem] for _ in range(grid_num)])
    xx = xx.reshape(grid_num**2)
    yy = np.array([[y for _ in range(grid_num)] for y in grid_elem])
    yy = yy.reshape(grid_num**2)

    xy = np.stack([xx, yy])
    xy = xy.T
    # grid_input = np.array(xy[:, np.newaxis, :, np.newaxis], dtype="float32")
    grid_input = np.array(xy, dtype="float32")
    grid_input = torch.from_numpy(grid_input)

    # 格子点を入力にする
    grid_output = net(grid_input)
    _, grid_pred = torch.max(grid_output, 1)  # ラベルを予測

    # 可視化
    grid_plot(xy, grid_pred.cpu().numpy(), path+"result.png", X, Y)
