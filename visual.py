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

def visualize_classify(ax1, net, grid_num=100):
    """
    識別関数を可視化

    Attribute
    grid_num: [0,1]の範囲でグリットを刻む回数

    """

    ### 格子点を入力する準備
    grid_elem = np.linspace(-1, 1, grid_num)

    xx = np.array([[x for x in grid_elem] for _ in range(grid_num)])
    xx = xx.reshape(grid_num**2)
    yy = np.array([[y for _ in range(grid_num)] for y in grid_elem])
    yy = yy.reshape(grid_num**2)

    xy = np.stack([xx, yy])
    xy = xy.T

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

    ax1.plot(grid0[:, 0],grid0[:, 1],marker=".",linestyle="None",color="lightgray", label=label0)
    ax1.plot(grid1[:, 0],grid1[:, 1],marker=".",linestyle="None",color="gray", label=label1)

    # xy軸
    ax1.plot([-1, 1], [0, 0], marker=".",color="black")
    ax1.plot([0, 0], [-1, 1], marker=".",color="black")

    # 凡例の表示
    ax1.legend()



def visualize_allowance(ax1, ax2, x, y, loss, color_step=100, cmap_type="jet"):
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """

    n = len(x)

    # color-map
    cm = plt.get_cmap(cmap_type, color_step)

    # lossを正規化(min ~ max -> 0 ~ 1)
    loss_max, loss_min = np.max(loss), np.min(loss)
    loss = (loss - loss_min) / (loss_max - loss_min)

    # lossを(0, 1, ..., color_step-1)の離散値に変換
    loss_digit = (loss // (1 / color_step)).astype(np.int)

    # loss_digitの値がcolor_stepのものがあればcolor_step-1に置き換え
    loss_digit = np.where(loss_digit == color_step, color_step-1, loss_digit)

    # プロット
    # for k, (i,j) in enumerate(zip(x[:, 0], x[:, 1])):
    #     ax.plot(i,j,'o', color=cm(loss[k]))
    #     ax.annotate(y[k], xy=(i, j))

    true_y = np.where(x[:, 0] >= 0, 0, 1)

    for i in range(n):
        ax1.plot(x[i, 0],x[i, 1],'o', color=cm(loss[i]))

        if true_y[i] == y[i]:
            ax1.annotate(y[i], xy=(x[i, 0],x[i, 1]))
        else:
            ax1.annotate(y[i], xy=(x[i, 0],x[i, 1]), color="red")

    # color-barを表示
    gradient = np.linspace(0, 1, cm.N)
    gradient_array = np.vstack((gradient, gradient))
    
    ax2.imshow(gradient_array, aspect='auto', cmap=cm)
    ax2.set_axis_off()
 
    ax2.set_title("loss_min(Blue): {:.2f}  ~   loss_max(Red): {:.2f}".format(loss_min, loss_max))

def visualization(net, x, y, loss, epoch, path):
    """
    Attribute

    net: 学習モデル
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    epoch: 何エポック目での可視化か
    path: 保存するフォルダのパス

    """

    # グラフ作成
    fig = plt.figure()
    ax1  = fig.add_axes((0.1,0.3,0.8,0.6))
    ax2 = fig.add_axes((0.1,0.1,0.8,0.05))

    # 識別関数の可視化
    visualize_classify(ax1, net)

    # 斟酌の可視化
    visualize_allowance(ax1, ax2, x, y, loss)

    # plt.show()
    plt.savefig(path + str(epoch) + ".png")
    plt.close()
