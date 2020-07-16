import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tensorflow as tf

# import dataset
# import model
from dataset import Random


if __name__ == '__main__':

    n = 100
    color_step = 100

    # color-map
    cm = plt.get_cmap("jet", color_step)

    # loss(この値に応じてcolorを変える)
    # loss = np.random.choice(color_step//2, n)
    # loss = np.array([Random(0, 20) for i in range(n)])
    loss = np.array([Random(0, 20) for i in range(n-1)])
    loss = np.append(loss, 50)

    # 正規化(min ~ max -> 0 ~ 1)
    loss_max, loss_min = np.max(loss), np.min(loss)
    loss = (loss - loss_min) / (loss_max - loss_min)
    print(loss)

    # lossを(0, 1, ..., color_step-1)の離散値に変換
    loss_digit = (loss // (1 / color_step)).astype(np.int)
    print(loss_digit)

    # loss_digitの値がcolor_stepのものがあればcolor_step-1に置き換え
    loss_digit = np.where(loss_digit == color_step, color_step-1, loss_digit)
    print(loss_digit)

    # 正解ラベル
    labels = np.array([int(i/int(n/2)) for i in range(n)])
    # データ
    dataset = np.zeros((n, 2))

    for i, label in enumerate(labels):

        # label 0
        if label == 0:
            dataset[i] = [Random(0, 1), Random(-1, 1)]

        # label 1
        else:
            dataset[i] = [Random(-1, 0), Random(-1, 1)]

    fig = plt.figure()
    ax  = fig.add_axes((0.1,0.3,0.8,0.6))

    for k, (i,j) in enumerate(zip(dataset[:, 0], dataset[:, 1])):
        ax.plot(i,j,'o', color=cm(loss[k]))

        if k < n //2:
            ax.annotate(0, xy=(i, j))

        else:
            ax.annotate(1, xy=(i, j))

    # 1d array
    gradient = np.linspace(0, 1, cm.N)
    # 2d array (for imshow)
    gradient_array = np.vstack((gradient, gradient))
    
    ax2 = fig.add_axes((0.1,0.1,0.8,0.05))
    ax2.imshow(gradient_array, aspect='auto', cmap=cm)
    ax2.set_axis_off()

    print("loss_max(Red): {}, loss_min(Blue): {}".format(loss_max, loss_min))

    plt.show()

